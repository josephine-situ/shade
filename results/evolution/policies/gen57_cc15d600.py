from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-max shade corridors v2"
DESCRIPTION = (
    "Maximise heat_relief_c by saturating highest heat×population corridors "
    "with shade. Strategy: (1) Medium trees at 1-pixel spacing on top cubic-synergy "
    "pixels (72% budget) — maximises crown overlap and UTCI drop in hottest zones; "
    "(2) Shade canopies on remaining open buildable ground (23% budget) targeting "
    "priority tracts and hot areas not reachable by trees; "
    "(3) Small trees gap-fill remaining plantable pixels (5% budget). "
    "Priority surface uses (heat×pop)^1.5 cubic synergy to strongly concentrate "
    "budget on hot+dense co-occurrence pixels. Priority-tract boost 0.35 "
    "maintains equity_ratio > 1. No reflective surfaces (albedo trap avoided)."
)

PRIORITY_BOOST = 0.35

FRAC_MED    = 0.72
FRAC_CANOPY = 0.23
FRAC_SML    = 0.05

SPACING_MED_PX = 1
SPACING_SML_PX = 1

CROWN_MED_M = 4.0
CROWN_SML_M = 2.0


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0,1] over mask; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Cubic-synergy composite priority. (heat×pop)^1.5 strongly concentrates
    placement on pixels that are simultaneously very hot AND very populated,
    maximising population-weighted UTCI drop per tree.
    """
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    synergy_cubic = np.power(np.clip(heat_n * pop_n, 0.0, 1.0), 1.5)
    synergy_sqrt  = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.45 * synergy_cubic
        + 0.20 * synergy_sqrt
        + 0.15 * heat_n
        + 0.08 * pop_n
        + 0.07 * vuln_n
        + 0.03 * hours_n
        + 0.02 * uhii_n
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def canopy_priority_surface(ctx: PlanningContext) -> np.ndarray:
    """Priority for shade canopy placement — same cubic synergy, canopy-tuned."""
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)

    synergy_cubic = np.power(np.clip(heat_n * pop_n, 0.0, 1.0), 1.5)
    synergy_sqrt  = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.50 * synergy_cubic
        + 0.20 * synergy_sqrt
        + 0.12 * heat_n
        + 0.08 * pop_n
        + 0.07 * vuln_n
        + 0.03 * hours_n
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy spaced selection: pick highest-scoring candidate, suppress
    spacing_px neighbourhood, repeat to `limit`.
    """
    rows, cols = np.nonzero(candidates)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    order = np.argsort(-score[rows, cols])
    rows, cols = rows[order], cols[order]

    taken = np.zeros(score.shape, dtype=bool)
    span  = max(int(spacing_px), 1)
    pick_r: list[int] = []
    pick_c: list[int] = []
    H, W = score.shape

    for r, c in zip(rows, cols):
        if taken[r, c]:
            continue
        pick_r.append(int(r))
        pick_c.append(int(c))
        r0 = max(0, r - span);  r1 = min(H, r + span + 1)
        c0 = max(0, c - span);  c1 = min(W, c + span + 1)
        taken[r0:r1, c0:c1] = True
        if len(pick_r) >= limit:
            break

    return np.array(pick_r, dtype=int), np.array(pick_c, dtype=int)


def _top_pixels(
    score: np.ndarray,
    candidates: np.ndarray,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select top `limit` pixels by score with no spacing constraint."""
    rows, cols = np.nonzero(candidates)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    order = np.argsort(-score[rows, cols])[:limit]
    return rows[order], cols[order]


def _stamp_exclusion(
    covered: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    radius_px: int,
    shape: tuple[int, int],
) -> None:
    """Mark radius_px box around each (r,c) as covered."""
    H, W = shape
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        r0 = max(0, r - radius_px);  r1 = min(H, r + radius_px + 1)
        c0 = max(0, c - radius_px);  c1 = min(W, c + radius_px + 1)
        covered[r0:r1, c0:c1] = True


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    tree_score   = priority_surface(ctx)
    canopy_score = canopy_priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0

    used    = np.zeros(ctx.shape, dtype=bool)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # pre-existing canopy

    # ── 1. Medium street trees (72% budget, 1-px spacing) ────────────────
    med_budget = budget_usd * FRAC_MED
    n_med_max  = ctx.affordable("tree_medium", med_budget)
    cand_med   = ctx.plantable & ~used
    n_med      = min(n_med_max, int(cand_med.sum()))

    tr_m, tc_m = _greedy_spaced(tree_score, cand_med, SPACING_MED_PX, n_med)

    if tr_m.size:
        placements.append(Placement("tree_medium", tr_m, tc_m))
        spent += ctx.cost("tree_medium", tr_m.size)
        used[tr_m, tc_m] = True
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_exclusion(covered, tr_m, tc_m, crown_med_px, ctx.shape)

    # ── 2. Shade canopies (23% budget, hottest open buildable) ───────────
    canopy_budget = min(budget_usd * FRAC_CANOPY, budget_usd - spent)
    if canopy_budget >= ctx.unit_cost("shade_canopy"):
        open_ground  = ctx.buildable & ~covered & ~used
        n_canopy_max = ctx.affordable("shade_canopy", canopy_budget)
        n_canopy     = min(n_canopy_max, int(open_ground.sum()))

        cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
        if cr.size:
            actual_cost = ctx.cost("shade_canopy", cr.size)
            if spent + actual_cost <= budget_usd + 0.01:
                placements.append(Placement("shade_canopy", cr, cc))
                spent += actual_cost
                used[cr, cc] = True
            else:
                affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
                if affordable_n > 0:
                    cr2, cc2 = cr[:affordable_n], cc[:affordable_n]
                    placements.append(Placement("shade_canopy", cr2, cc2))
                    spent += ctx.cost("shade_canopy", affordable_n)
                    used[cr2, cc2] = True

    # ── 3. Small trees (remaining budget, gap-fill uncovered plantable) ───
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("tree_small"):
        n_sml_max = ctx.affordable("tree_small", remaining)
        # Only plant where not already covered by canopy/trees
        cand_sml  = ctx.plantable & ~used & ~covered
        n_sml     = min(n_sml_max, int(cand_sml.sum()))

        tr_s, tc_s = _greedy_spaced(tree_score, cand_sml, SPACING_SML_PX, n_sml)
        if tr_s.size:
            actual_cost = ctx.cost("tree_small", tr_s.size)
            if spent + actual_cost <= budget_usd + 0.01:
                placements.append(Placement("tree_small", tr_s, tc_s))
            else:
                affordable_n = ctx.affordable("tree_small", budget_usd - spent)
                if affordable_n > 0:
                    placements.append(
                        Placement(
                            "tree_small",
                            tr_s[:affordable_n],
                            tc_s[:affordable_n],
                        )
                    )

    return placements