from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "dense-medium-tree synergy equity v2"
DESCRIPTION = (
    "Improve cell (1,0,1,0): medium trees at 3m spacing (denser than parent 4m) "
    "with strongly population-weighted synergy scoring. Budget split: 65% medium "
    "trees at 3m spacing, 15% small trees at 2m gap-fill, 20% shade canopies on "
    "remaining hot open ground. Priority surface uses geometric mean of heat×pop "
    "(0.45 synergy) + 0.25 heat_ta3pm + 0.12 population + 0.10 vulnerability + "
    "0.05 heat_hours + 0.03 UHII. Moderate priority-tract boost (0.25) maintains "
    "equity_ratio >= 1. All albedo actions avoided. Dense 3m spacing maximises "
    "per-corridor canopy fraction, boosting population-weighted UTCI relief."
)

PRIORITY_BOOST = 0.25

FRAC_MED    = 0.65
FRAC_SML    = 0.15
FRAC_CANOPY = 0.20

SPACING_MED_M = 3.0   # denser than parent's 4m
SPACING_SML_M = 2.0

CROWN_MED_M = 3.5
CROWN_SML_M = 2.0


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1] over mask; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def _priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Synergy-based priority surface maximizing person-degC relief.
    Geometric mean of heat × population ensures targeting of pixels that
    are BOTH hot AND populated — directly aligned with heat_relief_c fitness.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    # Geometric mean: high only when BOTH heat and population are high
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.45 * synergy
        + 0.25 * heat_n
        + 0.12 * pop_n
        + 0.10 * vuln_n
        + 0.05 * hours_n
        + 0.03 * uhii_n
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _canopy_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for shade canopy placement.
    Slightly higher vulnerability weight to target equity areas.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.42 * synergy
        + 0.23 * heat_n
        + 0.15 * vuln_n
        + 0.12 * pop_n
        + 0.05 * hours_n
        + 0.03 * uhii_n
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
    Greedy spaced placement: pick highest-scoring candidate pixel,
    suppress spacing_px-radius neighbourhood, repeat until limit reached.
    """
    rows, cols = np.nonzero(candidates)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    order = np.argsort(-score[rows, cols])
    rows, cols = rows[order], cols[order]

    taken = np.zeros(score.shape, dtype=bool)
    span = max(int(spacing_px), 1)
    pick_r: list[int] = []
    pick_c: list[int] = []
    H, W = score.shape

    for r, c in zip(rows, cols):
        if taken[r, c]:
            continue
        pick_r.append(int(r))
        pick_c.append(int(c))
        r0 = max(0, r - span)
        r1 = min(H, r + span + 1)
        c0 = max(0, c - span)
        c1 = min(W, c + span + 1)
        taken[r0:r1, c0:c1] = True
        if len(pick_r) >= limit:
            break

    return np.array(pick_r, dtype=int), np.array(pick_c, dtype=int)


def _top_pixels(
    score: np.ndarray,
    candidates: np.ndarray,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Select top `limit` candidate pixels by score, no spacing constraint."""
    rows, cols = np.nonzero(candidates)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    order = np.argsort(-score[rows, cols])[:limit]
    return rows[order], cols[order]


def _stamp_crown(
    covered: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    radius_px: int,
    shape: tuple[int, int],
) -> None:
    """Mark a box of radius_px around each (r, c) as covered."""
    H, W = shape
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        r0 = max(0, r - radius_px)
        r1 = min(H, r + radius_px + 1)
        c0 = max(0, c - radius_px)
        c1 = min(W, c + radius_px + 1)
        covered[r0:r1, c0:c1] = True


def _safe_place(
    ctx: PlanningContext,
    action: str,
    rows: np.ndarray,
    cols: np.ndarray,
    spent: float,
    budget_usd: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Trim placement to fit within remaining budget, return updated spent."""
    if rows.size == 0:
        return rows, cols, spent
    remaining = budget_usd - spent
    affordable_n = ctx.affordable(action, remaining)
    if affordable_n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), spent
    rows = rows[:affordable_n]
    cols = cols[:affordable_n]
    spent += ctx.cost(action, rows.size)
    return rows, cols, spent


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score        = _priority_surface(ctx)
    canopy_score = _canopy_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0

    used    = np.zeros(ctx.shape, dtype=bool)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True  # pre-existing canopy

    # ── Phase 1: Medium street trees (65% budget, 3m spacing) ────────────
    # 3m spacing (vs parent's 4m) packs ~78% more trees per corridor length,
    # maximising shaded pedestrian area and population-weighted UTCI drop.
    med_budget = budget_usd * FRAC_MED
    n_med_max  = ctx.affordable("tree_medium", med_budget)
    cand_med   = ctx.plantable & ~used

    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    n_med = min(n_med_max, int(cand_med.sum()))

    mr, mc = _greedy_spaced(score, cand_med, spacing_med_px, n_med)
    if mr.size:
        mr, mc, spent = _safe_place(ctx, "tree_medium", mr, mc, spent, budget_usd)
        if mr.size:
            placements.append(Placement("tree_medium", mr, mc))
            used[mr, mc] = True
            crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
            _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)

    # ── Phase 2: Small street trees (15% budget, 2m spacing, gap-fill) ───
    # Fill narrow gaps in tree corridors where medium trees can't fit.
    sml_budget = min(budget_usd * FRAC_SML, budget_usd - spent)
    sr = sc = np.array([], dtype=int)
    if sml_budget >= ctx.unit_cost("tree_small"):
        n_sml_max  = ctx.affordable("tree_small", sml_budget)
        # Only on plantable ground not yet used or under existing canopy
        cand_sml   = ctx.plantable & ~used

        spacing_sml_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)
        n_sml = min(n_sml_max, int(cand_sml.sum()))

        sr, sc = _greedy_spaced(score, cand_sml, spacing_sml_px, n_sml)
        if sr.size:
            sr, sc, spent = _safe_place(ctx, "tree_small", sr, sc, spent, budget_usd)
            if sr.size:
                placements.append(Placement("tree_small", sr, sc))
                used[sr, sc] = True
                crown_sml_px = max(int(round(CROWN_SML_M / ctx.res_m)), 1)
                _stamp_crown(covered, sr, sc, crown_sml_px, ctx.shape)

    # ── Phase 3: Shade canopies (remaining ~20% budget) ───────────────────
    # Target open hot buildable ground not already shaded by trees.
    remaining = budget_usd - spent
    if remaining < ctx.unit_cost("shade_canopy"):
        return placements

    open_ground  = ctx.buildable & ~covered & ~used
    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
    if cr.size:
        cr, cc, spent = _safe_place(ctx, "shade_canopy", cr, cc, spent, budget_usd)
        if cr.size:
            placements.append(Placement("shade_canopy", cr, cc))
            used[cr, cc] = True

    # ── Phase 4: Any remaining budget → additional shade canopies ─────────
    # Sweep up remaining budget with more canopy coverage.
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("shade_canopy"):
        open_ground2 = ctx.buildable & ~covered & ~used
        n_extra      = ctx.affordable("shade_canopy", remaining)
        n_extra      = min(n_extra, int(open_ground2.sum()))

        if n_extra > 0:
            er, ec = _top_pixels(canopy_score, open_ground2, n_extra)
            if er.size:
                er, ec, spent = _safe_place(
                    ctx, "shade_canopy", er, ec, spent, budget_usd
                )
                if er.size:
                    placements.append(Placement("shade_canopy", er, ec))

    return placements