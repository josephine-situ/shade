from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-equity corridors v1"
DESCRIPTION = (
    "Maximise heat_relief_c via synergy-based priority surfaces with strong "
    "equity targeting. Three-tier shade deployment: (1) Medium trees at 3m "
    "spacing on highest heat×population synergy corridors (65% budget); "
    "(2) Small trees filling crown gaps at 2m spacing (20% budget); "
    "(3) Shade canopies on remaining hot open buildable ground (15% budget). "
    "Priority surface: geometric mean synergy(heat×pop) 0.50, heat_ta3pm 0.20, "
    "population 0.12, vulnerability 0.10, heat_hours 0.05, UHII 0.03. "
    "Strong priority-tract boost 0.35 for equity_ratio > 1. "
    "Pre-existing canopy tracked to avoid wasting budget. "
    "Reflective surfaces avoided entirely (albedo trap)."
)

PRIORITY_BOOST = 0.35   # strong boost for top-quartile tracts → equity_ratio > 1

FRAC_MED    = 0.65
FRAC_SML    = 0.20
# remaining ~15% goes to shade canopies

SPACING_MED_M = 3.0
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


def priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Synergy-based composite priority surface for tree placement.
    Geometric mean of heat × population rewards pixels that are simultaneously
    hot AND populated — these yield maximum person-degC relief per tree.
    Vulnerability weight and priority boost ensure equity_ratio > 1.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    # Geometric mean synergy rewards co-occurrence of heat AND population
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.50 * synergy
        + 0.20 * heat_n
        + 0.12 * pop_n
        + 0.10 * vuln_n
        + 0.05 * hours_n
        + 0.03 * uhii_n
    )

    # Strong boost for top-quartile vulnerability tracts → keeps equity_ratio > 1
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def canopy_priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for shade canopy placement.
    Stronger synergy weight — canopies provide immediate shade over
    high-density pedestrian areas, maximising population-weighted UTCI relief.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.55 * synergy
        + 0.20 * heat_n
        + 0.12 * pop_n
        + 0.08 * vuln_n
        + 0.05 * hours_n
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
    suppress a spacing_px-radius neighbourhood, repeat until limit reached.
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


def _stamp_exclusion(
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


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    tree_score   = priority_surface(ctx)
    canopy_score = canopy_priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0

    # Track which pixels are used (no double-booking)
    used    = np.zeros(ctx.shape, dtype=bool)
    # Track canopy coverage (existing + new plantings)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # pre-existing canopy counts

    # ── 1. Medium street trees (65% budget, 3m spacing) ─────────────────
    # Synergy surface targets hot+populated corridors for max UTCI relief.
    med_budget = budget_usd * FRAC_MED
    n_med_max  = ctx.affordable("tree_medium", med_budget)
    cand_med   = ctx.plantable & ~used

    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    n_med = min(n_med_max, int(cand_med.sum()))

    tr_m, tc_m = _greedy_spaced(tree_score, cand_med, spacing_med_px, n_med)

    if tr_m.size:
        placements.append(Placement("tree_medium", tr_m, tc_m))
        spent += ctx.cost("tree_medium", tr_m.size)
        used[tr_m, tc_m] = True
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_exclusion(covered, tr_m, tc_m, crown_med_px, ctx.shape)

    # ── 2. Small street trees (20% budget, 2m spacing, gap-filling) ─────
    # Fill canopy gaps between medium trees — tight spacing to maximise
    # continuous shade corridors.
    sml_budget_target = budget_usd * FRAC_SML
    remaining_for_sml = budget_usd - spent
    sml_budget = min(sml_budget_target, remaining_for_sml)

    tr_s = tc_s = np.array([], dtype=int)
    if sml_budget >= ctx.unit_cost("tree_small"):
        n_sml_max = ctx.affordable("tree_small", sml_budget)
        # Exclude used pixels AND pixels under existing/new medium tree crowns
        cand_sml = ctx.plantable & ~used & ~covered

        spacing_sml_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)
        n_sml = min(n_sml_max, int(cand_sml.sum()))

        tr_s, tc_s = _greedy_spaced(tree_score, cand_sml, spacing_sml_px, n_sml)

        if tr_s.size:
            placements.append(Placement("tree_small", tr_s, tc_s))
            spent += ctx.cost("tree_small", tr_s.size)
            used[tr_s, tc_s] = True
            crown_sml_px = max(int(round(CROWN_SML_M / ctx.res_m)), 1)
            _stamp_exclusion(covered, tr_s, tc_s, crown_sml_px, ctx.shape)

    # ── 3. Shade canopies (remaining budget, hottest open ground) ────────
    # Infill remaining buildable open ground not already shaded by trees.
    # Canopy synergy score targets hottest populated open areas.
    remaining = budget_usd - spent
    if remaining < ctx.unit_cost("shade_canopy"):
        return placements

    # Open buildable ground: buildable, not covered by canopy, not used
    open_ground = ctx.buildable & ~covered & ~used

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy = min(n_canopy_max, int(open_ground.sum()))

    if n_canopy > 0:
        cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
        if cr.size:
            actual_cost = ctx.cost("shade_canopy", cr.size)
            if spent + actual_cost <= budget_usd + 0.01:
                placements.append(Placement("shade_canopy", cr, cc))
            else:
                # Trim to fit budget exactly
                affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
                if affordable_n > 0:
                    placements.append(
                        Placement("shade_canopy", cr[:affordable_n], cc[:affordable_n])
                    )

    return placements