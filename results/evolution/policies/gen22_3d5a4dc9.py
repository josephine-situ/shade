from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "triple-action heat-equity corridors v2"
DESCRIPTION = (
    "Improved heat_relief_c maximisation: (1) medium trees at tight 3m spacing "
    "on highest heat×population×vulnerability corridors (72% budget), prioritising "
    "population-weighted UTCI drops; (2) shade canopies on hottest open buildable "
    "ground near high-population areas (28% budget). Priority surface strongly "
    "weights population (0.35) and afternoon heat (0.40) with vulnerability boost "
    "of 0.25 for top-quartile tracts. No reflective surfaces. Two-pass canopy "
    "placement avoids crown overlap for maximum pixel efficiency."
)

# Priority surface weights — population-weighted UTCI is the fitness
WEIGHTS = {
    "heat_ta3pm":    0.40,
    "population":    0.35,
    "vulnerability": 0.10,
    "heat_hours":    0.10,
    "uhii":          0.05,
}

PRIORITY_BOOST = 0.25   # strong boost for top-quartile vulnerability tracts

# Budget fractions — concentrate on highest-impact action (medium trees)
FRAC_MED    = 0.72
FRAC_CANOPY = 0.28

# Spacing — tighter than parent for denser shade corridors
SPACING_MED_M = 3.0   # tighter than parent's 4m

# Crown radius for canopy exclusion
CROWN_MED_M = 3.5


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
    """Composite priority surface; only meaningful where ctx.exposure is True."""
    mask = ctx.exposure
    score = (
          WEIGHTS["heat_ta3pm"]    * _norm(ctx.heat_ta3pm,    mask)
        + WEIGHTS["population"]    * _norm(ctx.population,    mask)
        + WEIGHTS["vulnerability"] * _norm(ctx.vulnerability, mask)
        + WEIGHTS["heat_hours"]    * _norm(ctx.heat_hours,    mask)
        + WEIGHTS["uhii"]          * _norm(ctx.heat_uhii,     mask)
    )
    # Boost top-quartile vulnerability tracts
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

    for r, c in zip(rows, cols):
        if taken[r, c]:
            continue
        pick_r.append(int(r))
        pick_c.append(int(c))
        r0 = max(0, r - span)
        r1 = min(score.shape[0], r + span + 1)
        c0 = max(0, c - span)
        c1 = min(score.shape[1], c + span + 1)
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
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        r0 = max(0, r - radius_px)
        r1 = min(shape[0], r + radius_px + 1)
        c0 = max(0, c - radius_px)
        c1 = min(shape[1], c + radius_px + 1)
        covered[r0:r1, c0:c1] = True


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score = priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0

    # Track which pixels are already used (no double-booking)
    used = np.zeros(ctx.shape, dtype=bool)
    # Track canopy coverage (existing + new trees) for canopy placement
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # existing canopy already covered

    # ── 1. Medium street trees ───────────────────────────────────────────
    # 72% of budget, 3 m spacing for dense shade corridors on hottest ground
    med_budget = budget_usd * FRAC_MED
    n_med_max  = ctx.affordable("tree_medium", med_budget)
    cand_med   = ctx.plantable & ~used

    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    n_med = min(n_med_max, int(cand_med.sum()))

    tr_m, tc_m = _greedy_spaced(score, cand_med, spacing_med_px, n_med)

    if tr_m.size:
        placements.append(Placement("tree_medium", tr_m, tc_m))
        spent += ctx.cost("tree_medium", tr_m.size)
        used[tr_m, tc_m] = True
        # Crown exclusion for canopy placement
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_exclusion(covered, tr_m, tc_m, crown_med_px, ctx.shape)

    # ── 2. Shade canopies ────────────────────────────────────────────────
    # Remaining ~28% budget; fill hottest open buildable ground not already shaded
    # Use a canopy-specific score that emphasises population density even more
    # to maximise person-weighted UTCI drops
    remaining = budget_usd - spent
    if remaining <= 0.0:
        return placements

    # Canopy score: emphasise population even more for person-weighted metric
    mask = ctx.exposure
    canopy_score = (
          0.35 * _norm(ctx.heat_ta3pm,    mask)
        + 0.40 * _norm(ctx.population,    mask)
        + 0.10 * _norm(ctx.vulnerability, mask)
        + 0.10 * _norm(ctx.heat_hours,    mask)
        + 0.05 * _norm(ctx.heat_uhii,     mask)
    )
    canopy_score = canopy_score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    canopy_score = np.where(mask, canopy_score, -np.inf)

    # Open ground: buildable, no existing or new canopy coverage, not used
    open_ground = ctx.buildable & ~covered & ~used

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
    if cr.size:
        actual_cost = ctx.cost("shade_canopy", cr.size)
        if spent + actual_cost <= budget_usd:
            placements.append(Placement("shade_canopy", cr, cc))
            spent += actual_cost
        else:
            # Trim to fit budget
            affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
            if affordable_n > 0:
                cr2, cc2 = cr[:affordable_n], cc[:affordable_n]
                placements.append(Placement("shade_canopy", cr2, cc2))

    return placements