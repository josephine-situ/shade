from __future__ import annotations

import numpy as np

from policy_api import Placement, PlanningContext

POLICY_NAME = "heat-first hybrid trees canopy v3"
DESCRIPTION = (
    "Maximise UTCI relief with a three-layer shade strategy: (1) medium trees "
    "at 6m spacing in the hottest pedestrian corridors (60% budget), "
    "(2) small trees at 4m spacing to extend corridor coverage (20% budget), "
    "(3) shade canopies to fill remaining unshaded hot ground (20% budget). "
    "Priority surface weights heat strongly (0.50 afternoon temp + 0.15 heat "
    "hours + 0.10 UHII) with moderate vulnerability (0.15) and population "
    "(0.10) signals. A small priority-tract boost (0.08) preserves equity. "
    "Reflective surfaces avoided entirely (albedo trap)."
)

WEIGHTS = {
    "heat_ta3pm": 0.50,
    "heat_hours": 0.15,
    "uhii": 0.10,
    "vulnerability": 0.15,
    "population": 0.10,
}

PRIORITY_BOOST = 0.08

MEDIUM_TREE_BUDGET_FRAC = 0.60
SMALL_TREE_BUDGET_FRAC = 0.20
CANOPY_BUDGET_FRAC = 0.20

MEDIUM_TREE_SPACING_M = 6.0
SMALL_TREE_SPACING_M = 4.0


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def priority_surface(ctx: PlanningContext) -> np.ndarray:
    mask = ctx.exposure
    score = (
        WEIGHTS["heat_ta3pm"]      * _norm(ctx.heat_ta3pm,     mask)
        + WEIGHTS["heat_hours"]    * _norm(ctx.heat_hours,     mask)
        + WEIGHTS["uhii"]          * _norm(ctx.heat_uhii,      mask)
        + WEIGHTS["vulnerability"] * _norm(ctx.vulnerability,  mask)
        + WEIGHTS["population"]    * _norm(ctx.population,     mask)
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
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
    rows, cols = np.nonzero(candidates)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    order = np.argsort(-score[rows, cols])[:limit]
    return rows[order], cols[order]


def _mark_covered(
    covered: np.ndarray,
    tr: np.ndarray,
    tc: np.ndarray,
    crown_radius_m: float,
    res_m: float,
    shape: tuple[int, int],
) -> None:
    crown_px = max(int(round(crown_radius_m / res_m)), 1)
    for r, c in zip(tr, tc):
        r0 = max(0, r - crown_px)
        r1 = min(shape[0], r + crown_px + 1)
        c0 = max(0, c - crown_px)
        c1 = min(shape[1], c + crown_px + 1)
        covered[r0:r1, c0:c1] = True


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score = priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0

    # Track placed pixels to avoid double-booking
    placed = np.zeros(ctx.shape, dtype=bool)
    covered = np.zeros(ctx.shape, dtype=bool)

    # ------------------------------------------------------------------ #
    # 1. Medium trees: dominant shade action, 6m spacing for corridors    #
    # ------------------------------------------------------------------ #
    medium_budget = budget_usd * MEDIUM_TREE_BUDGET_FRAC
    n_med_max = ctx.affordable("tree_medium", medium_budget)
    plantable_med = ctx.plantable & ~placed
    n_med = min(n_med_max, int(plantable_med.sum()))

    spacing_med = max(int(round(MEDIUM_TREE_SPACING_M / ctx.res_m)), 1)
    mr, mc = _greedy_spaced(score, plantable_med, spacing_med, n_med)

    if mr.size:
        placements.append(Placement("tree_medium", mr, mc))
        spent += ctx.cost("tree_medium", mr.size)
        placed[mr, mc] = True
        # Medium tree crown ~3.5m radius
        _mark_covered(covered, mr, mc, 3.5, ctx.res_m, ctx.shape)

    # ------------------------------------------------------------------ #
    # 2. Small trees: extend coverage in areas medium trees didn't reach  #
    #    4m spacing, use smaller budget fraction                          #
    # ------------------------------------------------------------------ #
    small_budget_alloc = budget_usd * SMALL_TREE_BUDGET_FRAC
    # Allow underspend from medium trees to roll into small trees
    small_budget = min(small_budget_alloc + max(0.0, medium_budget - spent), 
                       budget_usd - spent)
    
    if small_budget > ctx.unit_cost("tree_small"):
        n_small_max = ctx.affordable("tree_small", small_budget)
        # Prefer planting small trees where medium trees didn't go
        # Prioritize uncovered hot areas
        plantable_small = ctx.plantable & ~placed
        n_small = min(n_small_max, int(plantable_small.sum()))

        spacing_small = max(int(round(SMALL_TREE_SPACING_M / ctx.res_m)), 1)
        sr, sc = _greedy_spaced(score, plantable_small, spacing_small, n_small)

        if sr.size:
            placements.append(Placement("tree_small", sr, sc))
            spent += ctx.cost("tree_small", sr.size)
            placed[sr, sc] = True
            # Small tree crown ~2m radius
            _mark_covered(covered, sr, sc, 2.0, ctx.res_m, ctx.shape)

    # ------------------------------------------------------------------ #
    # 3. Shade canopies: fill remaining hot open pedestrian ground        #
    # ------------------------------------------------------------------ #
    remaining = budget_usd - spent
    if remaining > ctx.unit_cost("shade_canopy"):
        open_ground = ctx.buildable & (ctx.cdsm <= 0.0) & ~covered & ~placed
        n_canopy_max = ctx.affordable("shade_canopy", remaining)
        n_canopy = min(n_canopy_max, int(open_ground.sum()))

        cr, cc = _top_pixels(score, open_ground, n_canopy)
        if cr.size:
            placements.append(Placement("shade_canopy", cr, cc))

    return placements