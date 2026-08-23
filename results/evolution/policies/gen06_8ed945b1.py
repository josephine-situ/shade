from __future__ import annotations

import numpy as np

from policy_api import Placement, PlanningContext

POLICY_NAME = "aggressive-tree corridors equity-boosted v2"
DESCRIPTION = (
    "Maximise UTCI relief by planting medium street trees at 5 m spacing "
    "(tightest viable corridors) along the hottest, most vulnerable pedestrian "
    "routes, allocating 85% of the budget to trees. Heat signal dominates the "
    "priority surface (0.40 afternoon temp + 0.15 heat-hours + 0.10 UHII) with "
    "a strong vulnerability weight (0.25) and a large priority-tract additive "
    "boost (0.25) to ensure equity. Remaining 15% budget goes to shade canopies "
    "over the hottest uncovered pixels. Reflective surfaces avoided entirely."
)

WEIGHTS = {
    "heat_ta3pm":    0.40,
    "heat_hours":    0.15,
    "uhii":          0.10,
    "vulnerability": 0.25,
    "population":    0.10,
}

PRIORITY_BOOST   = 0.25   # additive boost for top-quartile vulnerability tracts
TREE_BUDGET_FRAC = 0.85
TREE_SPACING_M   = 5.0    # tight corridor spacing


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
    """Composite priority surface; high score = hot + vulnerable + populated."""
    mask = ctx.exposure
    score = (
        WEIGHTS["heat_ta3pm"]      * _norm(ctx.heat_ta3pm,    mask)
        + WEIGHTS["heat_hours"]    * _norm(ctx.heat_hours,    mask)
        + WEIGHTS["uhii"]          * _norm(ctx.heat_uhii,     mask)
        + WEIGHTS["vulnerability"] * _norm(ctx.vulnerability, mask)
        + WEIGHTS["population"]    * _norm(ctx.population,    mask)
    )
    # Strong boost for top-quartile tracts: ensures equity_ratio stays > 1
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy placement: pick highest-scoring candidate, suppress a
    spacing_px-radius neighbourhood, repeat until limit reached.
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


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score = priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0

    # ------------------------------------------------------------------ #
    # 1. Medium trees: best UTCI-per-dollar via shade                     #
    #    85% of budget, 5 m spacing for tight shade corridors             #
    # ------------------------------------------------------------------ #
    tree_budget = budget_usd * TREE_BUDGET_FRAC
    n_trees_max   = ctx.affordable("tree_medium", tree_budget)
    n_trees_avail = int(ctx.plantable.sum())
    n_trees       = min(n_trees_max, n_trees_avail)

    spacing_px = max(int(round(TREE_SPACING_M / ctx.res_m)), 1)
    tr, tc = _greedy_spaced(score, ctx.plantable, spacing_px, n_trees)

    if tr.size:
        placements.append(Placement("tree_medium", tr, tc))
        spent += ctx.cost("tree_medium", tr.size)

    # ------------------------------------------------------------------ #
    # 2. Shade canopies: remaining budget on hottest open pedestrian      #
    #    ground not already shaded by the trees we just planted.          #
    # ------------------------------------------------------------------ #
    remaining = budget_usd - spent

    # Build exclusion zone from newly planted tree crowns
    # Medium tree crown ~3 m radius → ~1-2 px at 2 m resolution
    covered = np.zeros(ctx.shape, dtype=bool)
    crown_px = max(int(round(3.0 / ctx.res_m)), 1)
    if tr.size:
        for r, c in zip(tr, tc):
            r0 = max(0, r - crown_px)
            r1 = min(ctx.shape[0], r + crown_px + 1)
            c0 = max(0, c - crown_px)
            c1 = min(ctx.shape[1], c + crown_px + 1)
            covered[r0:r1, c0:c1] = True

    # Open ground: buildable, no existing canopy, not newly shaded by trees
    open_ground = ctx.buildable & (ctx.cdsm <= 0.0) & ~covered
    # Also exclude tree placement pixels to prevent double-booking
    if tr.size:
        open_ground[tr, tc] = False

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(score, open_ground, n_canopy)
    if cr.size:
        placements.append(Placement("shade_canopy", cr, cc))

    return placements