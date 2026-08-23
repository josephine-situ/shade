from __future__ import annotations

import numpy as np

from policy_api import Placement, PlanningContext

POLICY_NAME = "tight-tree corridors with canopy infill"
DESCRIPTION = (
    "Build a heat × heat-hours × vulnerability × population priority surface "
    "with a priority-tract boost; plant medium street trees at 6 m spacing "
    "(tighter than default to maximise shade corridor coverage); then "
    "aggressively infill all remaining open pedestrian ground with shade "
    "canopies, prioritising the hottest unshaded pixels. Budget split: 72% "
    "trees, 28% canopies. Reflective surfaces are avoided (albedo trap)."
)

# Priority surface weights
WEIGHTS = {
    "heat_ta3pm": 0.35,   # afternoon peak temperature
    "heat_hours": 0.20,   # cumulative heat exposure
    "uhii": 0.10,         # urban heat island intensity
    "vulnerability": 0.25,
    "population": 0.10,
}

PRIORITY_BOOST = 0.20     # additive boost for top-quartile vulnerability tracts

SPLIT = {
    "tree_medium": 0.72,
    "shade_canopy": 0.28,
}

TREE_SPACING_M = 6.0      # tighter packing → more shade corridors


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
        WEIGHTS["heat_ta3pm"]   * _norm(ctx.heat_ta3pm,   mask)
        + WEIGHTS["heat_hours"] * _norm(ctx.heat_hours,   mask)
        + WEIGHTS["uhii"]       * _norm(ctx.heat_uhii,    mask)
        + WEIGHTS["vulnerability"] * _norm(ctx.vulnerability, mask)
        + WEIGHTS["population"] * _norm(ctx.population,   mask)
    )
    # Boost top-quartile vulnerability tracts to improve equity and target
    # areas where heat + social risk overlap
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy selection: pick the highest-scoring candidate, then block a
    spacing_px-radius box around it, repeat until `limit` reached.
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
    # 1. Medium trees: best UTCI-per-dollar for shade; tight 6 m spacing  #
    #    to form continuous shade corridors along hot pedestrian routes.   #
    # ------------------------------------------------------------------ #
    tree_budget = budget_usd * SPLIT["tree_medium"]
    n_trees_max = ctx.affordable("tree_medium", tree_budget)
    n_trees_avail = int(ctx.plantable.sum())
    n_trees = min(n_trees_max, n_trees_avail)

    spacing_px = max(int(round(TREE_SPACING_M / ctx.res_m)), 1)
    tr, tc = _greedy_spaced(score, ctx.plantable, spacing_px, n_trees)

    if tr.size:
        placements.append(Placement("tree_medium", tr, tc))
        spent += ctx.cost("tree_medium", tr.size)

    # ------------------------------------------------------------------ #
    # 2. Shade canopies: infill all remaining open pedestrian ground.     #
    #    Budget = allocated canopy share + any tree underspend.           #
    #    Open ground = buildable, no existing canopy, not planted today.  #
    # ------------------------------------------------------------------ #
    remaining = budget_usd - spent

    # Mark footprint already shaded by newly planted trees
    # Medium tree crown ≈ 3–4 m radius → 2 px at 2 m resolution
    covered = np.zeros(ctx.shape, dtype=bool)
    crown_px = max(int(round(3.0 / ctx.res_m)), 1)
    if tr.size:
        for r, c in zip(tr, tc):
            r0 = max(0, r - crown_px)
            r1 = min(ctx.shape[0], r + crown_px + 1)
            c0 = max(0, c - crown_px)
            c1 = min(ctx.shape[1], c + crown_px + 1)
            covered[r0:r1, c0:c1] = True

    # Canopy eligibility: buildable, no existing canopy overhead, not covered
    # by trees we just planted
    open_ground = ctx.buildable & (ctx.cdsm <= 0.0) & ~covered

    # Also exclude tree placement pixels themselves (avoid double-booking)
    if tr.size:
        open_ground[tr, tc] = False

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(score, open_ground, n_canopy)
    if cr.size:
        placements.append(Placement("shade_canopy", cr, cc))
        # spent += ctx.cost("shade_canopy", cr.size)  # no need to track further

    return placements