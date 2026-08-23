from __future__ import annotations

import numpy as np

from policy_api import Placement, PlanningContext

POLICY_NAME = "dense-small-tree equity corridors"
DESCRIPTION = (
    "Build a heat × vulnerability × population priority surface, then fill "
    "plantable pixels densely with small street trees (cheaper → more coverage), "
    "top up with medium trees in the hottest unshaded spots, and spend the "
    "remainder on shade canopies. Tighter tree spacing maximises UTCI shade "
    "coverage; stronger vulnerability weighting targets priority tracts."
)

WEIGHTS = {
    "heat_ta3pm": 0.30,
    "heat_hours": 0.20,
    "uhii": 0.10,
    "vulnerability": 0.30,
    "population": 0.10,
}

# Budget allocation fractions
SPLIT = {
    "tree_small": 0.55,
    "tree_medium": 0.20,
    "shade_canopy": 0.25,
}

# Spacing for trees in pixels
SMALL_TREE_SPACING_M = 5.0   # small crowns, can pack tighter
MEDIUM_TREE_SPACING_M = 10.0  # medium crowns need more room


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Rescale arr to 0-1 over mask pixels; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def priority_surface(ctx: PlanningContext) -> np.ndarray:
    """Composite priority: hotter, more vulnerable, more people = higher score."""
    mask = ctx.exposure
    score = (
        WEIGHTS["heat_ta3pm"] * _norm(ctx.heat_ta3pm, mask)
        + WEIGHTS["heat_hours"] * _norm(ctx.heat_hours, mask)
        + WEIGHTS["uhii"] * _norm(ctx.heat_uhii, mask)
        + WEIGHTS["vulnerability"] * _norm(ctx.vulnerability, mask)
        + WEIGHTS["population"] * _norm(ctx.population, mask)
    )
    # Extra boost for priority (top vulnerability quartile) tracts
    priority_boost = np.where(ctx.priority, 0.15, 0.0)
    score = score + priority_boost
    return np.where(mask, score, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy selection of up to `limit` candidate pixels spaced >= spacing_px apart."""
    rows, cols = np.nonzero(candidates)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    order = np.argsort(-score[rows, cols])
    rows, cols = rows[order], cols[order]

    taken = np.zeros(score.shape, dtype=bool)
    span = max(int(spacing_px), 1)
    pick_r, pick_c = [], []

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
    """Select the top `limit` candidate pixels by score, no spacing constraint."""
    rows, cols = np.nonzero(candidates)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    order = np.argsort(-score[rows, cols])[:limit]
    return rows[order], cols[order]


def _mark_covered(
    covered: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    crown_px: int,
    shape: tuple[int, int],
) -> None:
    """Mark a circular-ish crown footprint around each (r, c) as covered."""
    for r, c in zip(rows, cols):
        r0 = max(0, r - crown_px)
        r1 = min(shape[0], r + crown_px + 1)
        c0 = max(0, c - crown_px)
        c1 = min(shape[1], c + crown_px + 1)
        covered[r0:r1, c0:c1] = True


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score = priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0
    used = np.zeros(ctx.shape, dtype=bool)  # track pixels used by ANY action

    # ------------------------------------------------------------------ #
    # 1. Small trees: cheap → many → maximum shade coverage               #
    # ------------------------------------------------------------------ #
    small_budget = budget_usd * SPLIT["tree_small"]
    n_small_max = ctx.affordable("tree_small", small_budget)
    # plantable excludes roadbed, existing canopy, hydrant setbacks etc.
    candidates_small = ctx.plantable & ~used
    spacing_small = max(int(round(SMALL_TREE_SPACING_M / ctx.res_m)), 1)
    n_small = min(n_small_max, int(candidates_small.sum()))

    sr, sc = _greedy_spaced(score, candidates_small, spacing_small, n_small)
    if sr.size:
        placements.append(Placement("tree_small", sr, sc))
        spent += ctx.cost("tree_small", sr.size)
        used[sr, sc] = True

    # ------------------------------------------------------------------ #
    # 2. Medium trees: fill remaining high-priority plantable spots       #
    # ------------------------------------------------------------------ #
    medium_budget = min(budget_usd * SPLIT["tree_medium"], budget_usd - spent)
    if medium_budget > ctx.unit_cost("tree_medium"):
        candidates_medium = ctx.plantable & ~used
        spacing_medium = max(int(round(MEDIUM_TREE_SPACING_M / ctx.res_m)), 1)
        n_medium_max = ctx.affordable("tree_medium", medium_budget)
        n_medium = min(n_medium_max, int(candidates_medium.sum()))

        mr, mc = _greedy_spaced(score, candidates_medium, spacing_medium, n_medium)
        if mr.size:
            placements.append(Placement("tree_medium", mr, mc))
            spent += ctx.cost("tree_medium", mr.size)
            used[mr, mc] = True

    # ------------------------------------------------------------------ #
    # 3. Shade canopies: cover hottest open pedestrian pixels             #
    # ------------------------------------------------------------------ #
    remaining = budget_usd - spent
    if remaining > ctx.unit_cost("shade_canopy"):
        # Mark footprints already shaded by planted trees
        covered = np.zeros(ctx.shape, dtype=bool)

        # Small tree crown ~2m radius → 1 px at 2m resolution
        small_crown_px = max(int(round(2.0 / ctx.res_m)), 1)
        if sr.size:
            _mark_covered(covered, sr, sc, small_crown_px, ctx.shape)

        # Medium tree crown ~3m radius → ~1-2 px
        medium_crown_px = max(int(round(3.0 / ctx.res_m)), 1)
        if "mr" in dir() and mr.size:
            _mark_covered(covered, mr, mc, medium_crown_px, ctx.shape)

        # Canopy goes on buildable ground not already shaded
        open_ground = ctx.buildable & ~covered & ~used & (ctx.cdsm <= 0.0)
        n_canopy_max = ctx.affordable("shade_canopy", remaining)
        n_canopy = min(n_canopy_max, int(open_ground.sum()))

        cr, cc = _top_pixels(score, open_ground, n_canopy)
        if cr.size:
            placements.append(Placement("shade_canopy", cr, cc))
            spent += ctx.cost("shade_canopy", cr.size)

    return placements