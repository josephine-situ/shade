from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "heat-population dual-tree canopy v1"
DESCRIPTION = (
    "Maximise population-weighted UTCI drop by combining dual tree species "
    "with shade canopy infill. Priority surface strongly weights afternoon "
    "heat (0.45) and population density (0.25) to align with heat_relief_c "
    "metric, plus vulnerability (0.15) and heat hours (0.10), UHII (0.05). "
    "Budget: 50% medium trees at 6m spacing for shade corridors, 20% small "
    "trees at 4m spacing for dense infill, 30% shade canopies on hottest "
    "open pedestrian ground. Priority-tract boost 0.15 for equity. "
    "Albedo interventions avoided (albedo trap)."
)

WEIGHTS = {
    "heat_ta3pm":    0.45,
    "population":    0.25,
    "vulnerability": 0.15,
    "heat_hours":    0.10,
    "uhii":          0.05,
}

PRIORITY_BOOST = 0.15

SPLIT = {
    "tree_medium": 0.55,
    "tree_small":  0.18,
    "shade_canopy": 0.27,
}

MEDIUM_TREE_SPACING_M = 6.0
SMALL_TREE_SPACING_M  = 4.0


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
    """Composite priority surface aligned with heat_relief_c metric."""
    mask = ctx.exposure
    score = (
        WEIGHTS["heat_ta3pm"]    * _norm(ctx.heat_ta3pm,    mask)
        + WEIGHTS["population"]  * _norm(ctx.population,    mask)
        + WEIGHTS["vulnerability"] * _norm(ctx.vulnerability, mask)
        + WEIGHTS["heat_hours"]  * _norm(ctx.heat_hours,    mask)
        + WEIGHTS["uhii"]        * _norm(ctx.heat_uhii,     mask)
    )
    score += np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy selection: pick highest-score candidate, block spacing_px radius, repeat."""
    rows, cols = np.nonzero(candidates)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    order = np.argsort(-score[rows, cols])
    rows, cols = rows[order], cols[order]

    taken = np.zeros(score.shape, dtype=bool)
    span  = max(int(spacing_px), 1)
    pick_r: list[int] = []
    pick_c: list[int] = []

    for r, c in zip(rows, cols):
        if taken[r, c]:
            continue
        pick_r.append(int(r))
        pick_c.append(int(c))
        r0 = max(0, r - span);  r1 = min(score.shape[0], r + span + 1)
        c0 = max(0, c - span);  c1 = min(score.shape[1], c + span + 1)
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


def _mark_crown(
    covered: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    crown_px: int,
    shape: tuple[int, int],
) -> None:
    """Stamp a square crown footprint around each (r,c) into covered."""
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        r0 = max(0, r - crown_px);  r1 = min(shape[0], r + crown_px + 1)
        c0 = max(0, c - crown_px);  c1 = min(shape[1], c + crown_px + 1)
        covered[r0:r1, c0:c1] = True


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score      = priority_surface(ctx)
    placements : list[Placement] = []
    spent      = 0.0
    used       = np.zeros(ctx.shape, dtype=bool)

    # ── 1. Medium street trees (55% budget, 6m/3px spacing) ─────────────
    # Medium trees give most UTCI relief per tree due to larger crown
    medium_budget = budget_usd * SPLIT["tree_medium"]
    n_medium_max  = ctx.affordable("tree_medium", medium_budget)
    cand_medium   = ctx.plantable & ~used
    n_medium      = min(n_medium_max, int(cand_medium.sum()))

    spacing_medium = max(int(round(MEDIUM_TREE_SPACING_M / ctx.res_m)), 1)
    mr, mc = _greedy_spaced(score, cand_medium, spacing_medium, n_medium)

    if mr.size:
        placements.append(Placement("tree_medium", mr, mc))
        spent += ctx.cost("tree_medium", mr.size)
        used[mr, mc] = True

    # ── 2. Small street trees (18% budget, 4m/2px spacing) ──────────────
    # Fill gaps between medium trees for continuous shade coverage
    remaining_small = min(budget_usd * SPLIT["tree_small"], budget_usd - spent)
    if remaining_small >= ctx.unit_cost("tree_small"):
        n_small_max = ctx.affordable("tree_small", remaining_small)
        cand_small  = ctx.plantable & ~used
        n_small     = min(n_small_max, int(cand_small.sum()))

        spacing_small = max(int(round(SMALL_TREE_SPACING_M / ctx.res_m)), 1)
        sr, sc = _greedy_spaced(score, cand_small, spacing_small, n_small)

        if sr.size:
            placements.append(Placement("tree_small", sr, sc))
            spent += ctx.cost("tree_small", sr.size)
            used[sr, sc] = True
    else:
        sr = np.array([], dtype=int)
        sc = np.array([], dtype=int)

    # ── 3. Shade canopies (remaining budget, hottest open pedestrian ground)
    remaining = budget_usd - spent
    if remaining <= 0.0:
        return placements

    # Mark crown footprints so canopies don't double-claim shaded area
    covered  = np.zeros(ctx.shape, dtype=bool)

    # Medium tree crown ~3.5m radius → ~2px at 2m resolution
    med_crown_px = max(int(round(3.5 / ctx.res_m)), 1)
    if mr.size:
        _mark_crown(covered, mr, mc, med_crown_px, ctx.shape)

    # Small tree crown ~2m radius → ~1px at 2m resolution
    sm_crown_px = max(int(round(2.0 / ctx.res_m)), 1)
    if sr.size:
        _mark_crown(covered, sr, sc, sm_crown_px, ctx.shape)

    # Canopy on buildable, uncovered, unshaded, unused ground
    open_ground  = ctx.buildable & (ctx.cdsm <= 0.0) & ~covered & ~used

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(score, open_ground, n_canopy)
    if cr.size:
        placements.append(Placement("shade_canopy", cr, cc))
        spent += ctx.cost("shade_canopy", cr.size)

    return placements