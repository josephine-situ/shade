from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "heat-first medium trees with canopy infill v3"
DESCRIPTION = (
    "Maximise population-weighted UTCI relief. Priority surface weights afternoon "
    "temperature (0.42) and population (0.28) strongly to align with heat_relief_c "
    "metric, plus heat_hours (0.12), vulnerability (0.12), UHII (0.06). Medium trees "
    "at 5m spacing consume 82% of budget forming dense shade corridors on hottest "
    "populated ground; shade canopies cover remaining open pedestrian ground. "
    "Priority-tract boost 0.15 maintains equity. Reflective surfaces avoided entirely."
)

WEIGHTS = {
    "heat_ta3pm":    0.42,
    "population":    0.28,
    "heat_hours":    0.12,
    "vulnerability": 0.12,
    "uhii":          0.06,
}

PRIORITY_BOOST   = 0.15
TREE_BUDGET_FRAC = 0.82
TREE_SPACING_M   = 5.0


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0,1] over masked pixels; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def priority_surface(ctx: PlanningContext) -> np.ndarray:
    """Composite priority; positive only where ctx.exposure is True."""
    mask = ctx.exposure
    score = (
          WEIGHTS["heat_ta3pm"]      * _norm(ctx.heat_ta3pm,    mask)
        + WEIGHTS["population"]      * _norm(ctx.population,    mask)
        + WEIGHTS["heat_hours"]      * _norm(ctx.heat_hours,    mask)
        + WEIGHTS["vulnerability"]   * _norm(ctx.vulnerability, mask)
        + WEIGHTS["uhii"]            * _norm(ctx.heat_uhii,     mask)
    )
    score += np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy selection: pick highest-scoring candidate, block spacing_px
    neighbourhood, repeat until `limit` reached or no candidates left.
    """
    rows, cols = np.nonzero(candidates)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    order = np.argsort(-score[rows, cols])
    rows, cols = rows[order], cols[order]

    taken  = np.zeros(score.shape, dtype=bool)
    span   = max(int(spacing_px), 1)
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
    """Select the top `limit` candidate pixels by score, no spacing constraint."""
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
    """Stamp a box crown_px wide around each (r, c) into covered."""
    H, W = shape
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        r0 = max(0, r - crown_px);  r1 = min(H, r + crown_px + 1)
        c0 = max(0, c - crown_px);  c1 = min(W, c + crown_px + 1)
        covered[r0:r1, c0:c1] = True


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score      = priority_surface(ctx)
    placements: list[Placement] = []
    spent      = 0.0
    used       = np.zeros(ctx.shape, dtype=bool)

    # ── 1. Medium street trees (82% budget, 5m / ~3px spacing) ──────────
    # Dense spacing maximises shade canopy coverage on hot pedestrian ground.
    # Medium trees provide the highest UTCI relief per dollar.
    tree_budget  = budget_usd * TREE_BUDGET_FRAC
    n_trees_max  = ctx.affordable("tree_medium", tree_budget)
    cand_trees   = ctx.plantable & ~used
    n_trees      = min(n_trees_max, int(cand_trees.sum()))

    spacing_px   = max(int(round(TREE_SPACING_M / ctx.res_m)), 1)
    tr, tc       = _greedy_spaced(score, cand_trees, spacing_px, n_trees)

    if tr.size:
        placements.append(Placement("tree_medium", tr, tc))
        spent += ctx.cost("tree_medium", tr.size)
        used[tr, tc] = True

    # ── 2. Shade canopies (remaining budget, hottest open ground) ────────
    remaining = budget_usd - spent
    if remaining <= ctx.unit_cost("shade_canopy"):
        return placements

    # Mark tree crown footprints; medium tree crown ~3.5m radius → ~2px at 2m res
    covered  = np.zeros(ctx.shape, dtype=bool)
    crown_px = max(int(round(3.5 / ctx.res_m)), 1)
    if tr.size:
        _mark_crown(covered, tr, tc, crown_px, ctx.shape)

    # Canopy: buildable, no existing canopy, not under new tree crowns, not used
    open_ground = ctx.buildable & (ctx.cdsm <= 0.0) & ~covered & ~used

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(score, open_ground, n_canopy)
    if cr.size:
        placements.append(Placement("shade_canopy", cr, cc))

    return placements