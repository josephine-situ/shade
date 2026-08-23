from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "dual-tier-tree dense shade v1"
DESCRIPTION = (
    "Two-tier tree strategy: medium trees (6687/tree) planted at 5m spacing "
    "on hottest high-priority pedestrian corridors (60% of budget), then small "
    "trees (2024/tree) fill remaining plantable spots at 4m spacing (20% budget), "
    "then shade canopies cover hottest open buildable ground (20% budget). "
    "Priority surface: 0.45 afternoon heat, 0.20 heat hours, 0.20 population, "
    "0.10 vulnerability, 0.05 UHII. Priority-tract boost 0.15. "
    "Dense multi-tier shade maximizes UTCI relief per dollar. Albedo avoided."
)

WEIGHTS = {
    "heat_ta3pm":    0.45,
    "heat_hours":    0.20,
    "population":    0.20,
    "vulnerability": 0.10,
    "uhii":          0.05,
}

PRIORITY_BOOST        = 0.15
MEDIUM_TREE_BUDGET_FRAC = 0.60
SMALL_TREE_BUDGET_FRAC  = 0.20
CANOPY_BUDGET_FRAC      = 0.20

MEDIUM_TREE_SPACING_M = 5.0
SMALL_TREE_SPACING_M  = 4.0


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
    """Composite priority surface over exposure pixels."""
    mask = ctx.exposure
    score = (
          WEIGHTS["heat_ta3pm"]    * _norm(ctx.heat_ta3pm,    mask)
        + WEIGHTS["heat_hours"]    * _norm(ctx.heat_hours,    mask)
        + WEIGHTS["population"]    * _norm(ctx.population,    mask)
        + WEIGHTS["vulnerability"] * _norm(ctx.vulnerability, mask)
        + WEIGHTS["uhii"]          * _norm(ctx.heat_uhii,     mask)
    )
    score += np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
    used: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy selection respecting spacing and already-used pixels.
    Returns picked (rows, cols).
    """
    cand = candidates & ~used
    rows, cols = np.nonzero(cand)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    order = np.argsort(-score[rows, cols])
    rows, cols = rows[order], cols[order]

    taken = np.zeros(score.shape, dtype=bool)
    # Pre-mark used pixels
    taken |= used
    span = max(int(spacing_px), 1)
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
    used: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Select the top `limit` candidate pixels by score, excluding used pixels."""
    cand = candidates & ~used
    rows, cols = np.nonzero(cand)
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
    r0s = np.maximum(0, rows - crown_px)
    r1s = np.minimum(shape[0], rows + crown_px + 1)
    c0s = np.maximum(0, cols - crown_px)
    c1s = np.minimum(shape[1], cols + crown_px + 1)
    for i in range(len(rows)):
        covered[r0s[i]:r1s[i], c0s[i]:c1s[i]] = True


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score      = priority_surface(ctx)
    placements : list[Placement] = []
    spent      = 0.0
    used       = np.zeros(ctx.shape, dtype=bool)

    # ── 1. Medium street trees (60% budget, 5 m spacing) ──────────────
    medium_budget = budget_usd * MEDIUM_TREE_BUDGET_FRAC
    n_medium_max  = ctx.affordable("tree_medium", medium_budget)
    spacing_med   = max(int(round(MEDIUM_TREE_SPACING_M / ctx.res_m)), 1)

    tr_m, tc_m = _greedy_spaced(
        score, ctx.plantable, spacing_med, n_medium_max, used
    )

    if tr_m.size:
        placements.append(Placement("tree_medium", tr_m, tc_m))
        spent += ctx.cost("tree_medium", tr_m.size)
        used[tr_m, tc_m] = True

    # ── 2. Small street trees (20% budget, 4 m spacing on remaining spots) ──
    remaining_small = budget_usd * SMALL_TREE_BUDGET_FRAC
    # Also allow underspent medium budget to roll over
    remaining_small = min(remaining_small + max(0.0, medium_budget - spent), budget_usd - spent)
    
    n_small_max  = ctx.affordable("tree_small", remaining_small)
    spacing_sml  = max(int(round(SMALL_TREE_SPACING_M / ctx.res_m)), 1)

    tr_s, tc_s = _greedy_spaced(
        score, ctx.plantable, spacing_sml, n_small_max, used
    )

    if tr_s.size:
        cost_small = ctx.cost("tree_small", tr_s.size)
        if spent + cost_small <= budget_usd:
            placements.append(Placement("tree_small", tr_s, tc_s))
            spent += cost_small
            used[tr_s, tc_s] = True

    # ── 3. Shade canopies (remaining budget, hottest open ground) ──────
    remaining = budget_usd - spent
    if remaining <= 0.0:
        return placements

    # Mark crown footprints so canopies don't conflict with tree placements
    covered  = np.zeros(ctx.shape, dtype=bool)
    # Medium tree crown ~3.5m radius = ~2px; small tree ~2m = ~1px
    crown_med = max(int(round(3.5 / ctx.res_m)), 1)
    crown_sml = max(int(round(2.0 / ctx.res_m)), 1)
    if tr_m.size:
        _mark_crown(covered, tr_m, tc_m, crown_med, ctx.shape)
    if tr_s.size:
        _mark_crown(covered, tr_s, tc_s, crown_sml, ctx.shape)

    open_ground = ctx.buildable & (ctx.cdsm <= 0.0) & ~covered & ~used

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(score, open_ground, n_canopy, used)
    if cr.size:
        cost_canopy = ctx.cost("shade_canopy", cr.size)
        if spent + cost_canopy <= budget_usd:
            placements.append(Placement("shade_canopy", cr, cc))

    return placements