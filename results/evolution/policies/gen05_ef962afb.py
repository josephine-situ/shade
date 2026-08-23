from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "population-heat max shade v2"
DESCRIPTION = (
    "Maximise population-weighted UTCI drop by targeting the most populated + "
    "hottest pedestrian corridors with dense shade. Priority surface weights "
    "population (0.30) and afternoon heat (0.40) very strongly to align with "
    "the heat_relief_c fitness metric, plus heat_hours (0.15), vulnerability "
    "(0.10), UHII (0.05). Three-phase planting: (1) medium trees at 4 m spacing "
    "consume 65% of budget forming dense shade corridors; (2) small trees fill "
    "remaining plantable gaps at 4 m spacing with 20% budget; (3) shade canopies "
    "cover hottest remaining open pedestrian ground with remaining 15% budget. "
    "Reflective surfaces avoided (albedo trap). Priority-tract boost 0.15."
)

WEIGHTS = {
    "heat_ta3pm":    0.40,
    "heat_hours":    0.15,
    "population":    0.30,
    "vulnerability": 0.10,
    "uhii":          0.05,
}

PRIORITY_BOOST       = 0.15
MEDIUM_BUDGET_FRAC   = 0.65
SMALL_BUDGET_FRAC    = 0.20
# Remaining goes to shade canopies
MEDIUM_SPACING_M     = 4.0
SMALL_SPACING_M      = 4.0


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1] over masked pixels; 0.5 if constant."""
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
        + WEIGHTS["heat_hours"]      * _norm(ctx.heat_hours,    mask)
        + WEIGHTS["population"]      * _norm(ctx.population,    mask)
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
    Greedy selection: pick highest-scoring candidate, block a spacing_px-radius
    box around it, repeat until `limit` reached or candidates exhausted.
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

    # ── 1. Medium street trees (65% budget, 4 m / 2 px spacing) ─────────
    medium_budget = budget_usd * MEDIUM_BUDGET_FRAC
    n_medium_max  = ctx.affordable("tree_medium", medium_budget)
    cand_medium   = ctx.plantable & ~used
    n_medium      = min(n_medium_max, int(cand_medium.sum()))

    spacing_medium = max(int(round(MEDIUM_SPACING_M / ctx.res_m)), 1)
    mr, mc = _greedy_spaced(score, cand_medium, spacing_medium, n_medium)

    if mr.size:
        placements.append(Placement("tree_medium", mr, mc))
        spent += ctx.cost("tree_medium", mr.size)
        used[mr, mc] = True

    # ── 2. Small street trees (20% budget, 4 m spacing, gap-fill) ────────
    small_budget = min(budget_usd * SMALL_BUDGET_FRAC, budget_usd - spent)
    if small_budget > ctx.unit_cost("tree_small"):
        cand_small   = ctx.plantable & ~used
        n_small_max  = ctx.affordable("tree_small", small_budget)
        n_small      = min(n_small_max, int(cand_small.sum()))

        spacing_small = max(int(round(SMALL_SPACING_M / ctx.res_m)), 1)
        sr, sc = _greedy_spaced(score, cand_small, spacing_small, n_small)

        if sr.size:
            placements.append(Placement("tree_small", sr, sc))
            spent += ctx.cost("tree_small", sr.size)
            used[sr, sc] = True
    else:
        sr = sc = np.array([], dtype=int)

    # ── 3. Shade canopies (remaining budget, hottest open ground) ─────────
    remaining = budget_usd - spent
    if remaining <= ctx.unit_cost("shade_canopy"):
        return placements

    # Mark tree crown footprints so canopies don't overlap with shaded ground
    covered  = np.zeros(ctx.shape, dtype=bool)
    # Medium tree crown ~3.5 m radius → ~2 px at 2 m resolution
    med_crown_px = max(int(round(3.5 / ctx.res_m)), 1)
    if mr.size:
        _mark_crown(covered, mr, mc, med_crown_px, ctx.shape)
    # Small tree crown ~2 m radius → 1 px
    sm_crown_px = max(int(round(2.0 / ctx.res_m)), 1)
    if sr.size:
        _mark_crown(covered, sr, sc, sm_crown_px, ctx.shape)

    open_ground = ctx.buildable & (ctx.cdsm <= 0.0) & ~covered & ~used

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(score, open_ground, n_canopy)
    if cr.size:
        placements.append(Placement("shade_canopy", cr, cc))

    return placements