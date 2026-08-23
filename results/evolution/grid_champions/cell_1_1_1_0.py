from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "dense-triple-action heat-population v4"
DESCRIPTION = (
    "Maximise population-weighted UTCI relief via three-phase shading: "
    "(1) Medium trees at 4m spacing consume 60% of budget on hottest, "
    "most-populated pedestrian corridors; (2) Small trees at 3m spacing "
    "fill remaining plantable gaps with 20% of budget; "
    "(3) Shade canopies cover the hottest remaining open pedestrian ground "
    "with the final 20% of budget. Priority surface strongly weights "
    "population (0.40) and afternoon heat (0.35) to maximise "
    "population-weighted UTCI drops, plus heat_hours (0.10), UHII (0.08), "
    "and vulnerability (0.07). A priority-tract boost of 0.15 preserves "
    "equity in top-vulnerability tracts. Reflective surfaces avoided entirely."
)

WEIGHTS = {
    "heat_ta3pm":    0.35,
    "population":    0.40,
    "heat_hours":    0.10,
    "uhii":          0.08,
    "vulnerability": 0.07,
}

PRIORITY_BOOST = 0.15   # moderate boost for top-quartile vulnerability tracts

MEDIUM_BUDGET_FRAC = 0.60
SMALL_BUDGET_FRAC  = 0.20
# Remaining ~20% goes to shade canopies

MEDIUM_SPACING_M = 4.0   # tighter than parent (7m) for denser shade corridors
SMALL_SPACING_M  = 3.0   # very tight gap-fill between mediums

CROWN_MED_M   = 3.5      # medium tree crown radius for exclusion
CROWN_SMALL_M = 2.0      # small tree crown radius for exclusion


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1] over masked region; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Composite priority surface. Only meaningful where ctx.exposure is True.
    High score = hot + heavily-populated pedestrian corridor.
    Population weight is dominant to align with population-weighted UTCI fitness.
    """
    mask = ctx.exposure
    score = (
          WEIGHTS["heat_ta3pm"]    * _norm(ctx.heat_ta3pm,    mask)
        + WEIGHTS["population"]    * _norm(ctx.population,    mask)
        + WEIGHTS["heat_hours"]    * _norm(ctx.heat_hours,    mask)
        + WEIGHTS["uhii"]          * _norm(ctx.heat_uhii,     mask)
        + WEIGHTS["vulnerability"] * _norm(ctx.vulnerability, mask)
    )
    score += np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _population_heat_score(ctx: PlanningContext) -> np.ndarray:
    """
    Alternative score for canopy placement: maximize population×heat product
    to squeeze the most person-degC out of each canopy pixel.
    """
    mask = ctx.exposure
    # Use product-like combination to focus on pixels that are BOTH hot AND populated
    heat_n = _norm(ctx.heat_ta3pm, mask)
    pop_n  = _norm(ctx.population, mask)
    vuln_n = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours, mask)
    
    # Geometric mean component for population × heat synergy
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0, None))
    
    score = (
          0.45 * synergy
        + 0.25 * heat_n
        + 0.15 * pop_n
        + 0.10 * hours_n
        + 0.05 * vuln_n
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
    Greedy selection by score with spatial exclusion zone.
    Pick highest-scoring candidate pixel, suppress spacing_px neighbourhood, repeat.
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
    """Select top `limit` candidate pixels by score, no spacing constraint."""
    rows, cols = np.nonzero(candidates)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    order = np.argsort(-score[rows, cols])[:limit]
    return rows[order], cols[order]


def _stamp_crown(
    covered: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    crown_px: int,
    shape: tuple[int, int],
) -> None:
    """Mark a box of radius crown_px around each (r, c) as covered."""
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

    # ── Phase 1: Medium street trees (60% budget, 4m spacing) ───────────
    # Medium trees give the strongest per-tree UTCI benefit.
    # 4m spacing packs dense shade into hottest, most-populated corridors.
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

    # ── Phase 2: Small street trees (20% budget, 3m spacing, gap-fill) ──
    # Fill remaining plantable gaps cheaply — 3x more trees per dollar than medium.
    # Very tight 3m spacing to maximise shade in gaps between medium trees.
    small_budget = min(budget_usd * SMALL_BUDGET_FRAC, budget_usd - spent)
    sr = sc = np.array([], dtype=int)
    if small_budget >= ctx.unit_cost("tree_small"):
        cand_small  = ctx.plantable & ~used
        n_small_max = ctx.affordable("tree_small", small_budget)
        n_small     = min(n_small_max, int(cand_small.sum()))

        spacing_small = max(int(round(SMALL_SPACING_M / ctx.res_m)), 1)
        sr, sc = _greedy_spaced(score, cand_small, spacing_small, n_small)

        if sr.size:
            placements.append(Placement("tree_small", sr, sc))
            spent += ctx.cost("tree_small", sr.size)
            used[sr, sc] = True

    # ── Phase 3: Shade canopies (remaining ~20% budget, hottest open ground)
    remaining = budget_usd - spent
    if remaining < ctx.unit_cost("shade_canopy"):
        return placements

    # Exclude pixels already under new tree crowns to avoid redundant spend
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # existing canopy already covered

    if mr.size:
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)
    if sr.size:
        crown_small_px = max(int(round(CROWN_SMALL_M / ctx.res_m)), 1)
        _stamp_crown(covered, sr, sc, crown_small_px, ctx.shape)

    # Use population×heat synergy score for canopy placement
    canopy_score = _population_heat_score(ctx)

    open_ground  = ctx.buildable & ~covered & ~used
    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
    if cr.size:
        actual_cost = ctx.cost("shade_canopy", cr.size)
        if spent + actual_cost <= budget_usd:
            placements.append(Placement("shade_canopy", cr, cc))
            spent += actual_cost
        else:
            # Trim to fit exactly within budget
            affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
            if affordable_n > 0:
                placements.append(Placement("shade_canopy", cr[:affordable_n], cc[:affordable_n]))

    return placements