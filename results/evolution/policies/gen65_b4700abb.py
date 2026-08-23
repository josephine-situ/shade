from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-dense-tree equity-boost v2"
DESCRIPTION = (
    "Three-phase heat-relief policy targeting population×heat synergy corridors "
    "with strong vulnerability weighting for equity. Phase 1: Medium trees at "
    "4m spacing (55% budget) on hottest, most-populated priority corridors. "
    "Phase 2: Small trees at 3m spacing (25% budget) for dense gap-fill. "
    "Phase 3: Shade canopies (20% budget) on remaining hot open ground. "
    "Priority surface uses geometric synergy of population×heat plus strong "
    "vulnerability boost (0.25) to maintain equity_ratio > 1. "
    "Reflective surfaces avoided entirely."
)

# Priority surface weights
WEIGHTS = {
    "heat_ta3pm":    0.30,
    "population":    0.35,
    "heat_hours":    0.12,
    "uhii":          0.08,
    "vulnerability": 0.15,
}

PRIORITY_BOOST = 0.20  # strong boost for top-vulnerability tracts

# Budget allocation
MEDIUM_BUDGET_FRAC = 0.55
SMALL_BUDGET_FRAC  = 0.25
# Remaining ~20% goes to shade canopies

# Spacing
MEDIUM_SPACING_M = 4.0
SMALL_SPACING_M  = 3.0

# Crown radii for exclusion
CROWN_MED_M   = 3.5
CROWN_SMALL_M = 2.0


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
    Composite priority surface with population×heat synergy.
    Strong vulnerability weighting maintains equity_ratio > 1.
    """
    mask = ctx.exposure
    
    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    
    # Geometric mean synergy: rewards pixels that are BOTH hot AND populated
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))
    
    score = (
          WEIGHTS["heat_ta3pm"]    * heat_n
        + WEIGHTS["population"]    * pop_n
        + WEIGHTS["heat_hours"]    * hours_n
        + WEIGHTS["uhii"]          * uhii_n
        + WEIGHTS["vulnerability"] * vuln_n
        + 0.10                     * synergy   # extra synergy bonus
    )
    
    # Strong priority boost for top-vulnerability tracts
    score += np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    
    return np.where(mask, score, -np.inf)


def canopy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Score for shade canopy placement: focus on hot, populated, vulnerable
    open pedestrian ground.
    """
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm, mask)
    pop_n   = _norm(ctx.population, mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours, mask)
    
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))
    
    score = (
          0.35 * synergy
        + 0.25 * heat_n
        + 0.20 * pop_n
        + 0.10 * hours_n
        + 0.10 * vuln_n
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
    Pick highest-scoring candidate, block spacing_px neighbourhood, repeat.
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
    score  = priority_surface(ctx)
    csc    = canopy_score(ctx)
    placements: list[Placement] = []
    spent  = 0.0
    used   = np.zeros(ctx.shape, dtype=bool)

    # ── Phase 1: Medium street trees (55% budget, 4m spacing) ───────────
    # Medium trees provide the strongest per-tree UTCI benefit.
    # 4m spacing creates dense shade corridors in hottest, most-populated areas.
    medium_budget  = budget_usd * MEDIUM_BUDGET_FRAC
    n_medium_max   = ctx.affordable("tree_medium", medium_budget)
    cand_medium    = ctx.plantable & ~used
    n_medium       = min(n_medium_max, int(cand_medium.sum()))

    spacing_medium = max(int(round(MEDIUM_SPACING_M / ctx.res_m)), 1)
    mr, mc = _greedy_spaced(score, cand_medium, spacing_medium, n_medium)

    if mr.size:
        placements.append(Placement("tree_medium", mr, mc))
        spent += ctx.cost("tree_medium", mr.size)
        used[mr, mc] = True

    # ── Phase 2: Small street trees (25% budget, 3m spacing, gap-fill) ──
    # Very tight 3m spacing fills gaps between medium trees.
    # 3x more trees per dollar — maximises total shade coverage.
    small_budget  = min(budget_usd * SMALL_BUDGET_FRAC, budget_usd - spent)
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

    # Stamp crown footprints for trees already planted
    covered = np.zeros(ctx.shape, dtype=bool)
    # Mark existing canopy as covered
    covered[ctx.cdsm > 0.0] = True

    if mr.size:
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)
    if sr.size:
        crown_small_px = max(int(round(CROWN_SMALL_M / ctx.res_m)), 1)
        _stamp_crown(covered, sr, sc, crown_small_px, ctx.shape)

    # Canopy on hot, open, buildable ground not covered by trees
    open_ground  = ctx.buildable & ~covered & ~used
    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(csc, open_ground, n_canopy)
    if cr.size:
        actual_cost = ctx.cost("shade_canopy", cr.size)
        if spent + actual_cost <= budget_usd:
            placements.append(Placement("shade_canopy", cr, cc))
            spent += actual_cost
        else:
            # Trim to budget
            affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
            if affordable_n > 0:
                placements.append(
                    Placement("shade_canopy", cr[:affordable_n], cc[:affordable_n])
                )

    return placements