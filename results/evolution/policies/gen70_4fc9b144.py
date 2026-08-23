from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "quad-optimised shade corridors v1"
DESCRIPTION = (
    "Target all four high behavioral axes: equity, access, cost-efficiency, greening. "
    "Three-phase shade deployment: "
    "(1) Medium trees at 4m spacing on heat×population×vulnerability corridors (55% budget); "
    "(2) Small trees at 3m spacing gap-fill remaining plantable ground (20% budget); "
    "(3) Shade canopies on hottest remaining open buildable ground (25% budget). "
    "Priority surface strongly weights population (0.38) and afternoon heat (0.38) "
    "for cost-efficiency, with moderate vulnerability weight (0.12) + priority-tract "
    "boost of 0.20 to keep equity_ratio > 1. Small trees boost cobenefit_greened_pct "
    "above threshold. Geometric synergy term maximises person-degC relief. "
    "No reflective surfaces — pure shade strategy."
)

# Priority surface weights — balanced for cost efficiency + equity
WEIGHTS = {
    "heat_ta3pm":    0.38,
    "population":    0.38,
    "vulnerability": 0.12,
    "heat_hours":    0.07,
    "uhii":          0.05,
}

PRIORITY_BOOST = 0.20   # keeps equity_ratio > 1 without sacrificing efficiency

# Budget fractions
FRAC_MED    = 0.55   # medium trees — highest per-unit UTCI benefit
FRAC_SMALL  = 0.20   # small trees — cheaper, boosts greening + coverage
FRAC_CANOPY = 0.25   # shade canopies — fill open buildable ground

# Tree spacing
SPACING_MED_M   = 4.0   # 4m → 2 pixel spacing at 2m res
SPACING_SMALL_M = 3.0   # 3m → 1-2 pixel spacing

# Crown radii for exclusion zones
CROWN_MED_M   = 3.5
CROWN_SMALL_M = 2.0


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
    """
    Composite priority for tree placement.
    Geometric synergy between heat and population maximises person-degC relief.
    Moderate vulnerability weighting + priority-tract boost keeps equity > 1.
    """
    mask = ctx.exposure
    heat_n = _norm(ctx.heat_ta3pm,    mask)
    pop_n  = _norm(ctx.population,    mask)
    vuln_n = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,   mask)
    uhii_n  = _norm(ctx.heat_uhii,    mask)

    # Geometric synergy: pixels that are BOTH hot AND populated
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.35 * synergy                        # heat×pop synergy
        + WEIGHTS["heat_ta3pm"]    * heat_n     # direct heat
        + WEIGHTS["population"]    * pop_n      # direct population
        + WEIGHTS["vulnerability"] * vuln_n     # equity
        + WEIGHTS["heat_hours"]    * hours_n    # chronic exposure
        + WEIGHTS["uhii"]          * uhii_n     # island intensity
    )
    score += np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def canopy_priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for shade canopy placement.
    Emphasises heat×population synergy for maximum cost efficiency.
    """
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.50 * synergy
        + 0.20 * heat_n
        + 0.15 * pop_n
        + 0.10 * vuln_n
        + 0.05 * hours_n
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
    Greedy spacing: pick highest-scoring candidate, suppress neighbours
    within spacing_px radius, repeat until limit reached.
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
    """Mark a square of radius crown_px around each (r,c) as covered."""
    H, W = shape
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        r0 = max(0, r - crown_px);  r1 = min(H, r + crown_px + 1)
        c0 = max(0, c - crown_px);  c1 = min(W, c + crown_px + 1)
        covered[r0:r1, c0:c1] = True


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    tree_score   = priority_surface(ctx)
    canopy_score = canopy_priority_surface(ctx)

    placements: list[Placement] = []
    spent = 0.0
    used  = np.zeros(ctx.shape, dtype=bool)

    # Track canopy coverage to avoid placing canopies under trees
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # existing canopy already blocked

    # ── Phase 1: Medium street trees (55% budget, 4m spacing) ───────────
    # Medium trees deliver the strongest per-tree UTCI benefit.
    # 4m spacing balances density vs. diminishing shade overlap.
    med_budget  = budget_usd * FRAC_MED
    n_med_max   = ctx.affordable("tree_medium", med_budget)
    cand_med    = ctx.plantable & ~used
    n_med       = min(n_med_max, int(cand_med.sum()))

    spacing_med = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    mr, mc = _greedy_spaced(tree_score, cand_med, spacing_med, n_med)

    if mr.size:
        placements.append(Placement("tree_medium", mr, mc))
        spent += ctx.cost("tree_medium", mr.size)
        used[mr, mc] = True
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)

    # ── Phase 2: Small street trees (20% budget, 3m spacing, gap-fill) ──
    # Small trees are cheaper → more coverage → higher cobenefit_greened_pct.
    # 3m spacing fills gaps between medium trees for continuous shade corridors.
    small_budget = min(budget_usd * FRAC_SMALL, budget_usd - spent)
    sr = sc = np.array([], dtype=int)

    if small_budget >= ctx.unit_cost("tree_small"):
        cand_small  = ctx.plantable & ~used
        n_small_max = ctx.affordable("tree_small", small_budget)
        n_small     = min(n_small_max, int(cand_small.sum()))

        spacing_small = max(int(round(SPACING_SMALL_M / ctx.res_m)), 1)
        sr, sc = _greedy_spaced(tree_score, cand_small, spacing_small, n_small)

        if sr.size:
            placements.append(Placement("tree_small", sr, sc))
            spent += ctx.cost("tree_small", sr.size)
            used[sr, sc] = True
            crown_small_px = max(int(round(CROWN_SMALL_M / ctx.res_m)), 1)
            _stamp_crown(covered, sr, sc, crown_small_px, ctx.shape)

    # ── Phase 3: Shade canopies (remaining ~25% budget) ──────────────────
    # Fill remaining hot open buildable ground not already under canopy.
    # Uses heat×population synergy score for maximum cost efficiency.
    remaining = budget_usd - spent
    if remaining < ctx.unit_cost("shade_canopy"):
        return placements

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
            # Trim to fit within budget
            affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
            if affordable_n > 0:
                placements.append(
                    Placement("shade_canopy", cr[:affordable_n], cc[:affordable_n])
                )

    return placements