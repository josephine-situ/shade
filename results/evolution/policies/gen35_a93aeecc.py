from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-corridors heat-equity v2"
DESCRIPTION = (
    "Maximise heat_relief_c in cell (1,1,0,0): "
    "(1) Medium street trees planted greedily on a multiplicative synergy "
    "priority surface (heat × population × vulnerability) with a strong "
    "priority-tract boost at 5m spacing — balancing corridor density with "
    "per-tree impact; "
    "(2) Shade canopies on remaining hot open buildable ground using remaining "
    "budget. Budget split: 75% medium trees, 25% shade canopies. "
    "Uses geometric-mean synergy to concentrate resources where heat AND "
    "population AND vulnerability all co-occur, maximising person-degC relief "
    "and equity_ratio simultaneously. Reflective surfaces avoided entirely "
    "(albedo trap). cobenefit_greened_pct kept moderate to stay in target cell."
)

# Synergy weights for tree priority
TREE_WEIGHTS = {
    "heat_ta3pm":    0.40,
    "population":    0.30,
    "vulnerability": 0.15,
    "heat_hours":    0.10,
    "uhii":          0.05,
}

# Canopy uses similar but slightly more heat-focused weights
CANOPY_WEIGHTS = {
    "heat_ta3pm":    0.50,
    "population":    0.25,
    "vulnerability": 0.15,
    "heat_hours":    0.07,
    "uhii":          0.03,
}

PRIORITY_BOOST = 0.30   # strong equity boost for top-quartile tracts

FRAC_TREE   = 0.75
FRAC_CANOPY = 0.25

TREE_SPACING_M = 5.0    # 5m spacing: denser than 7m, less crowded than 3m
CROWN_M        = 3.5    # medium tree crown radius for canopy exclusion


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1] over mask; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def _synergy_surface(
    ctx: PlanningContext,
    weights: dict,
    priority_boost: float,
    use_geometric: bool = True,
) -> np.ndarray:
    """
    Composite priority surface combining additive weighted sum with
    an optional geometric-mean synergy term for heat × population.
    
    Geometric mean ensures we only score high where BOTH heat AND
    population are high — maximises person-degC relief.
    """
    mask = ctx.exposure
    
    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)
    
    if use_geometric:
        # Geometric mean of heat × population: high only when both are high
        synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))
        # Blend synergy into the weighted score
        score = (
            weights["heat_ta3pm"]    * synergy          # synergy replaces raw heat
            + weights["population"]  * synergy          # double-weight synergy
            + weights["vulnerability"] * vuln_n
            + weights["heat_hours"]  * hours_n
            + weights["uhii"]        * uhii_n
        )
        # Normalise to [0,1] range before boost
        score_max = sum(weights.values())
        score = score / score_max
    else:
        score = (
            weights["heat_ta3pm"]    * heat_n
            + weights["population"]  * pop_n
            + weights["vulnerability"] * vuln_n
            + weights["heat_hours"]  * hours_n
            + weights["uhii"]        * uhii_n
        )
    
    # Equity boost: strong additive bump for top-quartile vulnerability tracts
    score = score + np.where(ctx.priority, priority_boost, 0.0)
    return np.where(mask, score, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy spaced placement: pick highest-scoring candidate pixel,
    suppress spacing_px-radius neighbourhood, repeat until limit reached.
    """
    rows, cols = np.nonzero(candidates)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    order = np.argsort(-score[rows, cols])
    rows, cols = rows[order], cols[order]

    taken = np.zeros(score.shape, dtype=bool)
    span  = max(int(spacing_px), 1)
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
    tree_score   = _synergy_surface(ctx, TREE_WEIGHTS,   PRIORITY_BOOST, use_geometric=True)
    canopy_score = _synergy_surface(ctx, CANOPY_WEIGHTS, PRIORITY_BOOST, use_geometric=False)

    placements: list[Placement] = []
    spent = 0.0

    # Track used pixels (no double-booking)
    used    = np.zeros(ctx.shape, dtype=bool)
    # Track canopy coverage (existing + new trees) for canopy exclusion
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # existing canopy already blocked

    # ── 1. Medium street trees (75% budget, 5m spacing) ─────────────────
    # 5m spacing: denser than 7m inspiration (more shade) but less than 3m
    # (avoids over-clustering on few high-score pixels).
    # Synergy surface concentrates trees where heat + population co-occur.
    tree_budget = budget_usd * FRAC_TREE
    n_med_max   = ctx.affordable("tree_medium", tree_budget)
    cand_med    = ctx.plantable & ~used

    spacing_px  = max(int(round(TREE_SPACING_M / ctx.res_m)), 1)
    n_med       = min(n_med_max, int(cand_med.sum()))

    tr, tc = _greedy_spaced(tree_score, cand_med, spacing_px, n_med)

    if tr.size:
        placements.append(Placement("tree_medium", tr, tc))
        spent += ctx.cost("tree_medium", tr.size)
        used[tr, tc] = True
        crown_px = max(int(round(CROWN_M / ctx.res_m)), 1)
        _stamp_crown(covered, tr, tc, crown_px, ctx.shape)

    # ── 2. Shade canopies (remaining budget) ─────────────────────────────
    # Infill hottest open buildable ground not already shaded.
    # Heat-dominant canopy score maximises direct UTCI reduction.
    remaining = budget_usd - spent
    if remaining <= 0.0:
        return placements

    open_ground  = ctx.buildable & ~covered & ~used

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
    if cr.size:
        actual_cost = ctx.cost("shade_canopy", cr.size)
        if spent + actual_cost <= budget_usd + 0.01:
            placements.append(Placement("shade_canopy", cr, cc))
        else:
            affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
            if affordable_n > 0:
                placements.append(
                    Placement("shade_canopy", cr[:affordable_n], cc[:affordable_n])
                )

    return placements