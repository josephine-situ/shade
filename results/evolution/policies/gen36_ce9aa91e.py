from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-triple-shade heat-equity v3"
DESCRIPTION = (
    "Three-phase shade maximisation targeting cell (1,1,1,1): "
    "(1) Medium trees at 4m spacing on highest heat×population synergy corridors "
    "consume 55% of budget — geometric mean synergy focuses spend where UTCI drop "
    "× pedestrian count is maximised; "
    "(2) Small trees at 3m spacing fill remaining plantable gaps (20% budget) — "
    "3× more trees per dollar expands cobenefit_greened_pct; "
    "(3) Shade canopies on hottest remaining open buildable ground (25% budget). "
    "Priority surface uses heat×pop geometric synergy (0.50), heat_ta3pm (0.20), "
    "vulnerability (0.15), heat_hours (0.10), UHII (0.05). "
    "Priority-tract boost 0.15 for equity_ratio > 1. Reflective surfaces avoided."
)

# ── Priority surface weights ─────────────────────────────────────────────────
PRIORITY_BOOST = 0.15   # boost top-quartile tracts for equity

# Budget fractions
FRAC_MED    = 0.55
FRAC_SMALL  = 0.20
# Remainder (~25%) → shade canopies

# Spacing for greedy placement
SPACING_MED_M   = 4.0    # tighter than parent for denser corridors
SPACING_SMALL_M = 3.0    # gap-fill between mediums

# Crown radii for post-placement canopy exclusion
CROWN_MED_M   = 3.5
CROWN_SMALL_M = 2.0


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
    """
    Composite priority surface for tree/canopy placement.
    Geometric mean synergy of heat × population maximises person-weighted UTCI drop.
    """
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    # Geometric mean synergy: only pixels hot AND populated get high score
    # This directly proxies the population-weighted UTCI fitness metric
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.50 * synergy
        + 0.20 * heat_n
        + 0.15 * vuln_n
        + 0.10 * hours_n
        + 0.05 * uhii_n
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
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
    """Mark a box of radius crown_px around each (r, c) as covered (vectorised)."""
    H, W = shape
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        r0 = max(0, r - crown_px);  r1 = min(H, r + crown_px + 1)
        c0 = max(0, c - crown_px);  c1 = min(W, c + crown_px + 1)
        covered[r0:r1, c0:c1] = True


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score = priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0

    # Track used pixels globally (no double-booking)
    used    = np.zeros(ctx.shape, dtype=bool)
    # Track covered ground (existing + new canopy) for canopy phase
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # existing canopy already blocks placement

    # ── Phase 1: Medium street trees (55% budget, 4m spacing) ────────────
    # Medium trees deliver the largest single-unit UTCI drop per tree.
    # 4m spacing (2px at 2m resolution) creates dense, overlapping shade corridors.
    med_budget  = budget_usd * FRAC_MED
    n_med_max   = ctx.affordable("tree_medium", med_budget)
    cand_med    = ctx.plantable & ~used
    n_med       = min(n_med_max, int(cand_med.sum()))

    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    mr, mc = _greedy_spaced(score, cand_med, spacing_med_px, n_med)

    if mr.size:
        actual_cost = ctx.cost("tree_medium", mr.size)
        # Safety trim if somehow over budget
        if actual_cost > med_budget + 0.01:
            n_fit = ctx.affordable("tree_medium", med_budget)
            mr, mc = mr[:n_fit], mc[:n_fit]
            actual_cost = ctx.cost("tree_medium", mr.size)
        placements.append(Placement("tree_medium", mr, mc))
        spent += actual_cost
        used[mr, mc] = True
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)

    # ── Phase 2: Small street trees (20% budget, 3m spacing, gap-fill) ───
    # Small trees are 3× cheaper per tree — dramatically boosts tree count,
    # cobenefit_greened_pct, and fills gaps between medium tree crowns.
    small_budget = min(budget_usd * FRAC_SMALL, budget_usd - spent)
    sr = sc = np.array([], dtype=int)

    if small_budget >= ctx.unit_cost("tree_small"):
        cand_small    = ctx.plantable & ~used
        n_small_max   = ctx.affordable("tree_small", small_budget)
        n_small       = min(n_small_max, int(cand_small.sum()))

        spacing_small_px = max(int(round(SPACING_SMALL_M / ctx.res_m)), 1)
        sr, sc = _greedy_spaced(score, cand_small, spacing_small_px, n_small)

        if sr.size:
            actual_cost_s = ctx.cost("tree_small", sr.size)
            if spent + actual_cost_s > budget_usd + 0.01:
                n_fit_s = ctx.affordable("tree_small", budget_usd - spent)
                sr, sc = sr[:n_fit_s], sc[:n_fit_s]
                actual_cost_s = ctx.cost("tree_small", sr.size)
            if sr.size:
                placements.append(Placement("tree_small", sr, sc))
                spent += actual_cost_s
                used[sr, sc] = True
                crown_small_px = max(int(round(CROWN_SMALL_M / ctx.res_m)), 1)
                _stamp_crown(covered, sr, sc, crown_small_px, ctx.shape)

    # ── Phase 3: Shade canopies (remaining budget, hottest open ground) ───
    # Cover remaining hot buildable ground not already shaded by trees.
    # This maximises cobenefit_greened_pct and captures residual UTCI gain.
    remaining = budget_usd - spent
    if remaining < ctx.unit_cost("shade_canopy"):
        return placements

    # Open buildable ground: not covered by any canopy, not used
    open_ground  = ctx.buildable & ~covered & ~used

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(score, open_ground, n_canopy)
    if cr.size:
        actual_cost_c = ctx.cost("shade_canopy", cr.size)
        if spent + actual_cost_c > budget_usd + 0.01:
            n_fit_c = ctx.affordable("shade_canopy", budget_usd - spent)
            cr, cc = cr[:n_fit_c], cc[:n_fit_c]
        if cr.size:
            placements.append(Placement("shade_canopy", cr, cc))

    return placements