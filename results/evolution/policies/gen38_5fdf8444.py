from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-shade equity corridors v3"
DESCRIPTION = (
    "Maximise population-weighted UTCI relief for cell (1,1,0,1): "
    "equity_ratio>=1, access_gain>=0.32, cost_efficiency<40.55, greened>=0.20. "
    "Three-phase strategy: (1) Medium trees at 4m spacing with 55% budget on "
    "hottest most-populated corridors; (2) Small trees at 3m spacing gap-fill "
    "with 15% budget; (3) Shade canopies with 30% budget on remaining open "
    "ground using population×heat synergy scoring. Priority surface weights "
    "population 0.40, heat 0.35, vulnerability 0.10 with strong priority boost "
    "0.20 to maintain equity. Canopy-heavy phase 3 ensures high greened_pct. "
    "Reflective surfaces avoided entirely."
)

WEIGHTS = {
    "heat_ta3pm":    0.35,
    "population":    0.40,
    "heat_hours":    0.10,
    "uhii":          0.05,
    "vulnerability": 0.10,
}

PRIORITY_BOOST = 0.20

MEDIUM_BUDGET_FRAC = 0.55
SMALL_BUDGET_FRAC  = 0.15
# Remaining ~30% goes to shade canopies

MEDIUM_SPACING_M = 4.0
SMALL_SPACING_M  = 3.0

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
    Composite priority surface for tree placement.
    Positive only where ctx.exposure is True.
    High score = hot + heavily-populated pedestrian corridor.
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


def canopy_priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Synergy score for canopy placement: focus on pixels that are both
    hot AND populated, weighted by vulnerability for equity.
    Uses geometric mean of heat × population for synergy effect.
    """
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    # Geometric mean synergy: rewards pixels that score well on BOTH dimensions
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.40 * synergy
        + 0.25 * heat_n
        + 0.15 * pop_n
        + 0.10 * hours_n
        + 0.05 * uhii_n
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
    score         = priority_surface(ctx)
    canopy_score  = canopy_priority_surface(ctx)
    placements: list[Placement] = []
    spent         = 0.0
    used          = np.zeros(ctx.shape, dtype=bool)

    # ── Phase 1: Medium street trees (55% budget, 4 m spacing) ──────────
    # Medium trees deliver strongest per-tree UTCI benefit via large crown shade.
    # 4m tight spacing forms dense shade corridors on hottest, most-populated streets.
    medium_budget  = budget_usd * MEDIUM_BUDGET_FRAC
    n_medium_max   = ctx.affordable("tree_medium", medium_budget)
    cand_medium    = ctx.plantable & ~used
    n_medium       = min(n_medium_max, int(cand_medium.sum()))

    spacing_medium = max(int(round(MEDIUM_SPACING_M / ctx.res_m)), 1)
    mr, mc = _greedy_spaced(score, cand_medium, spacing_medium, n_medium)

    if mr.size:
        actual_cost = ctx.cost("tree_medium", mr.size)
        if spent + actual_cost <= budget_usd:
            placements.append(Placement("tree_medium", mr, mc))
            spent += actual_cost
            used[mr, mc] = True
        else:
            n_fit = ctx.affordable("tree_medium", budget_usd - spent)
            if n_fit > 0:
                mr, mc = mr[:n_fit], mc[:n_fit]
                placements.append(Placement("tree_medium", mr, mc))
                spent += ctx.cost("tree_medium", mr.size)
                used[mr, mc] = True

    # ── Phase 2: Small street trees (15% budget, 3 m spacing, gap-fill) ─
    # Fill gaps between mediums — 3x more trees per dollar than medium.
    # Very tight 3m spacing maximises shade density in remaining plantable gaps.
    small_budget = min(budget_usd * SMALL_BUDGET_FRAC, budget_usd - spent)
    sr = sc = np.array([], dtype=int)
    if small_budget >= ctx.unit_cost("tree_small"):
        cand_small  = ctx.plantable & ~used
        n_small_max = ctx.affordable("tree_small", small_budget)
        n_small     = min(n_small_max, int(cand_small.sum()))

        spacing_small = max(int(round(SMALL_SPACING_M / ctx.res_m)), 1)
        sr, sc = _greedy_spaced(score, cand_small, spacing_small, n_small)

        if sr.size:
            actual_cost = ctx.cost("tree_small", sr.size)
            if spent + actual_cost <= budget_usd:
                placements.append(Placement("tree_small", sr, sc))
                spent += actual_cost
                used[sr, sc] = True
            else:
                n_fit = ctx.affordable("tree_small", budget_usd - spent)
                if n_fit > 0:
                    sr, sc = sr[:n_fit], sc[:n_fit]
                    placements.append(Placement("tree_small", sr, sc))
                    spent += ctx.cost("tree_small", sr.size)
                    used[sr, sc] = True

    # ── Phase 3: Shade canopies (remaining ~30% budget, hottest open ground)
    # Generous canopy budget boosts cobenefit_greened_pct above 0.1976 threshold.
    # Use synergy score to target pixels that are both hot AND populated.
    remaining = budget_usd - spent
    if remaining < ctx.unit_cost("shade_canopy"):
        return placements

    # Stamp out crown footprints of all planted trees
    covered = np.zeros(ctx.shape, dtype=bool)
    # Also exclude existing canopy
    covered[ctx.cdsm > 0.0] = True

    if mr.size:
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)
    if sr.size:
        crown_small_px = max(int(round(CROWN_SMALL_M / ctx.res_m)), 1)
        _stamp_crown(covered, sr, sc, crown_small_px, ctx.shape)

    # Buildable pixels not under any canopy and not used
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
            n_fit = ctx.affordable("shade_canopy", budget_usd - spent)
            if n_fit > 0:
                placements.append(Placement("shade_canopy", cr[:n_fit], cc[:n_fit]))

    return placements