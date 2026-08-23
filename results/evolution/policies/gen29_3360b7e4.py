from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "dense-shade corridors v2"
DESCRIPTION = (
    "Maximise heat_relief_c via aggressive shade deployment: "
    "(1) Medium street trees at 3m spacing on highest heat×population×vulnerability "
    "corridors (70% budget) — tighter spacing than parent for denser canopy cover; "
    "(2) Shade canopies on all remaining hot open buildable ground (30% budget). "
    "Small trees skipped — budget redirected to more impactful medium trees and "
    "canopies. Priority surface weights: afternoon heat (0.45), population (0.30), "
    "vulnerability (0.15), heat-hours (0.07), UHII (0.03). Priority-tract boost "
    "of 0.25 ensures equity_ratio > 1. Reflective surfaces avoided entirely."
)

# Priority surface weights — heat + population dominant, vulnerability for equity
WEIGHTS = {
    "heat_ta3pm":    0.45,
    "population":    0.30,
    "vulnerability": 0.15,
    "heat_hours":    0.07,
    "uhii":          0.03,
}

PRIORITY_BOOST = 0.25   # stronger boost for top-quartile tracts → equity_ratio > 1

# Budget fractions — no small trees, redirect to medium trees + canopies
FRAC_MED    = 0.70
FRAC_CANOPY = 0.30

# Tighter spacing for denser coverage
SPACING_MED_M = 3.0   # 3m → 1-2 pixel spacing at 2m resolution

# Crown radius for post-tree canopy exclusion
CROWN_MED_M = 3.5


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
    Composite priority surface for tree placement.
    High score = hot afternoon + populated + vulnerable pedestrian corridor.
    """
    mask = ctx.exposure
    score = (
          WEIGHTS["heat_ta3pm"]    * _norm(ctx.heat_ta3pm,    mask)
        + WEIGHTS["population"]    * _norm(ctx.population,    mask)
        + WEIGHTS["vulnerability"] * _norm(ctx.vulnerability, mask)
        + WEIGHTS["heat_hours"]    * _norm(ctx.heat_hours,    mask)
        + WEIGHTS["uhii"]          * _norm(ctx.heat_uhii,     mask)
    )
    # Boost top-quartile vulnerability tracts
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def canopy_priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Separate priority surface for shade canopy placement.
    Uses synergy of heat × population to maximise person-degC relief.
    """
    mask = ctx.exposure
    heat_n = _norm(ctx.heat_ta3pm,    mask)
    pop_n  = _norm(ctx.population,    mask)
    vuln_n = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,   mask)

    # Geometric mean synergy for pixels that are simultaneously hot + populated
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.45 * synergy
        + 0.25 * heat_n
        + 0.15 * vuln_n
        + 0.10 * pop_n
        + 0.05 * hours_n
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
    suppress a spacing_px-radius neighbourhood, repeat until limit reached.
    """
    rows, cols = np.nonzero(candidates)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    order = np.argsort(-score[rows, cols])
    rows, cols = rows[order], cols[order]

    taken = np.zeros(score.shape, dtype=bool)
    span = max(int(spacing_px), 1)
    pick_r: list[int] = []
    pick_c: list[int] = []
    H, W = score.shape

    for r, c in zip(rows, cols):
        if taken[r, c]:
            continue
        pick_r.append(int(r))
        pick_c.append(int(c))
        r0 = max(0, r - span)
        r1 = min(H, r + span + 1)
        c0 = max(0, c - span)
        c1 = min(W, c + span + 1)
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
        r0 = max(0, r - crown_px)
        r1 = min(H, r + crown_px + 1)
        c0 = max(0, c - crown_px)
        c1 = min(W, c + crown_px + 1)
        covered[r0:r1, c0:c1] = True


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    tree_score   = priority_surface(ctx)
    canopy_score = canopy_priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0

    # Track used pixels (no double-booking)
    used    = np.zeros(ctx.shape, dtype=bool)
    # Track canopy coverage (existing + new trees)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # existing canopy already blocked

    # ── 1. Medium street trees (70% budget, 3m spacing) ─────────────────
    # Tighter spacing vs parent → denser shade corridors → stronger UTCI drop.
    # Heat+population weights maximise population-weighted UTCI relief.
    med_budget  = budget_usd * FRAC_MED
    n_med_max   = ctx.affordable("tree_medium", med_budget)
    cand_med    = ctx.plantable & ~used

    spacing_px  = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    n_med       = min(n_med_max, int(cand_med.sum()))

    tr_m, tc_m = _greedy_spaced(tree_score, cand_med, spacing_px, n_med)

    if tr_m.size:
        placements.append(Placement("tree_medium", tr_m, tc_m))
        spent += ctx.cost("tree_medium", tr_m.size)
        used[tr_m, tc_m] = True
        # Mark crown exclusion zones for canopy placement
        crown_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_crown(covered, tr_m, tc_m, crown_px, ctx.shape)

    # ── 2. Shade canopies (remaining budget, hottest open ground) ────────
    # Fill all remaining buildable open ground not already shaded by trees
    # or existing canopy. Use heat×population synergy score for placement.
    remaining = budget_usd - spent
    if remaining <= 0.0:
        return placements

    # Open buildable ground: not covered by canopy, not used by trees
    open_ground  = ctx.buildable & ~covered & ~used

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
    if cr.size:
        actual_cost = ctx.cost("shade_canopy", cr.size)
        if spent + actual_cost <= budget_usd + 0.01:
            placements.append(Placement("shade_canopy", cr, cc))
        else:
            # Trim to fit within budget
            affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
            if affordable_n > 0:
                placements.append(
                    Placement("shade_canopy", cr[:affordable_n], cc[:affordable_n])
                )

    return placements