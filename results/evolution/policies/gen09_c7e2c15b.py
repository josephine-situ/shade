from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "dual-tree shade corridors v2"
DESCRIPTION = (
    "Maximise population-weighted UTCI drop (heat_relief_c) by deploying medium "
    "trees first at 5 m spacing on a heat×population priority surface, then "
    "filling remaining plantable ground with small trees at 4 m spacing, and "
    "finally using leftover budget on shade canopies over the hottest open "
    "pedestrian pixels. Budget split: 70% medium trees, 18% small trees, 12% "
    "canopies. Priority surface weights heat strongly (0.40 afternoon temp + "
    "0.15 heat-hours + 0.10 UHII) with population (0.25) and vulnerability "
    "(0.10) plus a 0.15 additive boost for top-quartile tracts. Albedo avoided."
)

WEIGHTS = {
    "heat_ta3pm":    0.40,
    "heat_hours":    0.15,
    "uhii":          0.10,
    "population":    0.25,
    "vulnerability": 0.10,
}

PRIORITY_BOOST       = 0.15   # additive boost for top-quartile vulnerability tracts
TREE_MED_BUDGET_FRAC = 0.70   # medium trees first
TREE_SML_BUDGET_FRAC = 0.18   # small trees fill gaps
CANOPY_BUDGET_FRAC   = 0.12   # shade canopies last

TREE_MED_SPACING_M   = 5.0
TREE_SML_SPACING_M   = 4.0


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
    """Composite priority surface weighted toward heat × population signal."""
    mask = ctx.exposure
    score = (
          WEIGHTS["heat_ta3pm"]    * _norm(ctx.heat_ta3pm,    mask)
        + WEIGHTS["heat_hours"]    * _norm(ctx.heat_hours,    mask)
        + WEIGHTS["uhii"]          * _norm(ctx.heat_uhii,     mask)
        + WEIGHTS["population"]    * _norm(ctx.population,    mask)
        + WEIGHTS["vulnerability"] * _norm(ctx.vulnerability, mask)
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Greedy spaced placement: pick best candidate, suppress neighbourhood."""
    rows, cols = np.nonzero(candidates)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    order = np.argsort(-score[rows, cols])
    rows, cols = rows[order], cols[order]

    taken = np.zeros(score.shape, dtype=bool)
    span = max(int(spacing_px), 1)
    pick_r: list[int] = []
    pick_c: list[int] = []

    for r, c in zip(rows, cols):
        if taken[r, c]:
            continue
        pick_r.append(int(r))
        pick_c.append(int(c))
        r0 = max(0, r - span)
        r1 = min(score.shape[0], r + span + 1)
        c0 = max(0, c - span)
        c1 = min(score.shape[1], c + span + 1)
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


def _stamp_exclusion(
    covered: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    radius_px: int,
    shape: tuple[int, int],
) -> None:
    """Mark a box of radius_px around each (r, c) in covered."""
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        r0 = max(0, r - radius_px)
        r1 = min(shape[0], r + radius_px + 1)
        c0 = max(0, c - radius_px)
        c1 = min(shape[1], c + radius_px + 1)
        covered[r0:r1, c0:c1] = True


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score = priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0

    # Track used pixels to prevent double-booking
    used = np.zeros(ctx.shape, dtype=bool)
    # Track canopy coverage (tree crowns + existing canopy)
    covered = np.zeros(ctx.shape, dtype=bool)
    # Mark existing canopy pixels as already covered
    covered[ctx.cdsm > 0.0] = True

    # ── 1. Medium street trees (70% budget, 5 m spacing) ────────────────
    med_budget  = budget_usd * TREE_MED_BUDGET_FRAC
    n_med_max   = ctx.affordable("tree_medium", med_budget)
    cand_med    = ctx.plantable & ~used

    spacing_med_px = max(int(round(TREE_MED_SPACING_M / ctx.res_m)), 1)
    n_med = min(n_med_max, int(cand_med.sum()))

    tr_m, tc_m = _greedy_spaced(score, cand_med, spacing_med_px, n_med)

    if tr_m.size:
        placements.append(Placement("tree_medium", tr_m, tc_m))
        spent += ctx.cost("tree_medium", tr_m.size)
        used[tr_m, tc_m] = True
        # Medium tree crown ~3.5 m radius → ~2 px at 2 m resolution
        _stamp_exclusion(covered, tr_m, tc_m, max(int(round(3.5 / ctx.res_m)), 1), ctx.shape)

    # ── 2. Small street trees (18% budget, 4 m spacing, fills gaps) ─────
    sml_budget = budget_usd * TREE_SML_BUDGET_FRAC
    # Don't exceed remaining budget
    remaining_after_med = budget_usd - spent
    sml_budget = min(sml_budget, remaining_after_med)

    n_sml_max  = ctx.affordable("tree_small", sml_budget)
    # Only plant small trees where medium trees don't already provide crown cover
    cand_sml   = ctx.plantable & ~used & ~covered

    spacing_sml_px = max(int(round(TREE_SML_SPACING_M / ctx.res_m)), 1)
    n_sml = min(n_sml_max, int(cand_sml.sum()))

    tr_s, tc_s = _greedy_spaced(score, cand_sml, spacing_sml_px, n_sml)

    if tr_s.size:
        placements.append(Placement("tree_small", tr_s, tc_s))
        spent += ctx.cost("tree_small", tr_s.size)
        used[tr_s, tc_s] = True
        # Small tree crown ~2 m radius → ~1 px at 2 m resolution
        _stamp_exclusion(covered, tr_s, tc_s, max(int(round(2.0 / ctx.res_m)), 1), ctx.shape)

    # ── 3. Shade canopies (remaining budget, hottest open ground) ────────
    remaining = budget_usd - spent
    if remaining <= 0.0:
        return placements

    # Open buildable ground: not already covered by canopy/trees, not used
    open_ground = ctx.buildable & ~covered & ~used

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(score, open_ground, n_canopy)
    if cr.size:
        placements.append(Placement("shade_canopy", cr, cc))
        # No need to track spend for last action - budget check via affordable

    return placements