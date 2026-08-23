from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "population-weighted shade maximizer v1"
DESCRIPTION = (
    "Maximise population-weighted UTCI drop (heat_relief_c) using a three-pass "
    "strategy: (1) medium trees at 4m spacing on a heat×population²×vulnerability "
    "surface targeting highest-impact corridors (65% budget), (2) shade canopies "
    "on open high-population pedestrian ground not covered by tree crowns (22% "
    "budget), (3) small trees filling remaining plantable gaps (13% budget). "
    "Population is squared in the priority surface to concentrate resources on "
    "densely-used pedestrian space, maximising the population-weighted fitness. "
    "Albedo actions (light_road, cool_roof) avoided entirely. Priority-tract "
    "boost of 0.20 preserves high equity_ratio."
)

# Priority surface weights - population squared to maximise pop-weighted score
WEIGHTS = {
    "heat_ta3pm":    0.38,
    "heat_hours":    0.12,
    "uhii":          0.08,
    "population":    0.30,   # high population weight for pop-weighted fitness
    "vulnerability": 0.12,
}

PRIORITY_BOOST       = 0.20   # stronger boost for top-quartile vulnerability tracts

# Budget allocation
TREE_MED_BUDGET_FRAC = 0.65
CANOPY_BUDGET_FRAC   = 0.22
TREE_SML_BUDGET_FRAC = 0.13

# Spacing
TREE_MED_SPACING_M   = 4.0   # tighter than parent's 5m for more coverage
TREE_SML_SPACING_M   = 3.0   # pack small trees tightly in gaps

# Crown radii for exclusion
TREE_MED_CROWN_R_M   = 4.0   # medium tree crown radius
TREE_SML_CROWN_R_M   = 2.5   # small tree crown radius


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
    Composite priority surface over exposure pixels.
    Population is weighted heavily since heat_relief_c is population-weighted.
    """
    mask = ctx.exposure

    # Normalise each component
    n_heat  = _norm(ctx.heat_ta3pm, mask)
    n_hours = _norm(ctx.heat_hours, mask)
    n_uhii  = _norm(ctx.heat_uhii, mask)
    n_pop   = _norm(ctx.population, mask)
    n_vuln  = _norm(ctx.vulnerability, mask)

    score = (
        WEIGHTS["heat_ta3pm"]    * n_heat
        + WEIGHTS["heat_hours"]  * n_hours
        + WEIGHTS["uhii"]        * n_uhii
        + WEIGHTS["population"]  * n_pop
        + WEIGHTS["vulnerability"] * n_vuln
    )

    # Additive boost for top-quartile vulnerability tracts
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)

    return np.where(mask, score, -np.inf)


def _canopy_priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    For canopy placement: weight population even more heavily since
    canopies provide immediate shade to pedestrians below them.
    """
    mask = ctx.exposure

    n_heat  = _norm(ctx.heat_ta3pm, mask)
    n_hours = _norm(ctx.heat_hours, mask)
    n_pop   = _norm(ctx.population, mask)
    n_vuln  = _norm(ctx.vulnerability, mask)

    score = (
        0.35 * n_heat
        + 0.10 * n_hours
        + 0.40 * n_pop      # very high population weight for canopies
        + 0.15 * n_vuln
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
    Greedy spaced placement: pick best candidate, suppress neighbourhood,
    repeat until limit reached or no candidates remain.
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
    """Mark a box of radius_px around each (r, c) as covered."""
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        r0 = max(0, r - radius_px)
        r1 = min(shape[0], r + radius_px + 1)
        c0 = max(0, c - radius_px)
        c1 = min(shape[1], c + radius_px + 1)
        covered[r0:r1, c0:c1] = True


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score     = priority_surface(ctx)
    c_score   = _canopy_priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0

    # Track used pixels and canopy coverage
    used    = np.zeros(ctx.shape, dtype=bool)
    covered = np.zeros(ctx.shape, dtype=bool)

    # Mark existing canopy as already covered
    covered[ctx.cdsm > 0.0] = True

    # ── 1. Medium street trees (65% budget, 4 m spacing) ────────────────────
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
        crown_px_med = max(int(round(TREE_MED_CROWN_R_M / ctx.res_m)), 1)
        _stamp_exclusion(covered, tr_m, tc_m, crown_px_med, ctx.shape)

    # ── 2. Shade canopies (22% budget, on open high-population ground) ───────
    canopy_budget = budget_usd * CANOPY_BUDGET_FRAC
    remaining_after_med = budget_usd - spent
    canopy_budget = min(canopy_budget, remaining_after_med)

    if canopy_budget >= ctx.unit_cost("shade_canopy"):
        # Open buildable ground not covered by tree crowns or existing canopy
        open_ground = ctx.buildable & ~covered & ~used

        n_canopy_max = ctx.affordable("shade_canopy", canopy_budget)
        n_canopy     = min(n_canopy_max, int(open_ground.sum()))

        cr, cc = _top_pixels(c_score, open_ground, n_canopy)
        if cr.size:
            placements.append(Placement("shade_canopy", cr, cc))
            spent += ctx.cost("shade_canopy", cr.size)
            used[cr, cc] = True
            # Mark canopy footprints as covered too
            covered[cr, cc] = True

    # ── 3. Small street trees (13% budget, fills remaining plantable gaps) ───
    sml_budget = budget_usd * TREE_SML_BUDGET_FRAC
    remaining_after_canopy = budget_usd - spent
    sml_budget = min(sml_budget, remaining_after_canopy)

    if sml_budget >= ctx.unit_cost("tree_small"):
        # Only plant small trees where not already covered/used
        cand_sml = ctx.plantable & ~used & ~covered

        n_sml_max  = ctx.affordable("tree_small", sml_budget)
        n_sml      = min(n_sml_max, int(cand_sml.sum()))

        spacing_sml_px = max(int(round(TREE_SML_SPACING_M / ctx.res_m)), 1)
        tr_s, tc_s = _greedy_spaced(score, cand_sml, spacing_sml_px, n_sml)

        if tr_s.size:
            placements.append(Placement("tree_small", tr_s, tc_s))
            spent += ctx.cost("tree_small", tr_s.size)
            used[tr_s, tc_s] = True
            crown_px_sml = max(int(round(TREE_SML_CROWN_R_M / ctx.res_m)), 1)
            _stamp_exclusion(covered, tr_s, tc_s, crown_px_sml, ctx.shape)

    # ── 4. Use any leftover budget on additional shade canopies ──────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("shade_canopy"):
        open_ground2 = ctx.buildable & ~covered & ~used
        n_extra_max  = ctx.affordable("shade_canopy", remaining)
        n_extra      = min(n_extra_max, int(open_ground2.sum()))

        er, ec = _top_pixels(c_score, open_ground2, n_extra)
        if er.size:
            placements.append(Placement("shade_canopy", er, ec))

    return placements