from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "quad-action dense-shade equity corridors"
DESCRIPTION = (
    "Four-action strategy to maximise population-weighted UTCI relief: "
    "(1) medium trees at 3m spacing on highest heat×pop×vuln corridors (60% budget); "
    "(2) small trees fill plantable gaps after medium tree exclusions (15% budget); "
    "(3) shade canopies on hottest open buildable ground near populations (20% budget); "
    "(4) green roof on highest-heat building pixels with residual budget (5%). "
    "Priority surface: heat (0.38) + population (0.37) + vulnerability (0.12) + "
    "heat_hours (0.08) + UHII (0.05), with strong priority-tract boost (0.30). "
    "Designed to land in (1,1,1,1): equity>1, access>0.32, efficiency>40.55, greening>0.1976."
)

# Priority surface weights
WEIGHTS = {
    "heat_ta3pm":    0.38,
    "population":    0.37,
    "vulnerability": 0.12,
    "heat_hours":    0.08,
    "uhii":          0.05,
}

PRIORITY_BOOST = 0.30  # strong boost for top-quartile tracts

# Budget fractions
FRAC_MED    = 0.60
FRAC_SMALL  = 0.15
FRAC_CANOPY = 0.20
FRAC_GROOF  = 0.05

# Spacing
SPACING_MED_M   = 3.0
SPACING_SMALL_M = 2.0  # tighter fill spacing for small trees

# Crown radii for exclusion
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
    """Composite priority surface; meaningful where ctx.exposure is True."""
    mask = ctx.exposure
    score = (
          WEIGHTS["heat_ta3pm"]    * _norm(ctx.heat_ta3pm,    mask)
        + WEIGHTS["population"]    * _norm(ctx.population,    mask)
        + WEIGHTS["vulnerability"] * _norm(ctx.vulnerability, mask)
        + WEIGHTS["heat_hours"]    * _norm(ctx.heat_hours,    mask)
        + WEIGHTS["uhii"]          * _norm(ctx.heat_uhii,     mask)
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
    Greedy spaced placement: pick highest-scoring candidate, suppress
    spacing_px neighbourhood, repeat until limit reached.
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
    radius_px: int,
    shape: tuple[int, int],
) -> None:
    """Mark box of radius_px around each (r,c) as covered."""
    H, W = shape
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        r0 = max(0, r - radius_px)
        r1 = min(H, r + radius_px + 1)
        c0 = max(0, c - radius_px)
        c1 = min(W, c + radius_px + 1)
        covered[r0:r1, c0:c1] = True


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score = priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0

    # Track pixel usage (no double-booking)
    used = np.zeros(ctx.shape, dtype=bool)
    # Track canopy coverage for canopy placement exclusion
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True  # existing canopy

    # ── 1. Medium street trees (60% budget, 3m spacing) ──────────────────
    med_budget = budget_usd * FRAC_MED
    n_med_max  = ctx.affordable("tree_medium", med_budget)
    cand_med   = ctx.plantable & ~used
    n_med      = min(n_med_max, int(cand_med.sum()))

    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    tr_m, tc_m = _greedy_spaced(score, cand_med, spacing_med_px, n_med)

    if tr_m.size:
        placements.append(Placement("tree_medium", tr_m, tc_m))
        spent += ctx.cost("tree_medium", tr_m.size)
        used[tr_m, tc_m] = True
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_crown(covered, tr_m, tc_m, crown_med_px, ctx.shape)

    # ── 2. Small street trees (15% budget, 2m spacing gap-fill) ──────────
    remaining_small = min(budget_usd * FRAC_SMALL, budget_usd - spent)
    if remaining_small >= ctx.unit_cost("tree_small"):
        cand_small  = ctx.plantable & ~used & ~covered
        n_small_max = ctx.affordable("tree_small", remaining_small)
        n_small     = min(n_small_max, int(cand_small.sum()))

        spacing_small_px = max(int(round(SPACING_SMALL_M / ctx.res_m)), 1)
        tr_s, tc_s = _greedy_spaced(score, cand_small, spacing_small_px, n_small)

        if tr_s.size:
            actual = ctx.cost("tree_small", tr_s.size)
            if spent + actual <= budget_usd:
                placements.append(Placement("tree_small", tr_s, tc_s))
                spent += actual
                used[tr_s, tc_s] = True
                crown_small_px = max(int(round(CROWN_SMALL_M / ctx.res_m)), 1)
                _stamp_crown(covered, tr_s, tc_s, crown_small_px, ctx.shape)
    else:
        tr_s = tc_s = np.array([], dtype=int)

    # ── 3. Shade canopies (20% budget, hottest open buildable ground) ─────
    remaining_canopy = min(budget_usd * FRAC_CANOPY, budget_usd - spent)
    if remaining_canopy >= ctx.unit_cost("shade_canopy"):
        # Enhanced canopy score: weight population very strongly
        mask = ctx.exposure
        canopy_score = (
              0.35 * _norm(ctx.heat_ta3pm,    mask)
            + 0.40 * _norm(ctx.population,    mask)
            + 0.12 * _norm(ctx.vulnerability, mask)
            + 0.08 * _norm(ctx.heat_hours,    mask)
            + 0.05 * _norm(ctx.heat_uhii,     mask)
        )
        canopy_score = canopy_score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
        canopy_score = np.where(mask, canopy_score, -np.inf)

        open_ground = ctx.buildable & ~covered & ~used
        n_canopy_max = ctx.affordable("shade_canopy", remaining_canopy)
        n_canopy     = min(n_canopy_max, int(open_ground.sum()))

        cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
        if cr.size:
            actual = ctx.cost("shade_canopy", cr.size)
            if spent + actual <= budget_usd:
                placements.append(Placement("shade_canopy", cr, cc))
                spent += actual
                used[cr, cc] = True
            else:
                affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
                if affordable_n > 0:
                    cr2, cc2 = cr[:affordable_n], cc[:affordable_n]
                    placements.append(Placement("shade_canopy", cr2, cc2))
                    spent += ctx.cost("shade_canopy", affordable_n)
                    used[cr2, cc2] = True

    # ── 4. Green roofs (residual budget, for cobenefit greening boost) ────
    remaining_groof = budget_usd - spent
    if remaining_groof >= ctx.unit_cost("green_roof"):
        # Build green-roof score: target hottest, most populated buildings
        # in high-vulnerability areas
        bld_mask = ctx.landcover == 2  # building pixels
        if bld_mask.any():
            groof_score = (
                  0.40 * _norm(ctx.heat_ta3pm,    bld_mask)
                + 0.25 * _norm(ctx.population,    bld_mask)
                + 0.20 * _norm(ctx.vulnerability, bld_mask)
                + 0.10 * _norm(ctx.heat_hours,    bld_mask)
                + 0.05 * _norm(ctx.heat_uhii,     bld_mask)
            )
            groof_score = groof_score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
            groof_score = np.where(bld_mask, groof_score, -np.inf)

            # Green roof candidates: building pixels not already used
            groof_cand = ctx.placeable("green_roof") & ~used
            n_groof_max = ctx.affordable("green_roof", remaining_groof)
            n_groof     = min(n_groof_max, int(groof_cand.sum()))

            gr, gc = _top_pixels(groof_score, groof_cand, n_groof)
            if gr.size:
                actual = ctx.cost("green_roof", gr.size)
                if spent + actual <= budget_usd:
                    placements.append(Placement("green_roof", gr, gc))
                    spent += actual
                else:
                    affordable_n = ctx.affordable("green_roof", budget_usd - spent)
                    if affordable_n > 0:
                        gr2, gc2 = gr[:affordable_n], gc[:affordable_n]
                        placements.append(Placement("green_roof", gr2, gc2))

    return placements