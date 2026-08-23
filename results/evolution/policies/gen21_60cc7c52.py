from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "triple-action heat-equity corridors v1"
DESCRIPTION = (
    "Maximise heat_relief_c (population-weighted UTCI drop) with a triple-action "
    "strategy: (1) medium trees at 4 m spacing on highest heat×population×vulnerability "
    "corridors (65% budget), (2) small trees filling crown gaps at 3 m spacing (20% "
    "budget), (3) shade canopies on remaining hot open ground (15% budget). "
    "Priority surface strongly weights afternoon heat (0.40) and population (0.30) "
    "with vulnerability (0.15), heat-hours (0.10), UHII (0.05). Top-quartile tract "
    "boost of 0.20 ensures high equity_ratio. Reflective surfaces avoided entirely."
)

# Priority surface weights — heat + population dominant for UTCI efficiency
WEIGHTS = {
    "heat_ta3pm":    0.40,
    "population":    0.30,
    "vulnerability": 0.15,
    "heat_hours":    0.10,
    "uhii":          0.05,
}

PRIORITY_BOOST = 0.20   # strong boost for top-quartile tracts → equity_ratio > 1

# Budget fractions
FRAC_MED    = 0.65
FRAC_SML    = 0.20
FRAC_CANOPY = 0.15

# Spacing
SPACING_MED_M = 4.0   # tight for dense shade corridors
SPACING_SML_M = 3.0   # even tighter for gap-fill small trees

# Crown radii for exclusion
CROWN_MED_M = 3.5
CROWN_SML_M = 2.0


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
    """Composite priority surface; only meaningful where ctx.exposure is True."""
    mask = ctx.exposure
    score = (
          WEIGHTS["heat_ta3pm"]    * _norm(ctx.heat_ta3pm,    mask)
        + WEIGHTS["population"]    * _norm(ctx.population,    mask)
        + WEIGHTS["vulnerability"] * _norm(ctx.vulnerability, mask)
        + WEIGHTS["heat_hours"]    * _norm(ctx.heat_hours,    mask)
        + WEIGHTS["uhii"]          * _norm(ctx.heat_uhii,     mask)
    )
    # Strong boost for top-quartile vulnerability tracts
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
    score = priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0

    # Track which pixels are already used (no double-booking)
    used = np.zeros(ctx.shape, dtype=bool)
    # Track canopy coverage (existing + new trees)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # existing canopy already covered

    # ── 1. Medium street trees ───────────────────────────────────────────
    # 65% of budget, 4 m spacing for dense shade corridors on hottest ground
    med_budget = budget_usd * FRAC_MED
    n_med_max  = ctx.affordable("tree_medium", med_budget)
    cand_med   = ctx.plantable & ~used

    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    n_med = min(n_med_max, int(cand_med.sum()))

    tr_m, tc_m = _greedy_spaced(score, cand_med, spacing_med_px, n_med)

    if tr_m.size:
        placements.append(Placement("tree_medium", tr_m, tc_m))
        spent += ctx.cost("tree_medium", tr_m.size)
        used[tr_m, tc_m] = True
        # Crown exclusion for small tree / canopy placement
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_exclusion(covered, tr_m, tc_m, crown_med_px, ctx.shape)

    # ── 2. Small street trees ────────────────────────────────────────────
    # 20% of budget, 3 m spacing, fills gaps between medium trees
    # Only plant where medium tree crowns don't already shade the spot
    sml_budget_target = budget_usd * FRAC_SML
    remaining_for_sml = budget_usd - spent
    sml_budget = min(sml_budget_target, remaining_for_sml)

    n_sml_max = ctx.affordable("tree_small", sml_budget)
    # Candidates: plantable, not used, not already under a medium crown
    cand_sml  = ctx.plantable & ~used & ~covered

    spacing_sml_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)
    n_sml = min(n_sml_max, int(cand_sml.sum()))

    tr_s, tc_s = _greedy_spaced(score, cand_sml, spacing_sml_px, n_sml)

    if tr_s.size:
        placements.append(Placement("tree_small", tr_s, tc_s))
        spent += ctx.cost("tree_small", tr_s.size)
        used[tr_s, tc_s] = True
        # Small tree crown exclusion
        crown_sml_px = max(int(round(CROWN_SML_M / ctx.res_m)), 1)
        _stamp_exclusion(covered, tr_s, tc_s, crown_sml_px, ctx.shape)

    # ── 3. Shade canopies ────────────────────────────────────────────────
    # Remaining budget; fill hottest open buildable ground not already shaded
    remaining = budget_usd - spent
    if remaining <= 0.0:
        return placements

    # Open ground: buildable, no existing or new canopy coverage, not used
    open_ground = ctx.buildable & ~covered & ~used

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(score, open_ground, n_canopy)
    if cr.size:
        placements.append(Placement("shade_canopy", cr, cc))

    return placements