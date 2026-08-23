from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-equity dense-medium tree corridors v2"
DESCRIPTION = (
    "Maximise heat_relief_c for cell (1,0,0,1) using a synergy-based priority "
    "surface (geometric mean of heat×population) combined with strong vulnerability "
    "weighting. Strategy: (1) Medium trees at 4m spacing covering hottest corridors "
    "(65% budget); (2) Small trees filling gaps between medium crowns (20% budget); "
    "(3) Shade canopies on remaining hot open ground (15% budget). "
    "Priority surface: synergy(heat×pop) 0.40, heat_ta3pm 0.20, vulnerability 0.20, "
    "population 0.10, heat_hours 0.07, UHII 0.03. Strong priority-tract boost (0.30) "
    "ensures equity_ratio >> 1. Avoids reflective surfaces (albedo trap)."
)

# Priority-tract boost — strong to ensure equity_ratio > 1
PRIORITY_BOOST = 0.30

# Budget fractions
FRAC_MED    = 0.65
FRAC_SML    = 0.20
FRAC_CANOPY = 0.15

# Spacing parameters (tight for maximum shade coverage)
SPACING_MED_M = 4.0   # tight but not as extreme as 3m
SPACING_SML_M = 2.5   # very tight gap-fill

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
    """
    Synergy-based composite priority surface with strong vulnerability weighting.
    Targets pixels that are simultaneously hot AND populated in vulnerable tracts.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    # Synergy term: geometric mean rewards pixels that are BOTH hot AND populated
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    # Triple synergy: heat × population × vulnerability
    triple_synergy = np.cbrt(np.clip(heat_n * pop_n * vuln_n, 0.0, None))

    score = (
          0.35 * synergy
        + 0.15 * triple_synergy
        + 0.20 * heat_n
        + 0.15 * vuln_n
        + 0.10 * pop_n
        + 0.03 * hours_n
        + 0.02 * uhii_n
    )

    # Strong boost for top-quartile vulnerability tracts
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def canopy_priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Separate priority surface for shade canopy placement.
    Emphasises heat × population synergy and vulnerability for maximum impact.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)

    # Stronger synergy weight for canopies
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))
    triple_synergy = np.cbrt(np.clip(heat_n * pop_n * vuln_n, 0.0, None))

    score = (
          0.40 * synergy
        + 0.15 * triple_synergy
        + 0.20 * heat_n
        + 0.15 * vuln_n
        + 0.05 * pop_n
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


def _stamp_exclusion(
    covered: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    radius_px: int,
    shape: tuple[int, int],
) -> None:
    """Mark a box of radius_px around each (r, c) as covered."""
    H, W = shape
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        r0 = max(0, r - radius_px)
        r1 = min(H, r + radius_px + 1)
        c0 = max(0, c - radius_px)
        c1 = min(W, c + radius_px + 1)
        covered[r0:r1, c0:c1] = True


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    tree_score   = priority_surface(ctx)
    canopy_score = canopy_priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0

    # Track which pixels are already used (no double-booking)
    used    = np.zeros(ctx.shape, dtype=bool)
    # Track canopy coverage (existing + new trees)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # existing canopy already covered

    # ── 1. Medium street trees ───────────────────────────────────────────
    # 65% of budget, 4m spacing for dense shade corridors
    # Synergy+vulnerability score ensures we target hot+populated+vulnerable corridors
    med_budget = budget_usd * FRAC_MED
    n_med_max  = ctx.affordable("tree_medium", med_budget)
    cand_med   = ctx.plantable & ~used

    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    n_med = min(n_med_max, int(cand_med.sum()))

    tr_m, tc_m = _greedy_spaced(tree_score, cand_med, spacing_med_px, n_med)

    if tr_m.size:
        placements.append(Placement("tree_medium", tr_m, tc_m))
        spent += ctx.cost("tree_medium", tr_m.size)
        used[tr_m, tc_m] = True
        # Crown exclusion for subsequent placements
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_exclusion(covered, tr_m, tc_m, crown_med_px, ctx.shape)

    # ── 2. Small street trees ────────────────────────────────────────────
    # 20% of budget, 2.5m spacing, fills crown gaps between medium trees
    # in high-priority areas
    sml_budget_target = budget_usd * FRAC_SML
    remaining_for_sml = budget_usd - spent
    sml_budget = min(sml_budget_target, remaining_for_sml)

    if sml_budget > ctx.unit_cost("tree_small"):
        n_sml_max = ctx.affordable("tree_small", sml_budget)
        # Candidates: plantable, not used, not under existing or new medium crown
        cand_sml = ctx.plantable & ~used & ~covered

        spacing_sml_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)
        n_sml = min(n_sml_max, int(cand_sml.sum()))

        tr_s, tc_s = _greedy_spaced(tree_score, cand_sml, spacing_sml_px, n_sml)

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
    if remaining <= ctx.unit_cost("shade_canopy"):
        return placements

    # Open ground: buildable, no existing or new canopy coverage, not used
    open_ground = ctx.buildable & ~covered & ~used

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy = min(n_canopy_max, int(open_ground.sum()))

    if n_canopy > 0:
        cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
        if cr.size:
            # Safety check: ensure we don't overspend
            actual_cost = ctx.cost("shade_canopy", cr.size)
            if spent + actual_cost <= budget_usd + 0.01:
                placements.append(Placement("shade_canopy", cr, cc))
            else:
                # Trim to fit budget
                affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
                if affordable_n > 0:
                    placements.append(
                        Placement("shade_canopy", cr[:affordable_n], cc[:affordable_n])
                    )

    return placements