from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-corridor heat-equity v3"
DESCRIPTION = (
    "Improved champion for cell (1,1,0,1): equity>=1, access>=0.32, "
    "cost_efficiency<40.55, cobenefit>=0.20. "
    "Four-phase shade strategy: (1) Medium trees at 2m spacing on highest "
    "heat×population×vulnerability synergy corridors (55% budget); "
    "(2) Small trees filling crown gaps at 2m spacing (20% budget); "
    "(3) Shade canopies on remaining open hot ground (15% budget); "
    "(4) Green roof on buildings in hot priority tracts (10% budget). "
    "Priority surface: geometric mean synergy(heat×pop) 0.45, heat_ta3pm 0.25, "
    "population 0.12, vulnerability 0.10, heat_hours 0.05, UHII 0.03. "
    "Priority-tract boost 0.25 preserves equity_ratio >= 1. "
    "Tighter tree spacing (2m) for denser canopy coverage. "
    "Green roof adds cobenefit_greened_pct and raises fitness via roof cooling. "
    "Reflective surfaces avoided entirely (albedo trap)."
)

PRIORITY_BOOST = 0.25

FRAC_MED        = 0.55
FRAC_SML        = 0.20
FRAC_CANOPY     = 0.15
FRAC_GREEN_ROOF = 0.10

SPACING_MED_M = 2.0   # tighter spacing for denser canopy
SPACING_SML_M = 2.0

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
    Synergy-based composite priority surface.
    Geometric mean of heat × population rewards pixels that are simultaneously
    hot AND populated — these yield maximum person-degC relief per intervention.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    # Geometric mean synergy: high only if BOTH heat and population are high
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.45 * synergy
        + 0.25 * heat_n
        + 0.12 * pop_n
        + 0.10 * vuln_n
        + 0.05 * hours_n
        + 0.03 * uhii_n
    )

    # Boost top-quartile vulnerability tracts
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def canopy_priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for shade canopy and green roof placement.
    Even stronger synergy weight for maximum person-degC relief.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.55 * synergy
        + 0.20 * heat_n
        + 0.12 * pop_n
        + 0.08 * vuln_n
        + 0.05 * hours_n
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def roof_priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for green roof placement.
    Target hot buildings in high-vulnerability, high-population tracts.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    score = (
          0.40 * heat_n
        + 0.25 * vuln_n
        + 0.20 * pop_n
        + 0.15 * uhii_n
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


def _safe_affordable(
    ctx: PlanningContext,
    action: str,
    budget_remaining: float,
    candidate_count: int,
) -> int:
    """Return the safe number of placements given budget and candidates."""
    if budget_remaining <= 0.0:
        return 0
    n = ctx.affordable(action, budget_remaining)
    return min(n, candidate_count)


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    tree_score   = priority_surface(ctx)
    canopy_score = canopy_priority_surface(ctx)
    roof_score   = roof_priority_surface(ctx)

    placements: list[Placement] = []
    spent = 0.0

    used    = np.zeros(ctx.shape, dtype=bool)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # pre-existing canopy

    # ── 1. Medium street trees (55% budget, 2m spacing) ──────────────────
    # Tighter 2m spacing gives denser shade corridors for stronger UTCI drop.
    med_budget = budget_usd * FRAC_MED
    cand_med   = ctx.plantable & ~used
    n_med_max  = _safe_affordable(ctx, "tree_medium", med_budget, int(cand_med.sum()))

    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    tr_m, tc_m = _greedy_spaced(tree_score, cand_med, spacing_med_px, n_med_max)

    if tr_m.size:
        actual_cost = ctx.cost("tree_medium", tr_m.size)
        if spent + actual_cost > budget_usd:
            n_fit = ctx.affordable("tree_medium", budget_usd - spent)
            tr_m, tc_m = tr_m[:n_fit], tc_m[:n_fit]
            actual_cost = ctx.cost("tree_medium", tr_m.size)
        if tr_m.size:
            placements.append(Placement("tree_medium", tr_m, tc_m))
            spent += actual_cost
            used[tr_m, tc_m] = True
            crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
            _stamp_exclusion(covered, tr_m, tc_m, crown_med_px, ctx.shape)

    # ── 2. Small street trees (20% budget, 2m spacing, gap-filling) ──────
    # Fill canopy gaps with small trees — still much better than paving.
    sml_budget = min(budget_usd * FRAC_SML, budget_usd - spent)
    if sml_budget > ctx.unit_cost("tree_small"):
        # Only plant where not already covered by medium tree crowns
        cand_sml  = ctx.plantable & ~used & ~covered
        n_sml_max = _safe_affordable(ctx, "tree_small", sml_budget, int(cand_sml.sum()))

        spacing_sml_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)
        tr_s, tc_s = _greedy_spaced(tree_score, cand_sml, spacing_sml_px, n_sml_max)

        if tr_s.size:
            actual_cost = ctx.cost("tree_small", tr_s.size)
            if spent + actual_cost > budget_usd:
                n_fit = ctx.affordable("tree_small", budget_usd - spent)
                tr_s, tc_s = tr_s[:n_fit], tc_s[:n_fit]
                actual_cost = ctx.cost("tree_small", tr_s.size)
            if tr_s.size:
                placements.append(Placement("tree_small", tr_s, tc_s))
                spent += actual_cost
                used[tr_s, tc_s] = True
                crown_sml_px = max(int(round(CROWN_SML_M / ctx.res_m)), 1)
                _stamp_exclusion(covered, tr_s, tc_s, crown_sml_px, ctx.shape)

    # ── 3. Green roofs (10% budget, hot priority-tract buildings) ────────
    # Green roofs lower building surface temps and add cobenefit_greened_pct.
    # Target hot buildings in high-vulnerability tracts.
    roof_budget = min(budget_usd * FRAC_GREEN_ROOF, budget_usd - spent)
    if roof_budget > ctx.unit_cost("green_roof"):
        cand_roof  = ctx.placeable("green_roof") & ~used
        n_roof_max = _safe_affordable(ctx, "green_roof", roof_budget, int(cand_roof.sum()))

        rr, rc = _top_pixels(roof_score, cand_roof, n_roof_max)
        if rr.size:
            actual_cost = ctx.cost("green_roof", rr.size)
            if spent + actual_cost > budget_usd:
                n_fit = ctx.affordable("green_roof", budget_usd - spent)
                rr, rc = rr[:n_fit], rc[:n_fit]
                actual_cost = ctx.cost("green_roof", rr.size)
            if rr.size:
                placements.append(Placement("green_roof", rr, rc))
                spent += actual_cost
                used[rr, rc] = True

    # ── 4. Shade canopies (remaining budget, hottest open ground) ─────────
    # Fill all remaining buildable open ground not already shaded.
    remaining = budget_usd - spent
    if remaining > ctx.unit_cost("shade_canopy"):
        open_ground  = ctx.buildable & ~covered & ~used
        n_canopy_max = _safe_affordable(ctx, "shade_canopy", remaining, int(open_ground.sum()))

        cr, cc = _top_pixels(canopy_score, open_ground, n_canopy_max)
        if cr.size:
            actual_cost = ctx.cost("shade_canopy", cr.size)
            if spent + actual_cost > budget_usd:
                n_fit = ctx.affordable("shade_canopy", budget_usd - spent)
                cr, cc = cr[:n_fit], cc[:n_fit]
                actual_cost = ctx.cost("shade_canopy", cr.size)
            if cr.size:
                placements.append(Placement("shade_canopy", cr, cc))
                spent += actual_cost
                used[cr, cc] = True

    # ── 5. Any remaining budget → more medium trees ───────────────────────
    remaining = budget_usd - spent
    if remaining > ctx.unit_cost("tree_medium"):
        cand_extra = ctx.plantable & ~used
        n_extra    = _safe_affordable(ctx, "tree_medium", remaining, int(cand_extra.sum()))
        if n_extra > 0:
            # No spacing constraint — fill remaining budget
            er, ec = _top_pixels(tree_score, cand_extra, n_extra)
            if er.size:
                actual_cost = ctx.cost("tree_medium", er.size)
                if spent + actual_cost > budget_usd:
                    n_fit = ctx.affordable("tree_medium", budget_usd - spent)
                    er, ec = er[:n_fit], ec[:n_fit]
                    actual_cost = ctx.cost("tree_medium", er.size)
                if er.size:
                    placements.append(Placement("tree_medium", er, ec))
                    spent += actual_cost
                    used[er, ec] = True

    # ── 6. Last remainder → grass conversion (cobenefit boost) ───────────
    remaining = budget_usd - spent
    if remaining > ctx.unit_cost("grass_conversion"):
        cand_grass  = ctx.placeable("grass_conversion") & ~ctx.roadbed & ~used
        n_grass_max = _safe_affordable(ctx, "grass_conversion", remaining, int(cand_grass.sum()))
        if n_grass_max > 0:
            # Score grass by heat + population (simple linear)
            mask = ctx.walkable
            heat_n = _norm(ctx.heat_ta3pm, mask)
            pop_n  = _norm(ctx.population,  mask)
            grass_score = np.where(
                mask,
                0.60 * heat_n + 0.25 * pop_n
                    + 0.15 * _norm(ctx.heat_uhii, mask)
                    + np.where(ctx.priority, PRIORITY_BOOST, 0.0),
                -np.inf,
            )
            gr, gc = _top_pixels(grass_score, cand_grass, n_grass_max)
            if gr.size:
                actual_cost = ctx.cost("grass_conversion", gr.size)
                if spent + actual_cost > budget_usd:
                    n_fit = ctx.affordable("grass_conversion", budget_usd - spent)
                    gr, gc = gr[:n_fit], gc[:n_fit]
                    actual_cost = ctx.cost("grass_conversion", gr.size)
                if gr.size:
                    placements.append(Placement("grass_conversion", gr, gc))

    return placements