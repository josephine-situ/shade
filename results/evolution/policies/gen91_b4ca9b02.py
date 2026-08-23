from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-equity-greened corridors v3"
DESCRIPTION = (
    "Champion attempt for cell (1,0,0,1): equity>=1, access<0.32, "
    "efficiency<40.55, cobenefit_greened>=0.1976. "
    "Uses 4m medium-tree spacing (wider coverage vs 3m parent) with "
    "strong priority-tract equity boost (0.35). "
    "Budget split: 65% medium trees at 4m spacing, 15% shade canopies, "
    "12% grass conversion (ensures cobenefit_greened>=0.1976), 8% small trees. "
    "Priority surface: 0.50 heat×pop synergy + 0.20 heat_ta3pm + "
    "0.12 population + 0.10 vulnerability + 0.08 heat_hours. "
    "Remaining budget cascades: more medium trees → canopies → grass. "
    "No reflective surfaces. Strong equity focus via priority boost."
)

# Budget fractions
FRAC_MED    = 0.65
FRAC_CANOPY = 0.15
FRAC_GRASS  = 0.12
FRAC_SML    = 0.08

# Spacing
SPACING_MED_M = 4.0
SPACING_SML_M = 2.5

# Crown radii for exclusion tracking
CROWN_MED_M = 3.5
CROWN_SML_M = 2.0

# Strong equity boost to ensure equity_ratio >= 1
PRIORITY_BOOST = 0.35


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1] over masked pixels; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def _tree_score(ctx: PlanningContext) -> np.ndarray:
    """
    Synergy-based priority surface for tree placement.
    Geometric mean of heat×population rewards pixels BOTH hot AND populated.
    Strong vulnerability boost ensures equity_ratio >= 1.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)

    # Geometric mean synergy: high only if BOTH heat AND population are high
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.50 * synergy
        + 0.20 * heat_n
        + 0.12 * pop_n
        + 0.10 * vuln_n
        + 0.08 * hours_n
    )
    # Strong priority-tract boost to drive equity_ratio >= 1
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _canopy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for shade canopy: emphasise heat + vulnerability.
    Canopies provide immediate UTCI relief.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)
    vuln_n  = _norm(ctx.vulnerability, mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.45 * synergy
        + 0.22 * heat_n
        + 0.18 * vuln_n
        + 0.10 * pop_n
        + 0.03 * hours_n
        + 0.02 * uhii_n
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _grass_score(ctx: PlanningContext) -> np.ndarray:
    """Score for grass conversion: target hot paved areas with high population."""
    mask = ctx.walkable

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)
    vuln_n  = _norm(ctx.vulnerability, mask)

    score = (
          0.45 * heat_n
        + 0.25 * pop_n
        + 0.20 * uhii_n
        + 0.10 * vuln_n
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
    Greedy selection: pick highest-scoring candidate, suppress spacing_px
    neighbourhood, repeat until `limit` reached or candidates exhausted.
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
        r0 = max(0, r - radius_px);  r1 = min(H, r + radius_px + 1)
        c0 = max(0, c - radius_px);  c1 = min(W, c + radius_px + 1)
        covered[r0:r1, c0:c1] = True


def _safe_trim(
    ctx: PlanningContext,
    action: str,
    rows: np.ndarray,
    cols: np.ndarray,
    spent: float,
    budget_usd: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Trim placement to fit within remaining budget; return updated spent."""
    if rows.size == 0:
        return rows, cols, spent
    remaining = budget_usd - spent
    affordable_n = ctx.affordable(action, remaining)
    if affordable_n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), spent
    rows = rows[:affordable_n]
    cols = cols[:affordable_n]
    spent += ctx.cost(action, rows.size)
    return rows, cols, spent


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    tree_sc   = _tree_score(ctx)
    canopy_sc = _canopy_score(ctx)
    grass_sc  = _grass_score(ctx)

    placements: list[Placement] = []
    spent = 0.0
    used    = np.zeros(ctx.shape, dtype=bool)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # pre-existing canopy overhead

    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    spacing_sml_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)
    crown_med_px   = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
    crown_sml_px   = max(int(round(CROWN_SML_M / ctx.res_m)), 1)

    # ── Phase 1: Medium street trees (65% budget, 4m spacing) ─────────────
    med_budget = budget_usd * FRAC_MED
    n_med_max  = ctx.affordable("tree_medium", med_budget)
    cand_med   = ctx.plantable & ~used
    n_med      = min(n_med_max, int(cand_med.sum()))

    mr, mc = _greedy_spaced(tree_sc, cand_med, spacing_med_px, n_med)
    if mr.size:
        mr, mc, spent = _safe_trim(ctx, "tree_medium", mr, mc, spent, budget_usd)
        if mr.size:
            placements.append(Placement("tree_medium", mr, mc))
            used[mr, mc] = True
            _stamp_exclusion(covered, mr, mc, crown_med_px, ctx.shape)

    # ── Phase 2: Grass conversion (12% budget, cobenefit_greened_pct boost) ─
    # Do grass BEFORE small trees and canopies to ensure cobenefit >= 0.1976
    grass_budget = min(budget_usd * FRAC_GRASS, budget_usd - spent)
    gr = gc = np.array([], dtype=int)
    if grass_budget >= ctx.unit_cost("grass_conversion"):
        cand_grass  = ctx.placeable("grass_conversion") & ~ctx.roadbed & ~used
        n_grass_max = ctx.affordable("grass_conversion", grass_budget)
        n_grass     = min(n_grass_max, int(cand_grass.sum()))

        gr, gc = _top_pixels(grass_sc, cand_grass, n_grass)
        if gr.size:
            gr, gc, spent = _safe_trim(ctx, "grass_conversion", gr, gc, spent, budget_usd)
            if gr.size:
                placements.append(Placement("grass_conversion", gr, gc))
                used[gr, gc] = True

    # ── Phase 3: Shade canopies (15% budget, hottest open buildable ground) ─
    canopy_budget = min(budget_usd * FRAC_CANOPY, budget_usd - spent)
    cr = cc = np.array([], dtype=int)
    if canopy_budget >= ctx.unit_cost("shade_canopy"):
        open_ground  = ctx.buildable & ~covered & ~used
        n_canopy_max = ctx.affordable("shade_canopy", canopy_budget)
        n_canopy     = min(n_canopy_max, int(open_ground.sum()))

        cr, cc = _top_pixels(canopy_sc, open_ground, n_canopy)
        if cr.size:
            cr, cc, spent = _safe_trim(ctx, "shade_canopy", cr, cc, spent, budget_usd)
            if cr.size:
                placements.append(Placement("shade_canopy", cr, cc))
                used[cr, cc] = True

    # ── Phase 4: Small street trees (8% budget, 2.5m spacing, gap-fill) ───
    sml_budget = min(budget_usd * FRAC_SML, budget_usd - spent)
    sr = sc = np.array([], dtype=int)
    if sml_budget >= ctx.unit_cost("tree_small"):
        cand_sml  = ctx.plantable & ~used & ~covered
        n_sml_max = ctx.affordable("tree_small", sml_budget)
        n_sml     = min(n_sml_max, int(cand_sml.sum()))

        sr, sc = _greedy_spaced(tree_sc, cand_sml, spacing_sml_px, n_sml)
        if sr.size:
            sr, sc, spent = _safe_trim(ctx, "tree_small", sr, sc, spent, budget_usd)
            if sr.size:
                placements.append(Placement("tree_small", sr, sc))
                used[sr, sc] = True
                _stamp_exclusion(covered, sr, sc, crown_sml_px, ctx.shape)

    # ── Phase 5: Remaining budget → more medium trees ─────────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("tree_medium"):
        cand_extra = ctx.plantable & ~used
        n_extra    = min(ctx.affordable("tree_medium", remaining), int(cand_extra.sum()))
        if n_extra > 0:
            xr, xc = _greedy_spaced(tree_sc, cand_extra, spacing_med_px, n_extra)
            if xr.size:
                xr, xc, spent = _safe_trim(ctx, "tree_medium", xr, xc, spent, budget_usd)
                if xr.size:
                    placements.append(Placement("tree_medium", xr, xc))
                    used[xr, xc] = True
                    _stamp_exclusion(covered, xr, xc, crown_med_px, ctx.shape)

    # ── Phase 6: Remaining → more shade canopies ──────────────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("shade_canopy"):
        open_ground2 = ctx.buildable & ~covered & ~used
        n_extra2     = min(ctx.affordable("shade_canopy", remaining), int(open_ground2.sum()))
        if n_extra2 > 0:
            er, ec = _top_pixels(canopy_sc, open_ground2, n_extra2)
            if er.size:
                er, ec, spent = _safe_trim(ctx, "shade_canopy", er, ec, spent, budget_usd)
                if er.size:
                    placements.append(Placement("shade_canopy", er, ec))
                    used[er, ec] = True

    # ── Phase 7: Remaining → more grass conversion ────────────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("grass_conversion"):
        cand_grass2  = ctx.placeable("grass_conversion") & ~ctx.roadbed & ~used
        n_grass2_max = ctx.affordable("grass_conversion", remaining)
        n_grass2     = min(n_grass2_max, int(cand_grass2.sum()))
        if n_grass2 > 0:
            gr2, gc2 = _top_pixels(grass_sc, cand_grass2, n_grass2)
            if gr2.size:
                gr2, gc2, spent = _safe_trim(ctx, "grass_conversion", gr2, gc2, spent, budget_usd)
                if gr2.size:
                    placements.append(Placement("grass_conversion", gr2, gc2))
                    used[gr2, gc2] = True

    # ── Phase 8: Final remainder → small trees ────────────────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("tree_small"):
        cand_sml2 = ctx.plantable & ~used
        n_sml2    = min(ctx.affordable("tree_small", remaining), int(cand_sml2.sum()))
        if n_sml2 > 0:
            sr2, sc2 = _greedy_spaced(tree_sc, cand_sml2, spacing_sml_px, n_sml2)
            if sr2.size:
                sr2, sc2, spent = _safe_trim(ctx, "tree_small", sr2, sc2, spent, budget_usd)
                if sr2.size:
                    placements.append(Placement("tree_small", sr2, sc2))

    return placements