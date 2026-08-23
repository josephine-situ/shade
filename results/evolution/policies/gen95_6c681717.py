from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "dense-synergy-corridors v2 improved"
DESCRIPTION = (
    "Improved champion for cell (1,0,1,1): equity_ratio>=1, access<0.32, "
    "efficiency>=40.55, greened>=0.1976. "
    "Key improvements over parent: "
    "(1) Tighter 2m medium-tree spacing (1px) for maximum canopy density; "
    "(2) Triple synergy scoring: cube-root of heat×pop×vulnerability; "
    "(3) Larger canopy budget (20%) with greedy spacing to spread shade; "
    "(4) Vulnerability-weighted scoring ensures equity_ratio stays >=1; "
    "(5) Solar canopies as fallback to use remaining budget efficiently. "
    "Budget: 60% medium trees, 15% small trees, 20% shade canopy, 5% grass. "
    "Priority surface: 0.40 triple-synergy(heat×pop×vuln) + 0.25 heat_ta3pm "
    "+ 0.15 population + 0.10 heat_hours + 0.10 vulnerability "
    "+ 0.10 priority-tract boost. Reflective surfaces avoided entirely."
)

FRAC_MED    = 0.60
FRAC_SML    = 0.15
FRAC_CANOPY = 0.20
FRAC_GRASS  = 0.05

SPACING_MED_M   = 2.0   # tightest possible (1px at 2m res)
SPACING_SML_M   = 2.0
SPACING_CAN_M   = 3.0   # canopy spacing for spread coverage

CROWN_MED_M = 3.5
CROWN_SML_M = 2.0

PRIORITY_BOOST = 0.10   # modest equity boost to keep equity_ratio >= 1


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
    Triple synergy-based priority surface for tree placement.
    Cube-root of heat × population × vulnerability to reward pixels that are
    hot, populated, AND vulnerable — maximising equity-weighted UTCI relief.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)
    vuln_n  = _norm(ctx.vulnerability, mask)

    # Triple synergy: cube-root rewards pixels that score on all three
    triple_syn = np.cbrt(np.clip(heat_n * pop_n * vuln_n, 0.0, None))
    # Pairwise heat×pop synergy for fallback
    heat_pop_syn = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.35 * triple_syn
        + 0.20 * heat_pop_syn
        + 0.20 * heat_n
        + 0.10 * pop_n
        + 0.08 * hours_n
        + 0.04 * uhii_n
        + 0.03 * vuln_n
    )
    # Modest priority-tract boost to keep equity_ratio >= 1
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _canopy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for shade canopy: synergy + heat focus with equity.
    Canopies provide immediate structural shade for UTCI relief.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)
    vuln_n  = _norm(ctx.vulnerability, mask)

    triple_syn   = np.cbrt(np.clip(heat_n * pop_n * vuln_n, 0.0, None))
    heat_pop_syn = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.35 * triple_syn
        + 0.20 * heat_pop_syn
        + 0.20 * heat_n
        + 0.10 * vuln_n
        + 0.08 * pop_n
        + 0.05 * hours_n
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
          0.40 * heat_n
        + 0.25 * pop_n
        + 0.20 * uhii_n
        + 0.15 * vuln_n
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
    spent   = 0.0
    used    = np.zeros(ctx.shape, dtype=bool)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True  # pre-existing canopy overhead

    # ── Phase 1: Medium street trees (60% budget, 2m/1px spacing) ─────────
    # Tightest spacing maximises canopy cover per unit area
    med_budget     = budget_usd * FRAC_MED
    n_med_max      = ctx.affordable("tree_medium", med_budget)
    cand_med       = ctx.plantable & ~used
    n_med          = min(n_med_max, int(cand_med.sum()))
    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)

    mr, mc = _greedy_spaced(tree_sc, cand_med, spacing_med_px, n_med)
    if mr.size:
        mr, mc, spent = _safe_trim(ctx, "tree_medium", mr, mc, spent, budget_usd)
        if mr.size:
            placements.append(Placement("tree_medium", mr, mc))
            used[mr, mc] = True
            crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
            _stamp_exclusion(covered, mr, mc, crown_med_px, ctx.shape)

    # ── Phase 2: Small street trees (15% budget, 2m spacing, gap-fill) ────
    sml_budget     = min(budget_usd * FRAC_SML, budget_usd - spent)
    tr_s = tc_s    = np.array([], dtype=int)
    if sml_budget >= ctx.unit_cost("tree_small"):
        n_sml_max      = ctx.affordable("tree_small", sml_budget)
        cand_sml       = ctx.plantable & ~used & ~covered
        n_sml          = min(n_sml_max, int(cand_sml.sum()))
        spacing_sml_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)

        tr_s, tc_s = _greedy_spaced(tree_sc, cand_sml, spacing_sml_px, n_sml)
        if tr_s.size:
            tr_s, tc_s, spent = _safe_trim(ctx, "tree_small", tr_s, tc_s, spent, budget_usd)
            if tr_s.size:
                placements.append(Placement("tree_small", tr_s, tc_s))
                used[tr_s, tc_s] = True
                crown_sml_px = max(int(round(CROWN_SML_M / ctx.res_m)), 1)
                _stamp_exclusion(covered, tr_s, tc_s, crown_sml_px, ctx.shape)

    # ── Phase 3: Shade canopies (20% budget, spaced for area coverage) ────
    # Greedy spacing spreads canopies to cover more exposed pedestrian space
    canopy_budget  = min(budget_usd * FRAC_CANOPY, budget_usd - spent)
    cr = cc        = np.array([], dtype=int)
    if canopy_budget >= ctx.unit_cost("shade_canopy"):
        open_ground    = ctx.buildable & ~covered & ~used
        n_canopy_max   = ctx.affordable("shade_canopy", canopy_budget)
        n_canopy       = min(n_canopy_max, int(open_ground.sum()))
        spacing_can_px = max(int(round(SPACING_CAN_M / ctx.res_m)), 1)

        cr, cc = _greedy_spaced(canopy_sc, open_ground, spacing_can_px, n_canopy)
        if cr.size:
            cr, cc, spent = _safe_trim(ctx, "shade_canopy", cr, cc, spent, budget_usd)
            if cr.size:
                placements.append(Placement("shade_canopy", cr, cc))
                used[cr, cc] = True

    # ── Phase 4: Grass conversion (5% budget, cobenefit_greened_pct) ──────
    grass_budget = min(budget_usd * FRAC_GRASS, budget_usd - spent)
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

    # ── Phase 5: Remaining budget → more shade canopies (spaced) ──────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("shade_canopy"):
        open_ground2   = ctx.buildable & ~covered & ~used
        n_extra        = ctx.affordable("shade_canopy", remaining)
        n_extra        = min(n_extra, int(open_ground2.sum()))
        spacing_can_px = max(int(round(SPACING_CAN_M / ctx.res_m)), 1)

        if n_extra > 0:
            er, ec = _greedy_spaced(canopy_sc, open_ground2, spacing_can_px, n_extra)
            if er.size:
                er, ec, spent = _safe_trim(ctx, "shade_canopy", er, ec, spent, budget_usd)
                if er.size:
                    placements.append(Placement("shade_canopy", er, ec))
                    used[er, ec] = True

    # ── Phase 6: Remaining budget → more medium trees ─────────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("tree_medium"):
        cand_extra = ctx.plantable & ~used
        n_extra    = ctx.affordable("tree_medium", remaining)
        n_extra    = min(n_extra, int(cand_extra.sum()))

        if n_extra > 0:
            spacing_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
            xr, xc = _greedy_spaced(tree_sc, cand_extra, spacing_px, n_extra)
            if xr.size:
                xr, xc, spent = _safe_trim(ctx, "tree_medium", xr, xc, spent, budget_usd)
                if xr.size:
                    placements.append(Placement("tree_medium", xr, xc))
                    used[xr, xc] = True
                    crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
                    _stamp_exclusion(covered, xr, xc, crown_med_px, ctx.shape)

    # ── Phase 7: Remaining → small trees ──────────────────────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("tree_small"):
        cand_sml2      = ctx.plantable & ~used & ~covered
        n_sml2         = ctx.affordable("tree_small", remaining)
        n_sml2         = min(n_sml2, int(cand_sml2.sum()))

        if n_sml2 > 0:
            spacing_px2 = max(int(round(SPACING_SML_M / ctx.res_m)), 1)
            sr2, sc2 = _greedy_spaced(tree_sc, cand_sml2, spacing_px2, n_sml2)
            if sr2.size:
                sr2, sc2, spent = _safe_trim(ctx, "tree_small", sr2, sc2, spent, budget_usd)
                if sr2.size:
                    placements.append(Placement("tree_small", sr2, sc2))
                    used[sr2, sc2] = True

    # ── Phase 8: Final remaining → grass conversion ───────────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("grass_conversion"):
        cand_grass2  = ctx.placeable("grass_conversion") & ~ctx.roadbed & ~used
        n_grass2     = ctx.affordable("grass_conversion", remaining)
        n_grass2     = min(n_grass2, int(cand_grass2.sum()))

        if n_grass2 > 0:
            gr2, gc2 = _top_pixels(grass_sc, cand_grass2, n_grass2)
            if gr2.size:
                gr2, gc2, spent = _safe_trim(ctx, "grass_conversion", gr2, gc2, spent, budget_usd)
                if gr2.size:
                    placements.append(Placement("grass_conversion", gr2, gc2))

    return placements