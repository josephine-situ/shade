from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-corridor heat-equity v1"
DESCRIPTION = (
    "Maximise population-weighted UTCI relief targeting cell (1,0,1,1). "
    "Uses heat×population geometric-mean synergy scoring to target pixels "
    "that are BOTH hot AND populated. Phase 1: Medium trees at 4m spacing "
    "(65% budget) on hottest populated corridors. Phase 2: Shade canopies "
    "on remaining open hot ground (25% budget). Phase 3: Grass conversion "
    "on hot paved pixels (10% budget) to maintain cobenefit_greened_pct. "
    "Strong vulnerability boost (0.20) preserves equity_ratio >= 1. "
    "Priority surface: 0.40 heat×pop synergy + 0.25 heat_ta3pm + "
    "0.15 population + 0.10 heat_hours + 0.10 vulnerability. "
    "Reflective surfaces avoided entirely (albedo trap)."
)

MEDIUM_BUDGET_FRAC = 0.65
CANOPY_BUDGET_FRAC = 0.25
GRASS_BUDGET_FRAC  = 0.10

MEDIUM_SPACING_M = 4.0   # tight spacing for corridor coverage
CROWN_MED_M      = 3.5

PRIORITY_BOOST = 0.20    # strong boost for top-quartile vulnerability tracts


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1] over masked pixels; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def _synergy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Heat×population geometric mean synergy scoring.
    Targets pixels BOTH hot AND populated for maximum person-degC relief.
    Includes vulnerability weighting for equity and priority-tract boost.
    """
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)

    # Geometric mean synergy: high only if BOTH heat and population are high
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.40 * synergy
        + 0.25 * heat_n
        + 0.15 * pop_n
        + 0.10 * hours_n
        + 0.10 * vuln_n
    )
    # Strong priority-tract boost to drive equity_ratio >= 1
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _grass_score(ctx: PlanningContext) -> np.ndarray:
    """Score for grass conversion: target hot paved non-roadbed areas."""
    mask = ctx.walkable
    heat_n  = _norm(ctx.heat_ta3pm,   mask)
    pop_n   = _norm(ctx.population,   mask)
    uhii_n  = _norm(ctx.heat_uhii,    mask)
    score = (
          0.50 * heat_n
        + 0.30 * pop_n
        + 0.20 * uhii_n
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


def _safe_place(
    ctx: PlanningContext,
    action: str,
    rows: np.ndarray,
    cols: np.ndarray,
    spent: float,
    budget_usd: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Trim placement to fit within remaining budget."""
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
    score    = _synergy_score(ctx)
    grass_sc = _grass_score(ctx)
    placements: list[Placement] = []
    spent = 0.0
    used  = np.zeros(ctx.shape, dtype=bool)

    # ── Phase 1: Medium street trees (65% budget, 4m spacing) ─────────────
    medium_budget  = budget_usd * MEDIUM_BUDGET_FRAC
    n_medium_max   = ctx.affordable("tree_medium", medium_budget)
    cand_medium    = ctx.plantable & ~used
    n_medium       = min(n_medium_max, int(cand_medium.sum()))

    spacing_medium = max(int(round(MEDIUM_SPACING_M / ctx.res_m)), 1)
    mr, mc = _greedy_spaced(score, cand_medium, spacing_medium, n_medium)

    if mr.size:
        mr, mc, spent = _safe_place(ctx, "tree_medium", mr, mc, spent, budget_usd)
        if mr.size:
            placements.append(Placement("tree_medium", mr, mc))
            used[mr, mc] = True

    # ── Phase 2: Shade canopies (25% budget, hottest open pedestrian ground)
    canopy_budget = min(budget_usd * CANOPY_BUDGET_FRAC, budget_usd - spent)
    cr = cc = np.array([], dtype=int)
    if canopy_budget >= ctx.unit_cost("shade_canopy"):
        covered = np.zeros(ctx.shape, dtype=bool)
        covered[ctx.cdsm > 0.0] = True   # existing canopy overhead

        if mr.size:
            crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
            _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)

        open_ground  = ctx.buildable & ~covered & ~used
        n_canopy_max = ctx.affordable("shade_canopy", canopy_budget)
        n_canopy     = min(n_canopy_max, int(open_ground.sum()))

        cr, cc = _top_pixels(score, open_ground, n_canopy)
        if cr.size:
            cr, cc, spent = _safe_place(ctx, "shade_canopy", cr, cc, spent, budget_usd)
            if cr.size:
                placements.append(Placement("shade_canopy", cr, cc))
                used[cr, cc] = True

    # ── Phase 3: Grass conversion (10% budget, cobenefit_greened_pct boost)
    grass_budget = min(budget_usd * GRASS_BUDGET_FRAC, budget_usd - spent)
    if grass_budget >= ctx.unit_cost("grass_conversion"):
        cand_grass  = ctx.placeable("grass_conversion") & ~ctx.roadbed & ~used
        n_grass_max = ctx.affordable("grass_conversion", grass_budget)
        n_grass     = min(n_grass_max, int(cand_grass.sum()))

        gr, gc = _top_pixels(grass_sc, cand_grass, n_grass)
        if gr.size:
            gr, gc, spent = _safe_place(ctx, "grass_conversion", gr, gc, spent, budget_usd)
            if gr.size:
                placements.append(Placement("grass_conversion", gr, gc))
                used[gr, gc] = True

    # ── Phase 4: Remaining budget → more shade canopies ──────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("shade_canopy"):
        covered2 = np.zeros(ctx.shape, dtype=bool)
        covered2[ctx.cdsm > 0.0] = True
        if mr.size:
            crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
            _stamp_crown(covered2, mr, mc, crown_med_px, ctx.shape)

        open_ground2 = ctx.buildable & ~covered2 & ~used
        n_extra      = ctx.affordable("shade_canopy", remaining)
        n_extra      = min(n_extra, int(open_ground2.sum()))

        if n_extra > 0:
            er, ec = _top_pixels(score, open_ground2, n_extra)
            if er.size:
                er, ec, spent = _safe_place(ctx, "shade_canopy", er, ec, spent, budget_usd)
                if er.size:
                    placements.append(Placement("shade_canopy", er, ec))
                    used[er, ec] = True

    # ── Phase 5: Any last remainder → more grass conversion ───────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("grass_conversion"):
        cand_grass2  = ctx.placeable("grass_conversion") & ~ctx.roadbed & ~used
        n_grass2_max = ctx.affordable("grass_conversion", remaining)
        n_grass2     = min(n_grass2_max, int(cand_grass2.sum()))
        if n_grass2 > 0:
            gr2, gc2 = _top_pixels(grass_sc, cand_grass2, n_grass2)
            if gr2.size:
                gr2, gc2, spent = _safe_place(
                    ctx, "grass_conversion", gr2, gc2, spent, budget_usd
                )
                if gr2.size:
                    placements.append(Placement("grass_conversion", gr2, gc2))

    return placements