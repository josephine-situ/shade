from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-equity-greening corridors v1"
DESCRIPTION = (
    "Targets cell (1,0,0,1): high equity_ratio + high cobenefit_greened_pct. "
    "Uses heat×population synergy scoring with strong priority-tract boost. "
    "Phase 1: Medium trees at 3m spacing (50% budget) on hot+populated corridors. "
    "Phase 2: Small trees at 2m spacing (20% budget) gap-filling. "
    "Phase 3: Grass conversion on hot paved pixels (15% budget) to maintain "
    "cobenefit_greened_pct >= threshold. "
    "Phase 4: Shade canopies on remaining open hot ground (~15% budget). "
    "Synergy surface: 0.45 sqrt(heat×pop) + 0.20 heat + 0.12 pop + "
    "0.13 vulnerability + 0.05 heat_hours + 0.05 UHII + 0.20 priority boost. "
    "Reflective surfaces avoided entirely (albedo trap)."
)

FRAC_MED    = 0.50
FRAC_SML    = 0.20
FRAC_GRASS  = 0.15
FRAC_CANOPY = 0.15

SPACING_MED_M = 3.0
SPACING_SML_M = 2.0
CROWN_MED_M   = 3.5
CROWN_SML_M   = 2.0

PRIORITY_BOOST = 0.20


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1] over mask; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def _synergy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Synergy-based priority surface for tree and canopy placement.
    Geometric mean of heat × population rewards pixels that are simultaneously
    hot AND populated — maximising person-degC relief per intervention.
    Strong vulnerability weighting drives equity_ratio >= 1.
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
        + 0.20 * heat_n
        + 0.12 * pop_n
        + 0.13 * vuln_n
        + 0.05 * hours_n
        + 0.05 * uhii_n
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _grass_score(ctx: PlanningContext) -> np.ndarray:
    """
    Score for grass conversion: target hot paved areas with high population
    and vulnerability. Grass contributes to cobenefit_greened_pct.
    """
    mask = ctx.walkable
    heat_n = _norm(ctx.heat_ta3pm,    mask)
    pop_n  = _norm(ctx.population,    mask)
    vuln_n = _norm(ctx.vulnerability, mask)
    uhii_n = _norm(ctx.heat_uhii,     mask)

    score = (
          0.40 * heat_n
        + 0.25 * pop_n
        + 0.20 * vuln_n
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
    Greedy spaced placement: pick highest-scoring candidate, suppress
    spacing_px neighbourhood, repeat until limit reached or exhausted.
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
    score     = _synergy_score(ctx)
    grass_sc  = _grass_score(ctx)
    placements: list[Placement] = []
    spent = 0.0
    used  = np.zeros(ctx.shape, dtype=bool)

    # Track canopy coverage to avoid placing canopies under existing trees
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # pre-existing canopy

    # ── Phase 1: Medium street trees (50% budget, 3m spacing) ─────────────
    # Synergy surface targets hot+populated corridors for max UTCI relief.
    med_budget    = budget_usd * FRAC_MED
    n_med_max     = ctx.affordable("tree_medium", med_budget)
    cand_med      = ctx.plantable & ~used
    n_med         = min(n_med_max, int(cand_med.sum()))
    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)

    mr, mc = _greedy_spaced(score, cand_med, spacing_med_px, n_med)
    if mr.size:
        mr, mc, spent = _safe_place(ctx, "tree_medium", mr, mc, spent, budget_usd)
        if mr.size:
            placements.append(Placement("tree_medium", mr, mc))
            used[mr, mc] = True
            crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
            _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)

    # ── Phase 2: Small trees (20% budget, 2m spacing, gap-filling) ────────
    # Fill gaps left by medium trees with smaller/cheaper trees.
    sml_budget_target = budget_usd * FRAC_SML
    sml_budget = min(sml_budget_target, budget_usd - spent)
    tr_s = tc_s = np.array([], dtype=int)

    if sml_budget >= ctx.unit_cost("tree_small"):
        n_sml_max     = ctx.affordable("tree_small", sml_budget)
        # Plant small trees where not already covered by medium tree crowns
        cand_sml      = ctx.plantable & ~used & ~covered
        n_sml         = min(n_sml_max, int(cand_sml.sum()))
        spacing_sml_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)

        tr_s, tc_s = _greedy_spaced(score, cand_sml, spacing_sml_px, n_sml)
        if tr_s.size:
            tr_s, tc_s, spent = _safe_place(ctx, "tree_small", tr_s, tc_s, spent, budget_usd)
            if tr_s.size:
                placements.append(Placement("tree_small", tr_s, tc_s))
                used[tr_s, tc_s] = True
                crown_sml_px = max(int(round(CROWN_SML_M / ctx.res_m)), 1)
                _stamp_crown(covered, tr_s, tc_s, crown_sml_px, ctx.shape)

    # ── Phase 3: Grass conversion (15% budget, hot paved pixels) ──────────
    # Grass conversion contributes to cobenefit_greened_pct (target >= 0.1976).
    # Also provides modest cooling through evapotranspiration.
    grass_budget_target = budget_usd * FRAC_GRASS
    grass_budget = min(grass_budget_target, budget_usd - spent)
    gr = gc = np.array([], dtype=int)

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

    # ── Phase 4: Shade canopies (~15% budget, hottest open ground) ─────────
    # Canopies provide strong UTCI relief without counting as green area.
    canopy_budget = min(budget_usd * FRAC_CANOPY, budget_usd - spent)
    if canopy_budget >= ctx.unit_cost("shade_canopy"):
        open_ground  = ctx.buildable & ~covered & ~used
        n_canopy_max = ctx.affordable("shade_canopy", canopy_budget)
        n_canopy     = min(n_canopy_max, int(open_ground.sum()))

        cr, cc = _top_pixels(score, open_ground, n_canopy)
        if cr.size:
            cr, cc, spent = _safe_place(ctx, "shade_canopy", cr, cc, spent, budget_usd)
            if cr.size:
                placements.append(Placement("shade_canopy", cr, cc))
                used[cr, cc] = True

    # ── Phase 5: Remaining budget → more medium trees ─────────────────────
    # Use any leftover for more medium trees (best UTCI per dollar for trees).
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("tree_medium"):
        cand_extra   = ctx.plantable & ~used
        n_extra_max  = ctx.affordable("tree_medium", remaining)
        n_extra      = min(n_extra_max, int(cand_extra.sum()))
        spacing_med_px2 = max(int(round(SPACING_MED_M / ctx.res_m)), 1)

        if n_extra > 0:
            er, ec = _greedy_spaced(score, cand_extra, spacing_med_px2, n_extra)
            if er.size:
                er, ec, spent = _safe_place(ctx, "tree_medium", er, ec, spent, budget_usd)
                if er.size:
                    placements.append(Placement("tree_medium", er, ec))
                    used[er, ec] = True

    # ── Phase 6: Final remainder → more grass conversion ──────────────────
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
                    used[gr2, gc2] = True

    return placements