from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-triple-green canopy v1"
DESCRIPTION = (
    "Maximise population-weighted UTCI relief while boosting cobenefit_greened_pct "
    "to enter the (0,1,1,1) MAP-Elites cell. Strategy: "
    "(1) Medium trees at 4m spacing consume 55% of budget on hottest, "
    "most-populated pedestrian corridors using heat×population synergy score; "
    "(2) Small trees at 3m spacing fill remaining gaps with 15% of budget; "
    "(3) Shade canopies cover hottest open pedestrian ground with 20% budget; "
    "(4) Grass conversion (depaving) on remaining paved non-roadbed pixels "
    "with 10% budget to boost cobenefit_greened_pct above 0.1976 threshold. "
    "Priority surface uses heat×population geometric mean synergy (0.45 weight) "
    "plus heat_ta3pm (0.25), population (0.15), heat_hours (0.10), vulnerability "
    "(0.05). Priority-tract boost of 0.12 for equity. Reflective surfaces avoided."
)

MEDIUM_BUDGET_FRAC = 0.55
SMALL_BUDGET_FRAC  = 0.15
CANOPY_BUDGET_FRAC = 0.20
GRASS_BUDGET_FRAC  = 0.10

MEDIUM_SPACING_M = 4.0
SMALL_SPACING_M  = 3.0

CROWN_MED_M   = 3.5
CROWN_SMALL_M = 2.0

PRIORITY_BOOST = 0.12


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1] over masked region; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def _synergy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Heat×population geometric mean synergy score for tree/canopy placement.
    Focus on pixels that are BOTH hot AND populated for max person-degC relief.
    """
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,  mask)
    pop_n   = _norm(ctx.population,  mask)
    hours_n = _norm(ctx.heat_hours,  mask)
    vuln_n  = _norm(ctx.vulnerability, mask)

    # Geometric mean of heat and population for synergy
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.45 * synergy
        + 0.25 * heat_n
        + 0.15 * pop_n
        + 0.10 * hours_n
        + 0.05 * vuln_n
    )
    score += np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _grass_score(ctx: PlanningContext) -> np.ndarray:
    """
    Score for grass conversion: target hot paved areas with moderate population
    to maximise cobenefit_greened_pct while not wasting budget on cold areas.
    Use full grid (not just exposure) to find plantable paved areas.
    """
    mask = ctx.walkable  # broader mask for grass conversion candidates
    heat_n  = _norm(ctx.heat_ta3pm,  mask)
    pop_n   = _norm(ctx.population,  mask)
    uhii_n  = _norm(ctx.heat_uhii,   mask)

    score = (
          0.50 * heat_n
        + 0.30 * pop_n
        + 0.20 * uhii_n
    )
    score += np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy selection by score with spatial exclusion zone.
    Pick highest-scoring candidate, suppress spacing_px neighbourhood, repeat.
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


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score  = _synergy_score(ctx)
    placements: list[Placement] = []
    spent  = 0.0
    used   = np.zeros(ctx.shape, dtype=bool)

    # ── Phase 1: Medium street trees (55% budget, 4m spacing) ───────────
    # Medium trees give the strongest per-tree UTCI benefit.
    # 4m spacing packs dense shade into hottest, most-populated corridors.
    medium_budget  = budget_usd * MEDIUM_BUDGET_FRAC
    n_medium_max   = ctx.affordable("tree_medium", medium_budget)
    cand_medium    = ctx.plantable & ~used
    n_medium       = min(n_medium_max, int(cand_medium.sum()))

    spacing_medium = max(int(round(MEDIUM_SPACING_M / ctx.res_m)), 1)
    mr, mc = _greedy_spaced(score, cand_medium, spacing_medium, n_medium)

    if mr.size:
        placements.append(Placement("tree_medium", mr, mc))
        spent += ctx.cost("tree_medium", mr.size)
        used[mr, mc] = True

    # ── Phase 2: Small street trees (15% budget, 3m spacing, gap-fill) ──
    small_budget = min(budget_usd * SMALL_BUDGET_FRAC, budget_usd - spent)
    sr = sc = np.array([], dtype=int)
    if small_budget >= ctx.unit_cost("tree_small"):
        cand_small  = ctx.plantable & ~used
        n_small_max = ctx.affordable("tree_small", small_budget)
        n_small     = min(n_small_max, int(cand_small.sum()))

        spacing_small = max(int(round(SMALL_SPACING_M / ctx.res_m)), 1)
        sr, sc = _greedy_spaced(score, cand_small, spacing_small, n_small)

        if sr.size:
            placements.append(Placement("tree_small", sr, sc))
            spent += ctx.cost("tree_small", sr.size)
            used[sr, sc] = True

    # ── Phase 3: Shade canopies (20% budget, hottest open pedestrian ground)
    canopy_budget = min(budget_usd * CANOPY_BUDGET_FRAC, budget_usd - spent)
    cr = cc = np.array([], dtype=int)
    if canopy_budget >= ctx.unit_cost("shade_canopy"):
        # Exclude pixels already under new tree crowns
        covered = np.zeros(ctx.shape, dtype=bool)
        covered[ctx.cdsm > 0.0] = True  # existing canopy

        if mr.size:
            crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
            _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)
        if sr.size:
            crown_small_px = max(int(round(CROWN_SMALL_M / ctx.res_m)), 1)
            _stamp_crown(covered, sr, sc, crown_small_px, ctx.shape)

        open_ground  = ctx.buildable & ~covered & ~used
        n_canopy_max = ctx.affordable("shade_canopy", canopy_budget)
        n_canopy     = min(n_canopy_max, int(open_ground.sum()))

        cr, cc = _top_pixels(score, open_ground, n_canopy)
        if cr.size:
            actual_cost = ctx.cost("shade_canopy", cr.size)
            if spent + actual_cost <= budget_usd:
                placements.append(Placement("shade_canopy", cr, cc))
                spent += actual_cost
                used[cr, cc] = True
            else:
                affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
                if affordable_n > 0:
                    cr, cc = cr[:affordable_n], cc[:affordable_n]
                    placements.append(Placement("shade_canopy", cr, cc))
                    spent += ctx.cost("shade_canopy", cr.size)
                    used[cr, cc] = True

    # ── Phase 4: Grass conversion (10% budget, boost cobenefit_greened_pct)
    # Depave hot paved non-roadbed pixels to grass to push greening above 0.1976
    grass_budget = min(budget_usd * GRASS_BUDGET_FRAC, budget_usd - spent)
    if grass_budget >= ctx.unit_cost("grass_conversion"):
        # grass_conversion requires paved lc=[1], not roadbed
        cand_grass = ctx.placeable("grass_conversion") & ~used
        # Further restrict: not on roadbed (should be handled by placeable but be explicit)
        cand_grass = cand_grass & ~ctx.roadbed

        n_grass_max = ctx.affordable("grass_conversion", grass_budget)
        n_grass     = min(n_grass_max, int(cand_grass.sum()))

        gsc = _grass_score(ctx)
        gr, gc = _top_pixels(gsc, cand_grass, n_grass)

        if gr.size:
            actual_cost = ctx.cost("grass_conversion", gr.size)
            if spent + actual_cost <= budget_usd:
                placements.append(Placement("grass_conversion", gr, gc))
                spent += actual_cost
                used[gr, gc] = True
            else:
                affordable_n = ctx.affordable("grass_conversion", budget_usd - spent)
                if affordable_n > 0:
                    gr, gc = gr[:affordable_n], gc[:affordable_n]
                    placements.append(Placement("grass_conversion", gr, gc))
                    spent += ctx.cost("grass_conversion", gr.size)
                    used[gr, gc] = True

    # ── Phase 5: Use any remaining budget for more shade canopies ────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("shade_canopy"):
        covered2 = np.zeros(ctx.shape, dtype=bool)
        covered2[ctx.cdsm > 0.0] = True
        if mr.size:
            crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
            _stamp_crown(covered2, mr, mc, crown_med_px, ctx.shape)
        if sr.size:
            crown_small_px = max(int(round(CROWN_SMALL_M / ctx.res_m)), 1)
            _stamp_crown(covered2, sr, sc, crown_small_px, ctx.shape)

        open_ground2 = ctx.buildable & ~covered2 & ~used
        n_extra      = ctx.affordable("shade_canopy", remaining)
        n_extra      = min(n_extra, int(open_ground2.sum()))

        if n_extra > 0:
            er, ec = _top_pixels(score, open_ground2, n_extra)
            if er.size:
                actual_cost = ctx.cost("shade_canopy", er.size)
                if spent + actual_cost <= budget_usd:
                    placements.append(Placement("shade_canopy", er, ec))
                    spent += actual_cost
                else:
                    affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
                    if affordable_n > 0:
                        placements.append(Placement("shade_canopy", er[:affordable_n], ec[:affordable_n]))

    return placements