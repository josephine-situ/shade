from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-triple-action greened corridors v1"
DESCRIPTION = (
    "Maximise heat_relief_c and push cobenefit_greened_pct above threshold by combining: "
    "(1) Medium trees at 3m spacing on highest synergy (heat×pop) corridors (65% budget); "
    "(2) Small trees at 2.5m spacing filling crown gaps (18% budget); "
    "(3) Shade canopies on hottest remaining open ground (12% budget); "
    "(4) Grass conversion on remaining paved budget to boost greening (5% budget). "
    "Priority surface uses geometric mean synergy (heat×pop) with 0.25 priority-tract boost "
    "to maintain equity_ratio > 1. No reflective surfaces (albedo trap avoided)."
)

# Priority-tract boost — strong enough to keep equity_ratio > 1
PRIORITY_BOOST = 0.25

# Budget fractions
FRAC_MED    = 0.65
FRAC_SML    = 0.18
FRAC_CANOPY = 0.12
FRAC_GRASS  = 0.05

# Spacing parameters
SPACING_MED_M = 3.0   # tight spacing for dense shade corridors
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
    Synergy-based composite priority surface.
    Geometric mean of heat × population identifies pixels simultaneously
    hot AND populated — these yield the highest person-degC relief per tree.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    # Synergy: pixels scoring high on BOTH heat AND population
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.45 * synergy
        + 0.20 * heat_n
        + 0.15 * pop_n
        + 0.12 * vuln_n
        + 0.05 * hours_n
        + 0.03 * uhii_n
    )

    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def canopy_priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for shade canopy placement — heavier synergy weight.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.50 * synergy
        + 0.20 * heat_n
        + 0.15 * pop_n
        + 0.10 * vuln_n
        + 0.05 * hours_n
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def grass_priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for grass conversion — target hot paved areas
    with pedestrian exposure to maximise both cooling and greening.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.40 * synergy
        + 0.30 * heat_n
        + 0.20 * pop_n
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
    grass_score  = grass_priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0

    # Track which pixels are already used (no double-booking)
    used    = np.zeros(ctx.shape, dtype=bool)
    # Track canopy/shade coverage (existing + new trees)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # existing canopy already covered

    # ── 1. Medium street trees ───────────────────────────────────────────
    # 65% of budget, 3 m spacing for dense shade corridors
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
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_exclusion(covered, tr_m, tc_m, crown_med_px, ctx.shape)

    # ── 2. Small street trees ────────────────────────────────────────────
    # 18% of budget, 2.5 m spacing, fills crown gaps between medium trees
    sml_budget_target = budget_usd * FRAC_SML
    remaining_for_sml = budget_usd - spent
    sml_budget = min(sml_budget_target, remaining_for_sml)

    tr_s = tc_s = np.array([], dtype=int)
    if sml_budget >= ctx.unit_cost("tree_small"):
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
            crown_sml_px = max(int(round(CROWN_SML_M / ctx.res_m)), 1)
            _stamp_exclusion(covered, tr_s, tc_s, crown_sml_px, ctx.shape)

    # ── 3. Shade canopies ────────────────────────────────────────────────
    # 12% of budget target; fill hottest open buildable ground not already shaded
    canopy_budget_target = budget_usd * FRAC_CANOPY
    remaining = budget_usd - spent
    canopy_budget = min(canopy_budget_target, remaining)

    cr = cc = np.array([], dtype=int)
    if canopy_budget >= ctx.unit_cost("shade_canopy"):
        open_ground = ctx.buildable & ~covered & ~used

        n_canopy_max = ctx.affordable("shade_canopy", canopy_budget)
        n_canopy = min(n_canopy_max, int(open_ground.sum()))

        cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
        if cr.size:
            actual_cost = ctx.cost("shade_canopy", cr.size)
            if spent + actual_cost <= budget_usd + 0.01:
                placements.append(Placement("shade_canopy", cr, cc))
                spent += actual_cost
                # Mark canopy pixels as covered and used
                used[cr, cc] = True
                covered[cr, cc] = True
            else:
                affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
                if affordable_n > 0:
                    cr_trim, cc_trim = cr[:affordable_n], cc[:affordable_n]
                    placements.append(Placement("shade_canopy", cr_trim, cc_trim))
                    spent += ctx.cost("shade_canopy", affordable_n)
                    used[cr_trim, cc_trim] = True
                    covered[cr_trim, cc_trim] = True

    # ── 4. Grass conversion ──────────────────────────────────────────────
    # 5% of budget target; depave hot paved areas to boost greening metric
    # and provide evaporative cooling. Only on paved, walkable, exposed areas.
    grass_budget_target = budget_usd * FRAC_GRASS
    remaining = budget_usd - spent
    grass_budget = min(grass_budget_target, remaining)

    if grass_budget >= ctx.unit_cost("grass_conversion"):
        # Eligible paved pixels not already used, on walkable exposure corridor
        # grass_conversion lc=[1] (paved), not roadbed
        cand_grass = (
            ctx.placeable("grass_conversion")
            & ~used
            & ctx.exposure
            & ~ctx.roadbed
        )

        n_grass_max = ctx.affordable("grass_conversion", grass_budget)
        n_grass = min(n_grass_max, int(cand_grass.sum()))

        gr, gc = _top_pixels(grass_score, cand_grass, n_grass)
        if gr.size:
            actual_cost = ctx.cost("grass_conversion", gr.size)
            if spent + actual_cost <= budget_usd + 0.01:
                placements.append(Placement("grass_conversion", gr, gc))
                spent += actual_cost
                used[gr, gc] = True
            else:
                affordable_n = ctx.affordable("grass_conversion", budget_usd - spent)
                if affordable_n > 0:
                    placements.append(
                        Placement("grass_conversion", gr[:affordable_n], gc[:affordable_n])
                    )

    return placements