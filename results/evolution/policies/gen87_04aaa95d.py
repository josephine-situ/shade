from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "equity-solar-canopy corridors v1"
DESCRIPTION = (
    "Target cell (1,1,0,1): equity_ratio>=1, access_gain>=0.3232, "
    "cost_efficiency<40.55 (LOW), cobenefit_greened_pct>=0.1976. "
    "Strategy: Medium trees (50% budget) on high heat×pop synergy corridors "
    "with strong priority-tract boost (0.35) for equity. Solar canopies "
    "(20% budget) on hot open buildable ground — expensive per m2 keeps "
    "cost_efficiency low while delivering strong shade relief. Green roofs "
    "(15% budget) on hot buildings for cobenefit and low efficiency. "
    "Grass conversion (10% budget) for greened_pct. Remainder to shade "
    "canopies. Reflective surfaces avoided entirely (albedo trap). "
    "Priority surface: 0.45 synergy(heat×pop) + 0.25 heat + 0.15 pop "
    "+ 0.15 vulnerability + priority-tract boost 0.35."
)

FRAC_MED      = 0.50
FRAC_SOLAR    = 0.20
FRAC_GREEN    = 0.15
FRAC_GRASS    = 0.10
FRAC_SHADE    = 0.05

SPACING_MED_M = 4.0
CROWN_MED_M   = 3.5

PRIORITY_BOOST = 0.35


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1] over mask; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def _tree_score(ctx: PlanningContext) -> np.ndarray:
    """Synergy-based priority surface for tree placement with equity boost."""
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.45 * synergy
        + 0.25 * heat_n
        + 0.15 * pop_n
        + 0.10 * vuln_n
        + 0.05 * hours_n
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _solar_canopy_score(ctx: PlanningContext) -> np.ndarray:
    """Priority surface for solar canopy: hot open buildable ground."""
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.50 * synergy
        + 0.20 * heat_n
        + 0.15 * vuln_n
        + 0.10 * pop_n
        + 0.05 * hours_n
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _green_roof_score(ctx: PlanningContext) -> np.ndarray:
    """Priority surface for green roofs: hot building pixels."""
    mask = ctx.walkable  # use broader mask for buildings
    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)
    pop_n   = _norm(ctx.population,    mask)

    score = (
          0.40 * heat_n
        + 0.25 * vuln_n
        + 0.20 * uhii_n
        + 0.15 * pop_n
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST * 0.5, 0.0)
    return np.where(mask, score, -np.inf)


def _grass_score(ctx: PlanningContext) -> np.ndarray:
    """Priority surface for grass conversion: hot paved areas."""
    mask = ctx.walkable
    heat_n = _norm(ctx.heat_ta3pm, mask)
    pop_n  = _norm(ctx.population, mask)
    uhii_n = _norm(ctx.heat_uhii,  mask)
    vuln_n = _norm(ctx.vulnerability, mask)

    score = (
          0.40 * heat_n
        + 0.25 * pop_n
        + 0.20 * uhii_n
        + 0.15 * vuln_n
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST * 0.3, 0.0)
    return np.where(mask, score, -np.inf)


def _shade_score(ctx: PlanningContext) -> np.ndarray:
    """Priority surface for shade canopy."""
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.50 * synergy
        + 0.22 * heat_n
        + 0.15 * vuln_n
        + 0.08 * pop_n
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
    """Greedy spaced selection."""
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
    """Select top `limit` candidate pixels by score."""
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
    """Mark box of radius_px around each (r, c) as covered."""
    H, W = shape
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        r0 = max(0, r - radius_px)
        r1 = min(H, r + radius_px + 1)
        c0 = max(0, c - radius_px)
        c1 = min(W, c + radius_px + 1)
        covered[r0:r1, c0:c1] = True


def _safe_trim(
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
    tree_sc   = _tree_score(ctx)
    solar_sc  = _solar_canopy_score(ctx)
    green_sc  = _green_roof_score(ctx)
    grass_sc  = _grass_score(ctx)
    shade_sc  = _shade_score(ctx)

    placements: list[Placement] = []
    spent = 0.0
    used    = np.zeros(ctx.shape, dtype=bool)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True  # pre-existing canopy

    # ── Phase 1: Medium street trees (50% budget, 4m spacing) ─────────────
    # Strong equity boost drives equity_ratio >= 1
    med_budget = budget_usd * FRAC_MED
    n_med_max  = ctx.affordable("tree_medium", med_budget)
    cand_med   = ctx.plantable & ~used
    n_med      = min(n_med_max, int(cand_med.sum()))

    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    mr, mc = _greedy_spaced(tree_sc, cand_med, spacing_med_px, n_med)

    if mr.size:
        mr, mc, spent = _safe_trim(ctx, "tree_medium", mr, mc, spent, budget_usd)
        if mr.size:
            placements.append(Placement("tree_medium", mr, mc))
            used[mr, mc] = True
            crown_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
            _stamp_exclusion(covered, mr, mc, crown_px, ctx.shape)

    # ── Phase 2: Solar canopies (20% budget, hot open buildable ground) ───
    # Solar canopies cost $2150/m² — expensive, lowers cost_efficiency
    # while providing strong shade (UTCI relief)
    solar_budget = min(budget_usd * FRAC_SOLAR, budget_usd - spent)
    sr = sc_arr = np.array([], dtype=int)
    if solar_budget >= ctx.unit_cost("solar_canopy"):
        open_ground = ctx.buildable & ~covered & ~used
        n_solar_max = ctx.affordable("solar_canopy", solar_budget)
        n_solar     = min(n_solar_max, int(open_ground.sum()))

        sr, sc_arr = _top_pixels(solar_sc, open_ground, n_solar)
        if sr.size:
            sr, sc_arr, spent = _safe_trim(ctx, "solar_canopy", sr, sc_arr, spent, budget_usd)
            if sr.size:
                placements.append(Placement("solar_canopy", sr, sc_arr))
                used[sr, sc_arr] = True
                # Solar canopies provide shade, mark as covered
                _stamp_exclusion(covered, sr, sc_arr, 1, ctx.shape)

    # ── Phase 3: Green roofs (15% budget, hot buildings) ──────────────────
    # Green roofs cost $377/m² on building pixels, provide greening cobenefit
    # and keep cost_efficiency low (high cost per person-degC)
    green_budget = min(budget_usd * FRAC_GREEN, budget_usd - spent)
    if green_budget >= ctx.unit_cost("green_roof"):
        cand_green  = ctx.placeable("green_roof") & ~used
        n_green_max = ctx.affordable("green_roof", green_budget)
        n_green     = min(n_green_max, int(cand_green.sum()))

        gr_r, gr_c = _top_pixels(green_sc, cand_green, n_green)
        if gr_r.size:
            gr_r, gr_c, spent = _safe_trim(ctx, "green_roof", gr_r, gr_c, spent, budget_usd)
            if gr_r.size:
                placements.append(Placement("green_roof", gr_r, gr_c))
                used[gr_r, gr_c] = True

    # ── Phase 4: Grass conversion (10% budget, cobenefit_greened_pct) ─────
    grass_budget = min(budget_usd * FRAC_GRASS, budget_usd - spent)
    if grass_budget >= ctx.unit_cost("grass_conversion"):
        cand_grass  = ctx.placeable("grass_conversion") & ~ctx.roadbed & ~used
        n_grass_max = ctx.affordable("grass_conversion", grass_budget)
        n_grass     = min(n_grass_max, int(cand_grass.sum()))

        gsr, gsc = _top_pixels(grass_sc, cand_grass, n_grass)
        if gsr.size:
            gsr, gsc, spent = _safe_trim(ctx, "grass_conversion", gsr, gsc, spent, budget_usd)
            if gsr.size:
                placements.append(Placement("grass_conversion", gsr, gsc))
                used[gsr, gsc] = True

    # ── Phase 5: Shade canopies (5% budget + remainder) ───────────────────
    shade_budget = min(budget_usd * FRAC_SHADE + (budget_usd - spent), budget_usd - spent)
    if shade_budget >= ctx.unit_cost("shade_canopy"):
        open_shade  = ctx.buildable & ~covered & ~used
        n_shade_max = ctx.affordable("shade_canopy", shade_budget)
        n_shade     = min(n_shade_max, int(open_shade.sum()))

        shr, shc = _top_pixels(shade_sc, open_shade, n_shade)
        if shr.size:
            shr, shc, spent = _safe_trim(ctx, "shade_canopy", shr, shc, spent, budget_usd)
            if shr.size:
                placements.append(Placement("shade_canopy", shr, shc))
                used[shr, shc] = True

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

    # ── Phase 7: Final remainder → more grass conversion ──────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("grass_conversion"):
        cand_g2  = ctx.placeable("grass_conversion") & ~ctx.roadbed & ~used
        n_g2_max = ctx.affordable("grass_conversion", remaining)
        n_g2     = min(n_g2_max, int(cand_g2.sum()))
        if n_g2 > 0:
            gr2, gc2 = _top_pixels(grass_sc, cand_g2, n_g2)
            if gr2.size:
                gr2, gc2, spent = _safe_trim(ctx, "grass_conversion", gr2, gc2, spent, budget_usd)
                if gr2.size:
                    placements.append(Placement("grass_conversion", gr2, gc2))

    return placements