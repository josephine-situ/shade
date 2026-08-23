from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "equity-dense-shade-greening v1"
DESCRIPTION = (
    "Cell (1,1,0,1): high equity, high access, LOW cost-efficiency, high greening. "
    "Maximise absolute UTCI relief with strong equity weighting to maintain "
    "equity_ratio >= 1. Phase 1: Medium trees at 3m spacing (60% budget) on "
    "highest heat×pop×vulnerability synergy corridors. Phase 2: Shade canopies "
    "on open hot buildable ground (20% budget). Phase 3: Grass conversion for "
    "cobenefit_greened_pct (10% budget). Phase 4: Green roofs on buildings (10% "
    "budget) to raise cost per unit and keep cost_efficiency LOW while adding "
    "greening. Strong priority boost (0.35) ensures equity_ratio >= 1. "
    "Composite score: 0.35 synergy(heat×pop) + 0.25 heat + 0.15 vulnerability "
    "+ 0.15 pop + 0.10 heat_hours. No reflective surfaces (albedo trap avoided)."
)

PRIORITY_BOOST = 0.35

FRAC_TREES   = 0.60
FRAC_CANOPY  = 0.20
FRAC_GRASS   = 0.10
FRAC_GROOF   = 0.10

SPACING_MED_M = 3.0
CROWN_MED_M   = 3.5


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
    Heat×population×vulnerability synergy scoring.
    Strong equity weighting to keep equity_ratio >= 1.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)

    # Geometric mean synergy: high only if BOTH heat and population are high
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.35 * synergy
        + 0.25 * heat_n
        + 0.15 * vuln_n
        + 0.15 * pop_n
        + 0.10 * hours_n
    )
    # Strong priority-tract boost for equity_ratio >= 1
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _canopy_score(ctx: PlanningContext) -> np.ndarray:
    """Score for shade canopy: synergy + vulnerability focus."""
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.40 * synergy
        + 0.22 * heat_n
        + 0.18 * vuln_n
        + 0.12 * pop_n
        + 0.05 * hours_n
        + 0.03 * uhii_n
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _grass_score(ctx: PlanningContext) -> np.ndarray:
    """Score for grass conversion: hot paved areas with population."""
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


def _roof_score(ctx: PlanningContext) -> np.ndarray:
    """Score for green roofs: target hot buildings in vulnerable tracts."""
    mask = ctx.exposure

    heat_n = _norm(ctx.heat_ta3pm,    mask)
    vuln_n = _norm(ctx.vulnerability, mask)
    uhii_n = _norm(ctx.heat_uhii,     mask)

    score = (
          0.45 * heat_n
        + 0.35 * vuln_n
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
    """Greedy selection with spatial suppression."""
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
    """Select top `limit` candidate pixels by score."""
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
    """Mark crown footprint around each tree as covered."""
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
    """Trim placement to remaining budget."""
    if rows.size == 0:
        return rows, cols, spent
    remaining = budget_usd - spent
    n_max = ctx.affordable(action, remaining)
    if n_max <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), spent
    rows = rows[:n_max]
    cols = cols[:n_max]
    spent += ctx.cost(action, rows.size)
    return rows, cols, spent


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    sc_tree   = _synergy_score(ctx)
    sc_canopy = _canopy_score(ctx)
    sc_grass  = _grass_score(ctx)
    sc_roof   = _roof_score(ctx)

    placements: list[Placement] = []
    spent = 0.0
    used    = np.zeros(ctx.shape, dtype=bool)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # pre-existing canopy

    # ── Phase 1: Medium street trees (60% budget, 3m spacing) ────────────
    # Tight 3m spacing maximises corridor shade coverage and absolute relief.
    med_budget     = budget_usd * FRAC_TREES
    n_med_max      = ctx.affordable("tree_medium", med_budget)
    cand_med       = ctx.plantable & ~used
    n_med          = min(n_med_max, int(cand_med.sum()))
    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)

    mr, mc = _greedy_spaced(sc_tree, cand_med, spacing_med_px, n_med)
    if mr.size:
        mr, mc, spent = _safe_place(ctx, "tree_medium", mr, mc, spent, budget_usd)
        if mr.size:
            placements.append(Placement("tree_medium", mr, mc))
            used[mr, mc] = True
            crown_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
            _stamp_crown(covered, mr, mc, crown_px, ctx.shape)

    # ── Phase 2: Shade canopies (20% budget, open hot buildable ground) ───
    canopy_budget = min(budget_usd * FRAC_CANOPY, budget_usd - spent)
    cr = cc = np.array([], dtype=int)
    if canopy_budget >= ctx.unit_cost("shade_canopy"):
        open_ground  = ctx.buildable & ~covered & ~used
        n_canopy_max = ctx.affordable("shade_canopy", canopy_budget)
        n_canopy     = min(n_canopy_max, int(open_ground.sum()))

        cr, cc = _top_pixels(sc_canopy, open_ground, n_canopy)
        if cr.size:
            cr, cc, spent = _safe_place(ctx, "shade_canopy", cr, cc, spent, budget_usd)
            if cr.size:
                placements.append(Placement("shade_canopy", cr, cc))
                used[cr, cc] = True

    # ── Phase 3: Grass conversion (10% budget, cobenefit_greened_pct) ─────
    grass_budget = min(budget_usd * FRAC_GRASS, budget_usd - spent)
    if grass_budget >= ctx.unit_cost("grass_conversion"):
        cand_grass  = ctx.placeable("grass_conversion") & ~ctx.roadbed & ~used
        n_grass_max = ctx.affordable("grass_conversion", grass_budget)
        n_grass     = min(n_grass_max, int(cand_grass.sum()))

        gr, gc = _top_pixels(sc_grass, cand_grass, n_grass)
        if gr.size:
            gr, gc, spent = _safe_place(ctx, "grass_conversion", gr, gc, spent, budget_usd)
            if gr.size:
                placements.append(Placement("grass_conversion", gr, gc))
                used[gr, gc] = True

    # ── Phase 4: Green roofs (10% budget, hot vulnerable buildings) ───────
    # Green roofs are expensive ($377/m2) which keeps cost_efficiency LOW
    # while providing genuine cooling and greening benefits.
    groof_budget = min(budget_usd * FRAC_GROOF, budget_usd - spent)
    if groof_budget >= ctx.unit_cost("green_roof"):
        cand_roof  = ctx.placeable("green_roof") & ~used
        n_roof_max = ctx.affordable("green_roof", groof_budget)
        n_roof     = min(n_roof_max, int(cand_roof.sum()))

        rr, rc = _top_pixels(sc_roof, cand_roof, n_roof)
        if rr.size:
            rr, rc, spent = _safe_place(ctx, "green_roof", rr, rc, spent, budget_usd)
            if rr.size:
                placements.append(Placement("green_roof", rr, rc))
                used[rr, rc] = True

    # ── Phase 5: Remaining budget → more medium trees ─────────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("tree_medium"):
        cand_extra = ctx.plantable & ~used
        n_extra    = min(ctx.affordable("tree_medium", remaining), int(cand_extra.sum()))
        if n_extra > 0:
            spacing_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
            xr, xc = _greedy_spaced(sc_tree, cand_extra, spacing_px, n_extra)
            if xr.size:
                xr, xc, spent = _safe_place(ctx, "tree_medium", xr, xc, spent, budget_usd)
                if xr.size:
                    placements.append(Placement("tree_medium", xr, xc))
                    used[xr, xc] = True
                    crown_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
                    _stamp_crown(covered, xr, xc, crown_px, ctx.shape)

    # ── Phase 6: Remaining → more shade canopies ──────────────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("shade_canopy"):
        covered2     = np.zeros(ctx.shape, dtype=bool)
        covered2[ctx.cdsm > 0.0] = True
        open_ground2 = ctx.buildable & ~covered2 & ~used
        n_extra2     = min(ctx.affordable("shade_canopy", remaining), int(open_ground2.sum()))
        if n_extra2 > 0:
            er, ec = _top_pixels(sc_canopy, open_ground2, n_extra2)
            if er.size:
                er, ec, spent = _safe_place(ctx, "shade_canopy", er, ec, spent, budget_usd)
                if er.size:
                    placements.append(Placement("shade_canopy", er, ec))
                    used[er, ec] = True

    # ── Phase 7: Final remainder → more grass conversion ──────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("grass_conversion"):
        cand_grass2 = ctx.placeable("grass_conversion") & ~ctx.roadbed & ~used
        n_grass2    = min(ctx.affordable("grass_conversion", remaining), int(cand_grass2.sum()))
        if n_grass2 > 0:
            gr2, gc2 = _top_pixels(sc_grass, cand_grass2, n_grass2)
            if gr2.size:
                gr2, gc2, spent = _safe_place(
                    ctx, "grass_conversion", gr2, gc2, spent, budget_usd
                )
                if gr2.size:
                    placements.append(Placement("grass_conversion", gr2, gc2))

    return placements