from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "dense-equity-shade corridors v1"
DESCRIPTION = (
    "Targets cell (1,0,1,0): equity_ratio>=1, access_gain<0.3232, "
    "cost_efficiency>=40.55, cobenefit_greened<0.1976. "
    "Strategy: dense medium trees at 3m spacing (55% budget) with strong "
    "vulnerability+priority boost for equity; small trees at 2m spacing "
    "gap-fill (20% budget); shade canopies on open hot ground (25% budget). "
    "NO grass conversion to keep cobenefit_greened_pct low. "
    "Focuses on the most extreme heat pixels to maximise UTCI relief per $, "
    "but applies priority-tract boost to steer investment into vulnerable "
    "tracts for equity_ratio>=1. Reflective surfaces avoided entirely."
)

FRAC_MED    = 0.55
FRAC_SML    = 0.20
FRAC_CANOPY = 0.25

SPACING_MED_M = 3.0
SPACING_SML_M = 2.0

CROWN_MED_M = 3.5
CROWN_SML_M = 2.0

PRIORITY_BOOST = 0.30


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
    """
    Priority surface for tree placement.
    Geometric mean of heat × population rewards pixels that are BOTH
    hot AND populated. Strong vulnerability + priority-tract boost ensures
    equity_ratio >= 1.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    # Geometric mean: only high if BOTH heat AND population are high
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.45 * synergy
        + 0.22 * heat_n
        + 0.15 * vuln_n
        + 0.10 * pop_n
        + 0.05 * hours_n
        + 0.03 * uhii_n
    )
    # Priority-tract boost steers investment to vulnerable census tracts
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _canopy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for shade canopy placement.
    Higher vulnerability weight than tree score to maximise equity_ratio
    for the canopy phase.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.40 * synergy
        + 0.20 * heat_n
        + 0.22 * vuln_n
        + 0.10 * pop_n
        + 0.05 * hours_n
        + 0.03 * uhii_n
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
    Greedy spaced selection: pick highest-scoring candidate, suppress
    spacing_px-radius neighbourhood, repeat until limit or exhausted.
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


def _safe_trim(
    ctx: PlanningContext,
    action: str,
    rows: np.ndarray,
    cols: np.ndarray,
    spent: float,
    budget_usd: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Trim placement array to fit within remaining budget."""
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

    placements: list[Placement] = []
    spent = 0.0
    used    = np.zeros(ctx.shape, dtype=bool)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True  # pre-existing canopy

    # ── Phase 1: Medium street trees (55% budget, 3m spacing) ─────────────
    # Dense 3m spacing creates continuous canopy corridors.
    # Priority boost steers to vulnerable tracts for equity_ratio >= 1.
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
            crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
            _stamp_exclusion(covered, mr, mc, crown_med_px, ctx.shape)

    # ── Phase 2: Small street trees (20% budget, 2m spacing, gap-fill) ───
    # Very tight 2m spacing fills gaps in corridors not covered by medium trees.
    sml_budget = min(budget_usd * FRAC_SML, budget_usd - spent)
    sr = sc = np.array([], dtype=int)
    if sml_budget >= ctx.unit_cost("tree_small"):
        n_sml_max = ctx.affordable("tree_small", sml_budget)
        # Exclude pixels under new medium tree crowns or existing canopy
        cand_sml  = ctx.plantable & ~used & ~covered
        n_sml     = min(n_sml_max, int(cand_sml.sum()))

        spacing_sml_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)
        sr, sc = _greedy_spaced(tree_sc, cand_sml, spacing_sml_px, n_sml)

        if sr.size:
            sr, sc, spent = _safe_trim(ctx, "tree_small", sr, sc, spent, budget_usd)
            if sr.size:
                placements.append(Placement("tree_small", sr, sc))
                used[sr, sc] = True
                crown_sml_px = max(int(round(CROWN_SML_M / ctx.res_m)), 1)
                _stamp_exclusion(covered, sr, sc, crown_sml_px, ctx.shape)

    # ── Phase 3: Shade canopies (25% budget, hottest open buildable ground) ─
    # Shade canopies provide immediate strong UTCI relief where trees can't go.
    # Emphasise vulnerability for equity. NO grass conversion (keeps
    # cobenefit_greened_pct < 0.1976 for cell (1,0,1,0)).
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

    # ── Phase 4: Remaining budget → more shade canopies ───────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("shade_canopy"):
        open_ground2 = ctx.buildable & ~covered & ~used
        n_extra      = ctx.affordable("shade_canopy", remaining)
        n_extra      = min(n_extra, int(open_ground2.sum()))

        if n_extra > 0:
            er, ec = _top_pixels(canopy_sc, open_ground2, n_extra)
            if er.size:
                er, ec, spent = _safe_trim(ctx, "shade_canopy", er, ec, spent, budget_usd)
                if er.size:
                    placements.append(Placement("shade_canopy", er, ec))
                    used[er, ec] = True

    # ── Phase 5: Any remaining budget → more medium trees ─────────────────
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

    # ── Phase 6: Final fallback → small trees if budget remains ──────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("tree_small"):
        cand_final = ctx.plantable & ~used
        n_final    = ctx.affordable("tree_small", remaining)
        n_final    = min(n_final, int(cand_final.sum()))

        if n_final > 0:
            spacing_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)
            fr, fc = _greedy_spaced(tree_sc, cand_final, spacing_px, n_final)
            if fr.size:
                fr, fc, spent = _safe_trim(ctx, "tree_small", fr, fc, spent, budget_usd)
                if fr.size:
                    placements.append(Placement("tree_small", fr, fc))

    return placements