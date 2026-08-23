from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-dense-medium-canopy v4"
DESCRIPTION = (
    "Maximise population-weighted UTCI relief via aggressive synergy scoring. "
    "Phase 1: Medium trees at 3m spacing consume 70% of budget targeting "
    "pixels with highest geometric-mean heat×population synergy. "
    "Phase 2: Shade canopies on remaining open hot ground (20% budget). "
    "Phase 3: Small trees fill remaining plantable gaps (10% budget). "
    "Priority surface uses geometric-mean synergy (heat×pop) as dominant term "
    "(0.50) to find pixels that are BOTH hot AND populated, plus "
    "heat_ta3pm (0.25), population (0.15), heat_hours (0.07), uhii (0.03). "
    "No equity boost to maximise raw population-weighted UTCI fitness. "
    "Reflective surfaces entirely avoided (albedo trap)."
)

# Budget fractions
FRAC_MED    = 0.70   # medium trees dominate — strongest UTCI per dollar in corridors
FRAC_CANOPY = 0.20   # shade canopies for rapid UTCI drop on hot open ground
FRAC_SML    = 0.10   # small trees for gap-fill

# Spacing
SPACING_MED_M = 3.0   # tight for dense shade
SPACING_SML_M = 3.0

# Crown radii for exclusion tracking
CROWN_MED_M   = 3.5
CROWN_SML_M   = 2.0


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0,1] over masked region; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def _synergy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Heat×population synergy surface for tree and canopy placement.
    Geometric mean of heat×population ensures pixels BOTH hot AND populated
    score highest — directly targeting population-weighted UTCI fitness.
    No equity boost: pure fitness maximisation.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,  mask)
    pop_n   = _norm(ctx.population,  mask)
    hours_n = _norm(ctx.heat_hours,  mask)
    uhii_n  = _norm(ctx.heat_uhii,   mask)

    # Geometric mean: rewards pixels that are BOTH hot AND populated
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.50 * synergy
        + 0.25 * heat_n
        + 0.15 * pop_n
        + 0.07 * hours_n
        + 0.03 * uhii_n
    )
    return np.where(mask, score, -np.inf)


def _canopy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Score for shade canopy placement: heavy synergy focus.
    Canopies have high cost/m2 so must land on highest-value pixels.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,  mask)
    pop_n   = _norm(ctx.population,  mask)
    hours_n = _norm(ctx.heat_hours,  mask)
    uhii_n  = _norm(ctx.heat_uhii,   mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.55 * synergy
        + 0.25 * heat_n
        + 0.12 * pop_n
        + 0.05 * hours_n
        + 0.03 * uhii_n
    )
    return np.where(mask, score, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy spatial selection: pick best candidate, suppress neighbourhood,
    repeat until `limit` reached or candidates exhausted.
    """
    rows, cols = np.nonzero(candidates)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)

    order = np.argsort(-score[rows, cols])
    rows, cols = rows[order], cols[order]

    taken  = np.zeros(score.shape, dtype=bool)
    span   = max(int(spacing_px), 1)
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
    """Select top `limit` candidate pixels by score (no spacing constraint)."""
    rows, cols = np.nonzero(candidates)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    order = np.argsort(-score[rows, cols])[:limit]
    return rows[order], cols[order]


def _stamp_crown(
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


def _safe_place(
    ctx: PlanningContext,
    action: str,
    rows: np.ndarray,
    cols: np.ndarray,
    spent: float,
    budget_usd: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Trim placement to fit remaining budget; return (rows, cols, new_spent)."""
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

    placements: list[Placement] = []
    spent = 0.0
    used    = np.zeros(ctx.shape, dtype=bool)
    covered = np.zeros(ctx.shape, dtype=bool)
    # Mark pre-existing canopy
    covered[ctx.cdsm > 0.0] = True

    # ── Phase 1: Medium street trees (70% budget, 3 m spacing) ──────────
    med_budget     = budget_usd * FRAC_MED
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

    # ── Phase 2: Shade canopies (20% budget, hottest open buildable ground)
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

    # ── Phase 3: Small trees (10% budget, gap-fill) ──────────────────────
    sml_budget = min(budget_usd * FRAC_SML, budget_usd - spent)
    sr = sc_arr = np.array([], dtype=int)
    if sml_budget >= ctx.unit_cost("tree_small"):
        cand_sml       = ctx.plantable & ~used
        n_sml_max      = ctx.affordable("tree_small", sml_budget)
        n_sml          = min(n_sml_max, int(cand_sml.sum()))
        spacing_sml_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)

        sr, sc_arr = _greedy_spaced(sc_tree, cand_sml, spacing_sml_px, n_sml)
        if sr.size:
            sr, sc_arr, spent = _safe_place(ctx, "tree_small", sr, sc_arr, spent, budget_usd)
            if sr.size:
                placements.append(Placement("tree_small", sr, sc_arr))
                used[sr, sc_arr] = True
                crown_sml_px = max(int(round(CROWN_SML_M / ctx.res_m)), 1)
                _stamp_crown(covered, sr, sc_arr, crown_sml_px, ctx.shape)

    # ── Phase 4: Remaining budget → more medium trees ────────────────────
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

    # ── Phase 5: Remaining → more shade canopies ─────────────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("shade_canopy"):
        open_ground2 = ctx.buildable & ~covered & ~used
        n_extra2     = min(ctx.affordable("shade_canopy", remaining), int(open_ground2.sum()))
        if n_extra2 > 0:
            er, ec = _top_pixels(sc_canopy, open_ground2, n_extra2)
            if er.size:
                er, ec, spent = _safe_place(ctx, "shade_canopy", er, ec, spent, budget_usd)
                if er.size:
                    placements.append(Placement("shade_canopy", er, ec))
                    used[er, ec] = True

    # ── Phase 6: Final remainder → small trees ───────────────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("tree_small"):
        cand_sml2  = ctx.plantable & ~used
        n_sml2     = min(ctx.affordable("tree_small", remaining), int(cand_sml2.sum()))
        if n_sml2 > 0:
            spacing_px2 = max(int(round(SPACING_SML_M / ctx.res_m)), 1)
            sr2, sc2 = _greedy_spaced(sc_tree, cand_sml2, spacing_px2, n_sml2)
            if sr2.size:
                sr2, sc2, spent = _safe_place(ctx, "tree_small", sr2, sc2, spent, budget_usd)
                if sr2.size:
                    placements.append(Placement("tree_small", sr2, sc2))

    return placements