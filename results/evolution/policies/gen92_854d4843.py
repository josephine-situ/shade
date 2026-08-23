from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "dense-synergy-max-trees v2"
DESCRIPTION = (
    "Maximise heat_relief_c in cell (0,0,1,0): no equity boost, low access "
    "threshold, high cost-efficiency, low greening. "
    "Strategy: aggressive medium-tree planting at minimum spacing (1px) on "
    "highest heat×population synergy pixels, consuming 75% of budget. "
    "Then shade canopies on remaining hot open ground (25% budget). "
    "Priority surface: pure heat×population geometric-mean synergy with "
    "strong heat weighting. NO priority-tract boost for maximum absolute "
    "UTCI relief. Spacing reduced to 1px (2m minimum) to maximise tree density "
    "and canopy closure on hottest corridors. Reflective surfaces avoided."
)

FRAC_MED    = 0.75
FRAC_CANOPY = 0.25

# Minimum spacing: 1 pixel (2m) -- pack trees as densely as plantable allows
SPACING_MED_PX = 1

CROWN_MED_M = 3.5   # crown radius for canopy exclusion after tree placement


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
    Strong heat×population synergy scoring for tree placement.
    Geometric mean ensures BOTH heat AND population must be high.
    No priority boost — pure heat-first for maximum UTCI relief.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm, mask)
    pop_n   = _norm(ctx.population, mask)
    hours_n = _norm(ctx.heat_hours, mask)
    uhii_n  = _norm(ctx.heat_uhii,  mask)

    # Geometric mean synergy
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    # Strong synergy + heat weighting, minimal other factors
    score = (
          0.55 * synergy
        + 0.28 * heat_n
        + 0.10 * pop_n
        + 0.05 * hours_n
        + 0.02 * uhii_n
    )
    # No priority boost
    return np.where(mask, score, -np.inf)


def _canopy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for shade canopy: maximize population×heat synergy.
    Strong synergy focus for maximum person-degC impact per m2.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm, mask)
    pop_n   = _norm(ctx.population, mask)
    hours_n = _norm(ctx.heat_hours, mask)
    uhii_n  = _norm(ctx.heat_uhii,  mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.58 * synergy
        + 0.24 * heat_n
        + 0.10 * pop_n
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
    Greedy selection: pick highest-scoring candidate, suppress spacing_px
    neighbourhood, repeat until limit reached or candidates exhausted.
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

    placements: list[Placement] = []
    spent = 0.0
    used    = np.zeros(ctx.shape, dtype=bool)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True  # pre-existing canopy

    # ── Phase 1: Medium street trees (75% budget, minimum 1px spacing) ────
    # Maximum tree density on hottest, most-populated corridors.
    # 1px spacing means trees at every eligible pixel — limited only by
    # plantable mask and budget.
    med_budget = budget_usd * FRAC_MED
    n_med_max  = ctx.affordable("tree_medium", med_budget)
    cand_med   = ctx.plantable & ~used
    n_med      = min(n_med_max, int(cand_med.sum()))

    mr, mc = _greedy_spaced(tree_sc, cand_med, SPACING_MED_PX, n_med)

    if mr.size:
        mr, mc, spent = _safe_trim(ctx, "tree_medium", mr, mc, spent, budget_usd)
        if mr.size:
            placements.append(Placement("tree_medium", mr, mc))
            used[mr, mc] = True
            crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
            _stamp_exclusion(covered, mr, mc, crown_med_px, ctx.shape)

    # ── Phase 2: Shade canopies (25% budget, hottest open ground) ──────────
    # Target buildable ground NOT already shaded by new/existing trees.
    # This fills the remaining exposed hot pedestrian space.
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("shade_canopy"):
        open_ground  = ctx.buildable & ~covered & ~used
        n_canopy_max = ctx.affordable("shade_canopy", remaining)
        n_canopy     = min(n_canopy_max, int(open_ground.sum()))

        cr, cc = _top_pixels(canopy_sc, open_ground, n_canopy)
        if cr.size:
            cr, cc, spent = _safe_trim(ctx, "shade_canopy", cr, cc, spent, budget_usd)
            if cr.size:
                placements.append(Placement("shade_canopy", cr, cc))
                used[cr, cc] = True

    # ── Phase 3: Any remaining budget → more medium trees (gap fill) ───────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("tree_medium"):
        cand_extra = ctx.plantable & ~used
        n_extra    = min(ctx.affordable("tree_medium", remaining), int(cand_extra.sum()))
        if n_extra > 0:
            # Still use minimum spacing for maximum density
            er, ec = _greedy_spaced(tree_sc, cand_extra, SPACING_MED_PX, n_extra)
            if er.size:
                er, ec, spent = _safe_trim(ctx, "tree_medium", er, ec, spent, budget_usd)
                if er.size:
                    placements.append(Placement("tree_medium", er, ec))
                    used[er, ec] = True
                    crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
                    _stamp_exclusion(covered, er, ec, crown_med_px, ctx.shape)

    # ── Phase 4: Any remaining budget → more shade canopies ─────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("shade_canopy"):
        open_ground2 = ctx.buildable & ~covered & ~used
        n_extra2     = min(ctx.affordable("shade_canopy", remaining), int(open_ground2.sum()))
        if n_extra2 > 0:
            er2, ec2 = _top_pixels(canopy_sc, open_ground2, n_extra2)
            if er2.size:
                er2, ec2, spent = _safe_trim(ctx, "shade_canopy", er2, ec2, spent, budget_usd)
                if er2.size:
                    placements.append(Placement("shade_canopy", er2, ec2))
                    used[er2, ec2] = True

    # ── Phase 5: Small trees to mop up any last remainder ────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("tree_small"):
        cand_small = ctx.plantable & ~used
        n_small    = min(ctx.affordable("tree_small", remaining), int(cand_small.sum()))
        if n_small > 0:
            sr, sc = _greedy_spaced(tree_sc, cand_small, SPACING_MED_PX, n_small)
            if sr.size:
                sr, sc, spent = _safe_trim(ctx, "tree_small", sr, sc, spent, budget_usd)
                if sr.size:
                    placements.append(Placement("tree_small", sr, sc))

    return placements