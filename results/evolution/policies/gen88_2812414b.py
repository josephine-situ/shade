from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "dense-synergy-medium-canopy v1"
DESCRIPTION = (
    "Maximise population-weighted UTCI relief via two-phase shading: "
    "(1) Medium trees at 3m spacing consuming 70% of budget on hottest, "
    "most-populated pedestrian corridors using heat×population geometric "
    "synergy scoring; (2) Shade canopies on the hottest remaining open "
    "pedestrian ground with the final 30% of budget. "
    "No small trees or grass conversion — concentrates budget on the "
    "highest-impact interventions. A modest priority-tract boost (0.10) "
    "maintains equity_ratio >= 1 without sacrificing fitness. "
    "Reflective surfaces avoided entirely (albedo trap)."
)

MEDIUM_BUDGET_FRAC = 0.70
# Remaining 30% goes to shade canopies

MEDIUM_SPACING_M = 3.0   # very tight for dense corridor shade
CROWN_MED_M      = 3.5   # medium tree crown radius for exclusion

PRIORITY_BOOST = 0.10    # moderate boost for top-quartile vulnerability tracts


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
    Heat×population geometric-mean synergy scoring.
    Targets pixels that are BOTH hot AND populated for maximum person-degC relief.
    Includes modest vulnerability weighting and priority-tract boost for equity.
    """
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)
    vuln_n  = _norm(ctx.vulnerability, mask)

    # Geometric mean synergy: high only if BOTH heat and population are high
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.45 * synergy
        + 0.25 * heat_n
        + 0.15 * pop_n
        + 0.10 * hours_n
        + 0.05 * vuln_n
    )
    # Moderate priority-tract boost to maintain equity_ratio >= 1
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _canopy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Score for canopy placement: maximize population×heat synergy product
    to squeeze the most person-degC out of each canopy pixel.
    """
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)
    vuln_n  = _norm(ctx.vulnerability, mask)

    # Strong synergy focus for canopy - maximise person-degC impact
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.50 * synergy
        + 0.25 * heat_n
        + 0.15 * pop_n
        + 0.07 * hours_n
        + 0.03 * vuln_n
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
    Greedy selection by score with spatial exclusion zone.
    Pick highest-scoring candidate pixel, suppress spacing_px neighbourhood, repeat.
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
    score        = _synergy_score(ctx)
    canopy_score = _canopy_score(ctx)
    placements: list[Placement] = []
    spent = 0.0
    used  = np.zeros(ctx.shape, dtype=bool)

    # ── Phase 1: Medium street trees (70% budget, 3m spacing) ────────────
    # Medium trees give the strongest per-tree UTCI benefit.
    # Very tight 3m spacing maximises shade corridor density on hottest,
    # most-populated pedestrian space.
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

    # ── Phase 2: Shade canopies (30% budget, hottest open pedestrian ground)
    # Fill the hottest exposed spaces not already shaded by new trees.
    remaining = budget_usd - spent
    if remaining < ctx.unit_cost("shade_canopy"):
        return placements

    # Exclude pixels already under new tree crowns to avoid redundant spend
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # existing canopy already provides shade

    if mr.size:
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)

    open_ground  = ctx.buildable & ~covered & ~used
    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
    if cr.size:
        cr, cc, spent = _safe_place(ctx, "shade_canopy", cr, cc, spent, budget_usd)
        if cr.size:
            placements.append(Placement("shade_canopy", cr, cc))
            used[cr, cc] = True

    # ── Phase 3: Remaining budget → more medium trees if plantable spots remain
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("tree_medium"):
        cand_extra    = ctx.plantable & ~used
        n_extra_max   = ctx.affordable("tree_medium", remaining)
        n_extra       = min(n_extra_max, int(cand_extra.sum()))

        if n_extra > 0:
            # Use tighter spacing for gap-fill pass
            spacing_fill = max(int(round(2.0 / ctx.res_m)), 1)
            er, ec = _greedy_spaced(score, cand_extra, spacing_fill, n_extra)
            if er.size:
                er, ec, spent = _safe_place(
                    ctx, "tree_medium", er, ec, spent, budget_usd
                )
                if er.size:
                    placements.append(Placement("tree_medium", er, ec))
                    used[er, ec] = True

    # ── Phase 4: Any last remainder → more shade canopies ─────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("shade_canopy"):
        covered2 = np.zeros(ctx.shape, dtype=bool)
        covered2[ctx.cdsm > 0.0] = True
        if mr.size:
            crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
            _stamp_crown(covered2, mr, mc, crown_med_px, ctx.shape)

        open_ground2 = ctx.buildable & ~covered2 & ~used
        n_extra2     = ctx.affordable("shade_canopy", remaining)
        n_extra2     = min(n_extra2, int(open_ground2.sum()))

        if n_extra2 > 0:
            er2, ec2 = _top_pixels(canopy_score, open_ground2, n_extra2)
            if er2.size:
                er2, ec2, spent = _safe_place(
                    ctx, "shade_canopy", er2, ec2, spent, budget_usd
                )
                if er2.size:
                    placements.append(Placement("shade_canopy", er2, ec2))

    return placements