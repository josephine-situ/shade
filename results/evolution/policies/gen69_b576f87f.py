from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-shade-corridors-v3"
DESCRIPTION = (
    "Maximise population-weighted UTCI relief for cell (1,1,0,0). "
    "Three-phase shade deployment: "
    "(1) Medium trees at 4m spacing on hottest populated corridors (60% budget) "
    "using heat×population geometric-mean synergy scoring; "
    "(2) Small trees at 3m spacing fill plantable gaps (15% budget); "
    "(3) Shade canopies on remaining open buildable ground (25% budget). "
    "Strong priority-tract boost (0.25) maintains equity_ratio >= 1. "
    "No reflective/albedo surfaces (albedo trap avoided). "
    "Priority surface: 0.40 heat×pop synergy + 0.25 heat_ta3pm + "
    "0.15 population + 0.10 heat_hours + 0.10 vulnerability. "
    "Targets cell (1,1,0,0): high equity, high access, low efficiency, low greening."
)

# Budget allocation
FRAC_MEDIUM = 0.60
FRAC_SMALL  = 0.15
# Remaining ~25% → shade canopies

# Spacing for trees
MEDIUM_SPACING_M = 4.0
SMALL_SPACING_M  = 3.0

# Crown radii for canopy exclusion
CROWN_MED_M   = 3.5
CROWN_SMALL_M = 2.0

# Equity boost
PRIORITY_BOOST = 0.25


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0,1] over masked region; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def _priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Composite priority surface using heat×population synergy.
    Targets pixels that are BOTH hot AND populated for maximum person-degC relief.
    Vulnerability weighting and priority-tract boost ensure equity_ratio >= 1.
    """
    mask    = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)

    # Geometric mean: high only if BOTH heat AND population are high
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.40 * synergy
        + 0.25 * heat_n
        + 0.15 * pop_n
        + 0.10 * hours_n
        + 0.10 * vuln_n
    )
    # Boost priority tracts for equity
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _canopy_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for shade canopy placement.
    Emphasises heat×population synergy to maximise person-degC relief.
    """
    mask    = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.45 * synergy
        + 0.20 * heat_n
        + 0.15 * pop_n
        + 0.10 * vuln_n
        + 0.05 * hours_n
        + 0.05 * uhii_n
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
    Greedy placement: pick highest-scoring candidate, suppress spacing_px
    neighbourhood, repeat until `limit` reached or candidates exhausted.
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
    """Select top `limit` candidate pixels by score; no spacing constraint."""
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
    remaining    = budget_usd - spent
    affordable_n = ctx.affordable(action, remaining)
    if affordable_n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), spent
    rows  = rows[:affordable_n]
    cols  = cols[:affordable_n]
    spent += ctx.cost(action, rows.size)
    return rows, cols, spent


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    tree_score   = _priority_surface(ctx)
    canopy_score = _canopy_surface(ctx)

    placements: list[Placement] = []
    spent = 0.0
    used  = np.zeros(ctx.shape, dtype=bool)

    # ── Phase 1: Medium street trees (60% budget, 4m spacing) ────────────
    # Medium trees give strongest UTCI benefit per tree.
    # 4m spacing balances density and budget efficiency.
    medium_budget  = budget_usd * FRAC_MEDIUM
    n_medium_max   = ctx.affordable("tree_medium", medium_budget)
    cand_medium    = ctx.plantable & ~used
    n_medium       = min(n_medium_max, int(cand_medium.sum()))
    spacing_medium = max(int(round(MEDIUM_SPACING_M / ctx.res_m)), 1)

    mr, mc = _greedy_spaced(tree_score, cand_medium, spacing_medium, n_medium)
    if mr.size:
        mr, mc, spent = _safe_place(ctx, "tree_medium", mr, mc, spent, budget_usd)
        if mr.size:
            placements.append(Placement("tree_medium", mr, mc))
            used[mr, mc] = True

    # ── Phase 2: Small street trees (15% budget, 3m spacing gap-fill) ────
    # Fill gaps between medium trees — 3× more trees per dollar than medium.
    small_budget  = min(budget_usd * FRAC_SMALL, budget_usd - spent)
    sr = sc = np.array([], dtype=int)
    if small_budget >= ctx.unit_cost("tree_small"):
        cand_small    = ctx.plantable & ~used
        n_small_max   = ctx.affordable("tree_small", small_budget)
        n_small       = min(n_small_max, int(cand_small.sum()))
        spacing_small = max(int(round(SMALL_SPACING_M / ctx.res_m)), 1)

        sr, sc = _greedy_spaced(tree_score, cand_small, spacing_small, n_small)
        if sr.size:
            sr, sc, spent = _safe_place(ctx, "tree_small", sr, sc, spent, budget_usd)
            if sr.size:
                placements.append(Placement("tree_small", sr, sc))
                used[sr, sc] = True

    # ── Phase 3: Shade canopies (remaining budget, hottest open ground) ───
    # Cover hot pedestrian ground not already under tree crowns.
    remaining = budget_usd - spent
    if remaining < ctx.unit_cost("shade_canopy"):
        return placements

    # Build canopy exclusion mask: existing canopy + new tree crowns
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True  # existing canopy overhead

    if mr.size:
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)
    if sr.size:
        crown_sm_px = max(int(round(CROWN_SMALL_M / ctx.res_m)), 1)
        _stamp_crown(covered, sr, sc, crown_sm_px, ctx.shape)

    open_ground  = ctx.buildable & ~covered & ~used
    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
    if cr.size:
        cr, cc, spent = _safe_place(ctx, "shade_canopy", cr, cc, spent, budget_usd)
        if cr.size:
            placements.append(Placement("shade_canopy", cr, cc))
            used[cr, cc] = True

    # ── Phase 4: Drain any leftover budget with more shade canopies ───────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("shade_canopy"):
        # Recompute open ground excluding all used pixels
        open_ground2 = ctx.buildable & ~covered & ~used
        n_extra      = ctx.affordable("shade_canopy", remaining)
        n_extra      = min(n_extra, int(open_ground2.sum()))
        if n_extra > 0:
            er, ec = _top_pixels(canopy_score, open_ground2, n_extra)
            if er.size:
                er, ec, spent = _safe_place(
                    ctx, "shade_canopy", er, ec, spent, budget_usd
                )
                if er.size:
                    placements.append(Placement("shade_canopy", er, ec))

    return placements