from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "dense-shade corridors v3"
DESCRIPTION = (
    "Maximise heat_relief_c in cell (1,1,0,0): high equity + high access, "
    "low cost-efficiency bucket, low greening. "
    "Phase 1: Medium trees at 3m spacing (65% budget) on hottest heat×population "
    "synergy corridors with strong priority-tract boost (0.30) for equity_ratio >= 1. "
    "Phase 2: Small trees at 2m spacing (15% budget) to gap-fill dense shade. "
    "Phase 3: Shade canopies on remaining hot buildable ground (20% budget). "
    "No grass conversion (keeps cobenefit_greened_pct low, maintains cell). "
    "Synergy scoring: 0.45 sqrt(heat×pop) + 0.25 heat_ta3pm + 0.15 pop + "
    "0.10 heat_hours + 0.05 vulnerability + 0.30 priority-tract boost. "
    "Reflective surfaces avoided entirely — albedo trap acknowledged."
)

# Budget allocation
FRAC_MEDIUM = 0.65
FRAC_SMALL  = 0.15
FRAC_CANOPY = 0.20  # remainder

# Spacing
MEDIUM_SPACING_M = 3.0   # tighter → denser shade corridors
SMALL_SPACING_M  = 2.0   # very tight gap-fill

# Crown radii for exclusion
CROWN_MED_M   = 3.5
CROWN_SMALL_M = 2.0

# Equity boost — strong to keep equity_ratio >= 1
PRIORITY_BOOST = 0.30


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0,1] over mask; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def _synergy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Heat×population geometric mean synergy scoring.
    Targets pixels BOTH hot AND populated for max person-degC relief.
    Strong priority-tract boost ensures equity_ratio >= 1.
    """
    mask = ctx.exposure
    heat_n   = _norm(ctx.heat_ta3pm,    mask)
    pop_n    = _norm(ctx.population,    mask)
    hours_n  = _norm(ctx.heat_hours,    mask)
    vuln_n   = _norm(ctx.vulnerability, mask)
    uhii_n   = _norm(ctx.heat_uhii,     mask)

    # Geometric mean: high only if BOTH heat AND population are high
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.45 * synergy
        + 0.25 * heat_n
        + 0.15 * pop_n
        + 0.08 * hours_n
        + 0.04 * vuln_n
        + 0.03 * uhii_n
    )
    # Strong boost for top-quartile vulnerability tracts → equity_ratio >= 1
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy spaced placement: pick highest-scoring valid pixel,
    suppress spacing_px-radius neighbourhood, repeat until limit reached.
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
    crown_px: int,
    shape: tuple[int, int],
) -> None:
    """Mark box of radius crown_px around each (r,c) as covered."""
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
    score      = _synergy_score(ctx)
    placements: list[Placement] = []
    spent      = 0.0
    used       = np.zeros(ctx.shape, dtype=bool)

    # ── Phase 1: Medium street trees (65% budget, 3m spacing) ────────────
    # Dense medium-tree corridors are the primary UTCI driver.
    # Tight 3m spacing maximises canopy fraction along hot pedestrian routes.
    medium_budget  = budget_usd * FRAC_MEDIUM
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

    # ── Phase 2: Small street trees (15% budget, 2m spacing gap-fill) ────
    # Very tight spacing fills gaps between mediums — maximises shade density.
    # Small trees cost ~3x less per tree → more tree placements per dollar.
    small_budget = min(budget_usd * FRAC_SMALL, budget_usd - spent)
    sr = sc = np.array([], dtype=int)
    if small_budget >= ctx.unit_cost("tree_small"):
        cand_small   = ctx.plantable & ~used
        n_small_max  = ctx.affordable("tree_small", small_budget)
        n_small      = min(n_small_max, int(cand_small.sum()))

        spacing_small = max(int(round(SMALL_SPACING_M / ctx.res_m)), 1)
        sr, sc = _greedy_spaced(score, cand_small, spacing_small, n_small)

        if sr.size:
            sr, sc, spent = _safe_place(ctx, "tree_small", sr, sc, spent, budget_usd)
            if sr.size:
                placements.append(Placement("tree_small", sr, sc))
                used[sr, sc] = True

    # ── Phase 3: Shade canopies (remaining budget, hot open ground) ───────
    # Fill remaining buildable open pedestrian ground not already under canopy.
    # Uses same synergy score to target hot+populated pixels.
    remaining = budget_usd - spent
    if remaining < ctx.unit_cost("shade_canopy"):
        return placements

    # Build covered mask: existing canopy + new tree crowns
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True  # existing canopy

    if mr.size:
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)
    if sr.size:
        crown_small_px = max(int(round(CROWN_SMALL_M / ctx.res_m)), 1)
        _stamp_crown(covered, sr, sc, crown_small_px, ctx.shape)

    # Buildable ground not already shaded and not used by trees
    open_ground  = ctx.buildable & ~covered & ~used
    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(score, open_ground, n_canopy)
    if cr.size:
        cr, cc, spent = _safe_place(ctx, "shade_canopy", cr, cc, spent, budget_usd)
        if cr.size:
            placements.append(Placement("shade_canopy", cr, cc))
            used[cr, cc] = True

    # ── Phase 4: Any tiny remainder → more shade canopies ─────────────────
    remaining2 = budget_usd - spent
    if remaining2 >= ctx.unit_cost("shade_canopy"):
        # Refresh covered mask
        covered2 = np.zeros(ctx.shape, dtype=bool)
        covered2[ctx.cdsm > 0.0] = True
        if mr.size:
            crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
            _stamp_crown(covered2, mr, mc, crown_med_px, ctx.shape)
        if sr.size:
            crown_small_px = max(int(round(CROWN_SMALL_M / ctx.res_m)), 1)
            _stamp_crown(covered2, sr, sc, crown_small_px, ctx.shape)

        open_ground2 = ctx.buildable & ~covered2 & ~used
        n_extra      = ctx.affordable("shade_canopy", remaining2)
        n_extra      = min(n_extra, int(open_ground2.sum()))

        if n_extra > 0:
            er, ec = _top_pixels(score, open_ground2, n_extra)
            if er.size:
                er, ec, _ = _safe_place(ctx, "shade_canopy", er, ec, spent, budget_usd)
                if er.size:
                    placements.append(Placement("shade_canopy", er, ec))

    return placements