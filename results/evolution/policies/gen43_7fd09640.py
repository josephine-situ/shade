from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "equity-medium-tree heat corridors v1"
DESCRIPTION = (
    "Improved equity-focused heat corridor policy. Builds a priority surface "
    "with strong vulnerability weighting (0.30) and heat weighting (0.35) to "
    "maintain equity_ratio > 1 while maximising UTCI relief. Phase 1: medium "
    "trees at 4m spacing (65% budget) for maximum per-tree shade benefit. "
    "Phase 2: small trees as gap-fill at 3m spacing (15% budget). Phase 3: "
    "shade canopies on hottest remaining open ground (20% budget). A large "
    "priority-tract boost (0.20) ensures top-quartile tracts receive "
    "disproportionate investment. Product-based synergy scoring for canopies "
    "maximises person-degC efficiency."
)

# Priority surface weights
WEIGHTS = {
    "heat_ta3pm":    0.35,
    "heat_hours":    0.12,
    "uhii":          0.08,
    "vulnerability": 0.30,
    "population":    0.15,
}

PRIORITY_BOOST = 0.20   # strong boost for top-quartile vulnerability tracts

# Budget allocation fractions
MEDIUM_FRAC = 0.65
SMALL_FRAC  = 0.15
# Remaining ~20% → shade canopies

# Spacing
MEDIUM_SPACING_M = 4.0
SMALL_SPACING_M  = 3.0

# Crown exclusion radii
CROWN_MED_M   = 3.5
CROWN_SMALL_M = 2.0


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
    Composite priority surface. Strong vulnerability weighting maintains
    equity_ratio > 1. Heat weighting drives UTCI relief.
    """
    mask = ctx.exposure
    score = (
          WEIGHTS["heat_ta3pm"]    * _norm(ctx.heat_ta3pm,    mask)
        + WEIGHTS["heat_hours"]    * _norm(ctx.heat_hours,    mask)
        + WEIGHTS["uhii"]          * _norm(ctx.heat_uhii,     mask)
        + WEIGHTS["vulnerability"] * _norm(ctx.vulnerability, mask)
        + WEIGHTS["population"]    * _norm(ctx.population,    mask)
    )
    # Strong boost for top-quartile vulnerability tracts
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def canopy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Canopy placement score: heat × population synergy with vulnerability.
    Uses geometric mean of heat and population to target pixels that are
    BOTH hot AND populated, maximising person-degC efficiency.
    """
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm, mask)
    pop_n   = _norm(ctx.population, mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours, mask)

    # Synergy: geometric mean of heat × population
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    sc = (
          0.40 * synergy
        + 0.25 * heat_n
        + 0.20 * vuln_n
        + 0.10 * pop_n
        + 0.05 * hours_n
    )
    sc = sc + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, sc, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy selection with spatial exclusion.
    Pick highest-scoring candidate, suppress neighbourhood, repeat.
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
    score   = priority_surface(ctx)
    cscore  = canopy_score(ctx)
    placements: list[Placement] = []
    spent   = 0.0
    used    = np.zeros(ctx.shape, dtype=bool)

    # ── Phase 1: Medium street trees (65% budget, 4m spacing) ───────────
    # Medium trees provide the strongest per-tree UTCI reduction.
    # 4m spacing creates dense shade corridors in hottest vulnerable areas.
    medium_budget = budget_usd * MEDIUM_FRAC
    n_med_max     = ctx.affordable("tree_medium", medium_budget)
    cand_med      = ctx.plantable & ~used
    n_med         = min(n_med_max, int(cand_med.sum()))

    spacing_med = max(int(round(MEDIUM_SPACING_M / ctx.res_m)), 1)
    mr, mc = _greedy_spaced(score, cand_med, spacing_med, n_med)

    if mr.size:
        placements.append(Placement("tree_medium", mr, mc))
        spent += ctx.cost("tree_medium", mr.size)
        used[mr, mc] = True

    # ── Phase 2: Small street trees (15% budget, 3m spacing gap-fill) ───
    # Fill remaining plantable gaps between medium trees.
    # Tight 3m spacing maximises canopy coverage in gaps.
    small_budget = min(budget_usd * SMALL_FRAC, budget_usd - spent)
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

    # ── Phase 3: Shade canopies (remaining ~20% budget) ──────────────────
    # Cover hottest remaining open pedestrian ground not shaded by new trees.
    remaining = budget_usd - spent
    if remaining < ctx.unit_cost("shade_canopy"):
        return placements

    # Build covered mask: existing canopy + new tree crowns
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True

    if mr.size:
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)

    if sr.size:
        crown_small_px = max(int(round(CROWN_SMALL_M / ctx.res_m)), 1)
        _stamp_crown(covered, sr, sc, crown_small_px, ctx.shape)

    # Eligible canopy pixels: buildable, not covered, not used
    open_ground  = ctx.buildable & ~covered & ~used
    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(cscore, open_ground, n_canopy)
    if cr.size:
        actual_cost = ctx.cost("shade_canopy", cr.size)
        if spent + actual_cost <= budget_usd:
            placements.append(Placement("shade_canopy", cr, cc))
            spent += actual_cost
        else:
            # Trim to fit exactly within remaining budget
            n_fit = ctx.affordable("shade_canopy", budget_usd - spent)
            if n_fit > 0:
                placements.append(Placement("shade_canopy", cr[:n_fit], cc[:n_fit]))

    return placements