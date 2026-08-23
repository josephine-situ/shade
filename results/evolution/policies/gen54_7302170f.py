from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "equity-efficiency canopy optimizer v1"
DESCRIPTION = (
    "Target cell (1,0,1,0): high equity_ratio, high cost_efficiency, low greening. "
    "Maximise person-degC relief per dollar by combining: (1) dense medium trees "
    "on highest heat×population×vulnerability synergy pixels (60% budget, 2.5m "
    "spacing for denser shade); (2) shade canopies on remaining hot buildable "
    "ground (35% budget); (3) small gap-fill trees (5% budget). Priority surface "
    "uses geometric mean synergy of heat×population (0.55 weight) plus "
    "vulnerability (0.25) and heat_hours (0.10) and UHII (0.10). Strong "
    "priority-tract boost (0.35) for equity_ratio > 1. Reflective surfaces "
    "avoided. Solar canopies avoided (high cost, same shade as cloth canopy). "
    "Tight spacing maximises covered pedestrian area per dollar spent."
)

PRIORITY_BOOST = 0.35  # strong equity boost

# Budget allocation
FRAC_MED    = 0.60
FRAC_CANOPY = 0.35
FRAC_SML    = 0.05

# Spacing
SPACING_MED_M = 2.5   # tighter than parent → denser shade
SPACING_SML_M = 2.0

# Crown radii for exclusion
CROWN_MED_M = 3.5
CROWN_SML_M = 2.0


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0,1] over mask; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Synergy priority surface for tree placement.
    Geometric mean of heat × population rewards co-occurring heat and people.
    Vulnerability weighting ensures equity_ratio > 1.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    # Geometric mean synergy: rewards pixels that are BOTH hot AND populated
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.55 * synergy
        + 0.25 * vuln_n
        + 0.10 * hours_n
        + 0.10 * uhii_n
    )

    # Strong priority-tract boost for equity_ratio
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def canopy_priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for shade canopy placement.
    Emphasises heat × population synergy and vulnerability.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.50 * synergy
        + 0.22 * vuln_n
        + 0.12 * heat_n
        + 0.10 * hours_n
        + 0.06 * uhii_n
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
    placements: list[Placement] = []
    spent = 0.0

    # Track used pixels (no double-booking)
    used    = np.zeros(ctx.shape, dtype=bool)
    # Track canopy coverage (existing + new plantings)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # pre-existing canopy

    # ── 1. Medium street trees (60% budget, 2.5m spacing) ───────────────
    # Tighter spacing than parent (3m → 2.5m) for denser shade corridors.
    # Synergy surface targets pixels that are simultaneously hot AND populated,
    # maximising population-weighted UTCI relief (= heat_relief_c fitness).
    # Strong vulnerability boost ensures equity_ratio > 1.
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

    # ── 2. Shade canopies (35% budget, hottest open ground) ─────────────
    # Fill hot open buildable ground with shade canopies.
    # Canopies give strong UTCI relief without counting toward greening %.
    # Placed before small trees to maximize coverage of highest-priority areas.
    canopy_budget_target = budget_usd * FRAC_CANOPY
    remaining_for_canopy = budget_usd - spent
    canopy_budget = min(canopy_budget_target, remaining_for_canopy)

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
                used[cr, cc] = True
            else:
                affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
                if affordable_n > 0:
                    cr, cc = cr[:affordable_n], cc[:affordable_n]
                    placements.append(Placement("shade_canopy", cr, cc))
                    spent += ctx.cost("shade_canopy", cr.size)
                    used[cr, cc] = True

    # ── 3. Small gap-fill trees (5% budget, 2m spacing) ─────────────────
    # Minimal small trees to fill remaining high-priority plantable gaps
    # that medium trees couldn't reach (too expensive or excluded).
    # Keep fraction small to avoid pushing cobenefit_greened_pct over threshold.
    sml_budget_target = budget_usd * FRAC_SML
    remaining_for_sml = budget_usd - spent
    sml_budget = min(sml_budget_target, remaining_for_sml)

    if sml_budget >= ctx.unit_cost("tree_small"):
        n_sml_max = ctx.affordable("tree_small", sml_budget)
        # Plant in gaps not already covered by medium tree crowns or canopies
        cand_sml = ctx.plantable & ~used & ~covered

        spacing_sml_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)
        n_sml = min(n_sml_max, int(cand_sml.sum()))

        tr_s, tc_s = _greedy_spaced(tree_score, cand_sml, spacing_sml_px, n_sml)

        if tr_s.size:
            actual_cost = ctx.cost("tree_small", tr_s.size)
            if spent + actual_cost <= budget_usd + 0.01:
                placements.append(Placement("tree_small", tr_s, tc_s))
                spent += actual_cost
                used[tr_s, tc_s] = True
            else:
                affordable_n = ctx.affordable("tree_small", budget_usd - spent)
                if affordable_n > 0:
                    placements.append(
                        Placement("tree_small", tr_s[:affordable_n], tc_s[:affordable_n])
                    )

    return placements