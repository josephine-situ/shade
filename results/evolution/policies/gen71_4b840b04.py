from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-shade-equity corridors v1"
DESCRIPTION = (
    "Maximise heat_relief_c in cell (1,1,0,0): high equity + high access + "
    "low efficiency + low greening. Strategy: aggressive shade via medium trees "
    "at 2m spacing (65% budget) on heat×population×vulnerability synergy corridors, "
    "then shade canopies on all remaining open hot ground (35% budget). "
    "No grass conversion (keeps cobenefit_greened_pct low). Strong priority-tract "
    "boost (0.35) ensures equity_ratio >= 1. Synergy surface: geometric mean "
    "(heat×pop) 0.50 + heat 0.20 + vulnerability 0.15 + pop 0.10 + hours 0.05. "
    "Tightest possible tree spacing to maximise canopy density and UTCI drop. "
    "Reflective surfaces avoided entirely."
)

PRIORITY_BOOST = 0.35   # strong boost → equity_ratio >= 1

FRAC_MED    = 0.65
FRAC_CANOPY = 0.35

SPACING_MED_M = 2.0   # tightest legal spacing → densest canopy
CROWN_MED_M   = 3.0   # crown exclusion for canopy placement


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0,1] over mask; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def _synergy_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Heat×population geometric-mean synergy surface.
    Strongly rewards pixels that are BOTH hot AND populated.
    Vulnerability and priority-tract boost ensures equity_ratio >= 1.
    """
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    # Geometric mean synergy: rewards co-occurrence of heat AND population
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.50 * synergy
        + 0.20 * heat_n
        + 0.15 * vuln_n
        + 0.10 * pop_n
        + 0.03 * hours_n
        + 0.02 * uhii_n
    )
    # Strong priority-tract boost for equity_ratio >= 1
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _canopy_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for shade canopy placement.
    Focus on hot+populated open ground in priority tracts.
    """
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
        + 0.10 * pop_n
        + 0.03 * hours_n
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


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    tree_score   = _synergy_surface(ctx)
    canopy_score = _canopy_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0

    # Track used pixels (no double-booking)
    used    = np.zeros(ctx.shape, dtype=bool)
    # Track canopy coverage for canopy placement exclusion
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True  # existing canopy already blocks

    # ── Phase 1: Medium street trees (65% budget, 2m spacing) ─────────────
    # Tightest spacing → densest canopy cover → strongest UTCI drop.
    # Synergy surface targets hot+populated corridors in vulnerable tracts.
    med_budget = budget_usd * FRAC_MED
    n_med_max  = ctx.affordable("tree_medium", med_budget)
    cand_med   = ctx.plantable & ~used

    # 2m spacing = 1 pixel at 2m resolution
    spacing_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    n_med      = min(n_med_max, int(cand_med.sum()))

    tr_m, tc_m = _greedy_spaced(tree_score, cand_med, spacing_px, n_med)

    if tr_m.size:
        # Safety trim: ensure we don't overspend
        actual_n = ctx.affordable("tree_medium", med_budget)
        if tr_m.size > actual_n:
            tr_m = tr_m[:actual_n]
            tc_m = tc_m[:actual_n]
        placements.append(Placement("tree_medium", tr_m, tc_m))
        spent += ctx.cost("tree_medium", tr_m.size)
        used[tr_m, tc_m] = True
        # Mark crown exclusion zones for canopy placement
        crown_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_crown(covered, tr_m, tc_m, crown_px, ctx.shape)

    # ── Phase 2: Shade canopies (remaining ~35% budget) ───────────────────
    # Cover all remaining open buildable ground not already shaded.
    # Shade canopies give strong UTCI relief without adding green area
    # (keeping cobenefit_greened_pct low for cell (1,1,0,0)).
    remaining = budget_usd - spent
    if remaining < ctx.unit_cost("shade_canopy"):
        return placements

    open_ground  = ctx.buildable & ~covered & ~used
    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    if n_canopy <= 0:
        return placements

    cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)

    if cr.size:
        actual_cost = ctx.cost("shade_canopy", cr.size)
        if spent + actual_cost <= budget_usd + 0.01:
            placements.append(Placement("shade_canopy", cr, cc))
            spent += actual_cost
        else:
            # Trim to fit budget exactly
            affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
            if affordable_n > 0:
                placements.append(
                    Placement("shade_canopy", cr[:affordable_n], cc[:affordable_n])
                )
                spent += ctx.cost("shade_canopy", affordable_n)

    # ── Phase 3: Spend any remaining budget on more medium trees ──────────
    # If there's still budget left after canopies, place more medium trees
    # in uncovered plantable areas.
    remaining2 = budget_usd - spent
    if remaining2 >= ctx.unit_cost("tree_medium"):
        cand_med2  = ctx.plantable & ~used
        n_med2_max = ctx.affordable("tree_medium", remaining2)
        n_med2     = min(n_med2_max, int(cand_med2.sum()))

        if n_med2 > 0:
            tr_m2, tc_m2 = _greedy_spaced(
                tree_score, cand_med2, spacing_px, n_med2
            )
            if tr_m2.size:
                actual_n2 = ctx.affordable("tree_medium", budget_usd - spent)
                if tr_m2.size > actual_n2:
                    tr_m2 = tr_m2[:actual_n2]
                    tc_m2 = tc_m2[:actual_n2]
                if tr_m2.size > 0:
                    placements.append(Placement("tree_medium", tr_m2, tc_m2))
                    spent += ctx.cost("tree_medium", tr_m2.size)

    return placements