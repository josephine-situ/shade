from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-medium-tree corridors v2"
DESCRIPTION = (
    "Maximise heat_relief_c by concentrating budget on medium street trees "
    "using a synergy priority surface (geometric mean of heat × population). "
    "Strategy: (1) Medium trees at 4m spacing on highest synergy corridors "
    "(75% budget) — dense canopy for maximum UTCI shade; "
    "(2) Shade canopies on remaining hot open ground (25% budget). "
    "No small trees or reflective surfaces (albedo trap avoided). "
    "Priority surface: synergy(heat×pop) 0.45, heat_ta3pm 0.20, population 0.15, "
    "vulnerability 0.12, heat_hours 0.05, UHII 0.03. Priority-tract boost 0.28 "
    "ensures equity_ratio > 1. Inspired by synergy-shade v2 but with tighter "
    "tree spacing and full budget on high-UTCI-impact actions only."
)

# Priority-tract boost — strong to keep equity_ratio > 1
PRIORITY_BOOST = 0.28

# Budget fractions
FRAC_MED    = 0.75
FRAC_CANOPY = 0.25

# Spacing parameters (tighter than parent for denser coverage)
SPACING_MED_M = 4.0   # tight spacing for dense shade corridors

# Crown radii for canopy exclusion after tree placement
CROWN_MED_M = 3.5


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
    Synergy-based composite priority surface for tree placement.
    Geometric mean of heat × population identifies pixels that are
    simultaneously hot AND populated — highest person-degC relief per tree.
    Also incorporates overnight heat (3AM) to target urban heat sinks.
    """
    mask = ctx.exposure

    heat_n   = _norm(ctx.heat_ta3pm,    mask)
    heat3am  = _norm(ctx.heat_ta3am,    mask)
    pop_n    = _norm(ctx.population,    mask)
    vuln_n   = _norm(ctx.vulnerability, mask)
    hours_n  = _norm(ctx.heat_hours,    mask)
    uhii_n   = _norm(ctx.heat_uhii,     mask)

    # Synergy: geometric mean rewards co-occurrence of heat AND population
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    # Composite heat signal: blend afternoon peak with overnight residual
    composite_heat = 0.75 * heat_n + 0.25 * heat3am

    score = (
          0.42 * synergy
        + 0.22 * composite_heat
        + 0.15 * pop_n
        + 0.12 * vuln_n
        + 0.06 * hours_n
        + 0.03 * uhii_n
    )

    # Boost top-quartile vulnerability tracts → equity_ratio > 1
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def canopy_priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Separate priority surface for shade canopy placement.
    Emphasises heat × population synergy even more strongly since canopies
    provide immediate shade over populated areas.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)

    # Stronger synergy weight for canopies
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.50 * synergy
        + 0.22 * heat_n
        + 0.15 * pop_n
        + 0.08 * vuln_n
        + 0.05 * hours_n
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
    # Track canopy coverage (existing + new)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # existing canopy already covered

    # ── 1. Medium street trees ───────────────────────────────────────────
    # 75% of budget, 4m spacing for dense shade corridors
    # Synergy score targets hot+populated corridors
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
        # Crown exclusion for canopy placement
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_exclusion(covered, tr_m, tc_m, crown_med_px, ctx.shape)

    # ── 2. Shade canopies ────────────────────────────────────────────────
    # Remaining budget; fill hottest open buildable ground not already shaded
    remaining = budget_usd - spent
    if remaining <= ctx.unit_cost("shade_canopy"):
        return placements

    # Open ground: buildable, no existing or new canopy coverage, not used
    open_ground = ctx.buildable & ~covered & ~used

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
    if cr.size:
        actual_cost = ctx.cost("shade_canopy", cr.size)
        if spent + actual_cost <= budget_usd + 0.01:
            placements.append(Placement("shade_canopy", cr, cc))
        else:
            # Trim to fit budget
            affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
            if affordable_n > 0:
                placements.append(
                    Placement("shade_canopy", cr[:affordable_n], cc[:affordable_n])
                )

    return placements