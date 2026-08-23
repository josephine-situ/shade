from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-dense shade corridors v1"
DESCRIPTION = (
    "Maximise heat_relief_c via geometric-mean synergy priority (heat×pop) "
    "with tighter 2.5m medium-tree spacing for maximum corridor shade density. "
    "Budget split: 70% medium trees (2.5m spacing), 13% small trees (gap-fill), "
    "17% shade canopies on remaining hot open ground. "
    "Priority surface: synergy(heat×pop) 0.50, heat_ta3pm 0.20, population 0.12, "
    "vulnerability 0.10, heat_hours 0.05, UHII 0.03. "
    "Strong priority-tract boost (0.20) for equity. Albedo surfaces avoided."
)

# Budget fractions
FRAC_MED    = 0.70
FRAC_SML    = 0.13
FRAC_CANOPY = 0.17  # remainder

# Spacing
SPACING_MED_M = 2.5   # very tight for maximum corridor shade
SPACING_SML_M = 2.0   # gap-fill

# Crown radii for exclusion (used to avoid wasting canopy on shaded ground)
CROWN_MED_M = 3.5
CROWN_SML_M = 2.0

# Priority boost for top-quartile vulnerability tracts
PRIORITY_BOOST = 0.20


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
    Synergy-based composite priority surface.
    Geometric mean of normalised heat × normalised population rewards pixels
    that are simultaneously hot AND heavily used — maximising person-degC
    relief per dollar spent.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    # Synergy: geometric mean — zero if either factor is zero
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.50 * synergy
        + 0.20 * heat_n
        + 0.12 * pop_n
        + 0.10 * vuln_n
        + 0.05 * hours_n
        + 0.03 * uhii_n
    )

    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def canopy_priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Canopy-specific priority surface: even stronger synergy weight since
    canopies provide immediate shade and are best placed where hot+crowded.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.55 * synergy
        + 0.18 * heat_n
        + 0.13 * pop_n
        + 0.09 * vuln_n
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
    Greedy placement with spatial exclusion zone.
    Pick highest-scoring candidate, suppress spacing_px-radius neighbourhood, repeat.
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

    used    = np.zeros(ctx.shape, dtype=bool)
    covered = np.zeros(ctx.shape, dtype=bool)
    # Pre-mark existing canopy as covered
    covered[ctx.cdsm > 0.0] = True

    # ── Phase 1: Medium street trees (70% budget, 2.5m spacing) ─────────
    # Tighter spacing than synergy-v2 (3m→2.5m) for even denser corridor shade
    med_budget     = budget_usd * FRAC_MED
    n_med_max      = ctx.affordable("tree_medium", med_budget)
    cand_med       = ctx.plantable & ~used
    n_med          = min(n_med_max, int(cand_med.sum()))
    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)

    tr_m, tc_m = _greedy_spaced(tree_score, cand_med, spacing_med_px, n_med)

    if tr_m.size:
        placements.append(Placement("tree_medium", tr_m, tc_m))
        spent += ctx.cost("tree_medium", tr_m.size)
        used[tr_m, tc_m] = True
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_exclusion(covered, tr_m, tc_m, crown_med_px, ctx.shape)

    # ── Phase 2: Small street trees (13% budget, 2.0m spacing) ──────────
    # Fill gaps between medium trees; exclude areas under new medium crowns
    sml_budget_target = budget_usd * FRAC_SML
    remaining_sml     = budget_usd - spent
    sml_budget        = min(sml_budget_target, remaining_sml)

    tr_s = np.array([], dtype=int)
    tc_s = np.array([], dtype=int)

    if sml_budget >= ctx.unit_cost("tree_small"):
        n_sml_max      = ctx.affordable("tree_small", sml_budget)
        # Gap-fill: plantable, not already used, not already under a crown
        cand_sml       = ctx.plantable & ~used & ~covered
        n_sml          = min(n_sml_max, int(cand_sml.sum()))
        spacing_sml_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)

        tr_s, tc_s = _greedy_spaced(tree_score, cand_sml, spacing_sml_px, n_sml)

        if tr_s.size:
            placements.append(Placement("tree_small", tr_s, tc_s))
            spent += ctx.cost("tree_small", tr_s.size)
            used[tr_s, tc_s] = True
            crown_sml_px = max(int(round(CROWN_SML_M / ctx.res_m)), 1)
            _stamp_exclusion(covered, tr_s, tc_s, crown_sml_px, ctx.shape)

    # ── Phase 3: Shade canopies (remaining budget) ───────────────────────
    # Target hottest open buildable ground not already covered by tree crowns
    remaining = budget_usd - spent
    if remaining < ctx.unit_cost("shade_canopy"):
        return placements

    # Open ground: buildable, no existing or new canopy coverage, not used
    open_ground = ctx.buildable & ~covered & ~used

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
    if cr.size:
        actual_cost = ctx.cost("shade_canopy", cr.size)
        if spent + actual_cost <= budget_usd + 0.01:
            placements.append(Placement("shade_canopy", cr, cc))
        else:
            affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
            if affordable_n > 0:
                placements.append(
                    Placement("shade_canopy", cr[:affordable_n], cc[:affordable_n])
                )

    return placements