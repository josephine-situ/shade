from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-equity-triple-shade v2"
DESCRIPTION = (
    "Maximise heat_relief_c in cell (1,1,0,0): triple-action shade deployment "
    "with geometric-mean heat×population synergy surface. "
    "(1) Medium trees at 5m spacing consume 62% of budget on highest synergy "
    "hot+populated corridors; (2) Small trees at 3m spacing fill remaining "
    "plantable gaps (13% budget); (3) Shade canopies cover the hottest remaining "
    "open buildable ground (25% budget). "
    "Priority surface: synergy 0.45 + heat 0.22 + vulnerability 0.15 + "
    "pop 0.10 + heat_hours 0.05 + UHII 0.03. Priority-tract boost 0.35 "
    "ensures equity_ratio > 1. Reflective surfaces avoided entirely."
)

PRIORITY_BOOST = 0.35

FRAC_MED    = 0.62
FRAC_SML    = 0.13
FRAC_CANOPY = 0.25

SPACING_MED_M = 5.0
SPACING_SML_M = 3.0

CROWN_MED_M = 3.5
CROWN_SML_M = 2.0


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
    Synergy-based composite priority surface.
    Geometric mean of heat × population rewards pixels that are simultaneously
    hot AND populated — maximising person-degC relief per tree.
    Strong vulnerability weighting ensures equity_ratio > 1.
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
          0.45 * synergy
        + 0.22 * heat_n
        + 0.15 * vuln_n
        + 0.10 * pop_n
        + 0.05 * hours_n
        + 0.03 * uhii_n
    )

    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def canopy_priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for shade canopy placement.
    Emphasises vulnerability for equity while maintaining heat×pop synergy.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.40 * synergy
        + 0.20 * heat_n
        + 0.20 * vuln_n
        + 0.12 * pop_n
        + 0.05 * hours_n
        + 0.03 * uhii_n
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

    # ── 1. Medium street trees (62% budget, 5m spacing) ─────────────────
    # Medium trees give the strongest single-tree UTCI benefit.
    # 5m spacing (vs parent's 3m) covers more ground with fewer, better-placed
    # trees, improving cost efficiency while maintaining dense corridor shade.
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

    # ── 2. Small street trees (13% budget, 3m spacing, gap-filling) ──────
    # Fill gaps between medium trees in corridors — 3x more trees per dollar
    # than medium. Gap-fill avoids redundant coverage.
    sml_budget_target = budget_usd * FRAC_SML
    remaining_for_sml = budget_usd - spent
    sml_budget = min(sml_budget_target, remaining_for_sml)

    tr_s = tc_s = np.array([], dtype=int)
    if sml_budget >= ctx.unit_cost("tree_small"):
        n_sml_max = ctx.affordable("tree_small", sml_budget)
        # Only plant small trees where not already covered by medium tree crowns
        # or existing canopy
        cand_sml = ctx.plantable & ~used & ~covered

        spacing_sml_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)
        n_sml = min(n_sml_max, int(cand_sml.sum()))

        tr_s, tc_s = _greedy_spaced(tree_score, cand_sml, spacing_sml_px, n_sml)

        if tr_s.size:
            placements.append(Placement("tree_small", tr_s, tc_s))
            spent += ctx.cost("tree_small", tr_s.size)
            used[tr_s, tc_s] = True
            crown_sml_px = max(int(round(CROWN_SML_M / ctx.res_m)), 1)
            _stamp_exclusion(covered, tr_s, tc_s, crown_sml_px, ctx.shape)

    # ── 3. Shade canopies (remaining ~25% budget, hottest open buildable) ─
    # Shade canopies provide strong UTCI relief on remaining hot exposed ground
    # not already shaded by trees. Use canopy score with stronger vulnerability
    # weighting to maximise equity_ratio.
    remaining = budget_usd - spent
    if remaining < ctx.unit_cost("shade_canopy"):
        return placements

    # Open buildable ground: not under existing/new canopy, not used
    open_ground = ctx.buildable & ~covered & ~used

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
    if cr.size:
        actual_cost = ctx.cost("shade_canopy", cr.size)
        if spent + actual_cost <= budget_usd + 0.01:
            placements.append(Placement("shade_canopy", cr, cc))
        else:
            # Trim to fit budget exactly
            affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
            if affordable_n > 0:
                placements.append(
                    Placement("shade_canopy", cr[:affordable_n], cc[:affordable_n])
                )

    return placements