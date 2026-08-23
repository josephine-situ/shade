from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "equity-synergy medium-tree canopy v1"
DESCRIPTION = (
    "Maximise heat_relief_c in equity-high cells: synergy priority surface "
    "(geometric mean heat×population) with strong vulnerability weighting "
    "for equity_ratio > 1. Budget: 60% medium trees at 4m spacing (strongest "
    "UTCI relief per tree), 15% small trees for gap-filling at 3m spacing, "
    "25% shade canopies on hottest open buildable ground. Priority-tract "
    "boost 0.35 targets top-vulnerability corridors. Reflective surfaces "
    "avoided entirely (albedo trap). Canopy score uses heat×population synergy "
    "to place shade where it reduces population-weighted UTCI most."
)

PRIORITY_BOOST = 0.35   # strong boost for top-quartile vulnerability tracts

FRAC_MED    = 0.60
FRAC_SML    = 0.15
FRAC_CANOPY = 0.25

SPACING_MED_M = 4.0
SPACING_SML_M = 3.0

CROWN_MED_M   = 3.5
CROWN_SML_M   = 2.0


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1] over masked region; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Synergy-based priority surface combining heat×population geometric mean
    with strong vulnerability weighting for equity.
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
          0.40 * synergy
        + 0.20 * heat_n
        + 0.20 * vuln_n
        + 0.10 * pop_n
        + 0.06 * hours_n
        + 0.04 * uhii_n
    )

    # Strong boost for priority tracts to ensure equity_ratio > 1
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def canopy_priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for shade canopy placement.
    Emphasises heat×population synergy and vulnerability.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.45 * synergy
        + 0.20 * heat_n
        + 0.18 * vuln_n
        + 0.10 * pop_n
        + 0.05 * hours_n
        + 0.02 * uhii_n
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
    Pick highest-scoring candidate, suppress spacing_px neighbourhood, repeat.
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
    tree_score   = priority_surface(ctx)
    canopy_score = canopy_priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0
    used  = np.zeros(ctx.shape, dtype=bool)

    # ── Phase 1: Medium street trees (60% budget, 4m spacing) ───────────
    # Medium trees provide the strongest per-tree UTCI benefit via shade.
    # 4m spacing balances density with realistic crown coverage.
    medium_budget = budget_usd * FRAC_MED
    n_medium_max  = ctx.affordable("tree_medium", medium_budget)
    cand_medium   = ctx.plantable & ~used
    n_medium      = min(n_medium_max, int(cand_medium.sum()))

    spacing_medium_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    mr, mc = _greedy_spaced(tree_score, cand_medium, spacing_medium_px, n_medium)

    if mr.size:
        placements.append(Placement("tree_medium", mr, mc))
        spent += ctx.cost("tree_medium", mr.size)
        used[mr, mc] = True

    # ── Phase 2: Small street trees (15% budget, 3m spacing, gap-fill) ──
    # Fill remaining high-priority plantable gaps cheaply.
    small_budget = min(budget_usd * FRAC_SML, budget_usd - spent)
    sr = sc = np.array([], dtype=int)

    if small_budget >= ctx.unit_cost("tree_small"):
        cand_small  = ctx.plantable & ~used
        n_small_max = ctx.affordable("tree_small", small_budget)
        n_small     = min(n_small_max, int(cand_small.sum()))

        spacing_small_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)
        sr, sc = _greedy_spaced(tree_score, cand_small, spacing_small_px, n_small)

        if sr.size:
            placements.append(Placement("tree_small", sr, sc))
            spent += ctx.cost("tree_small", sr.size)
            used[sr, sc] = True

    # ── Phase 3: Shade canopies (~25% budget) ───────────────────────────
    # Shade canopies cover hottest open pedestrian ground not under new trees.
    remaining = budget_usd - spent
    if remaining < ctx.unit_cost("shade_canopy"):
        return placements

    # Build coverage map: existing canopy + new tree crowns
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True  # pre-existing canopy

    if mr.size:
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)
    if sr.size:
        crown_sml_px = max(int(round(CROWN_SML_M / ctx.res_m)), 1)
        _stamp_crown(covered, sr, sc, crown_sml_px, ctx.shape)

    open_ground  = ctx.buildable & ~covered & ~used
    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
    if cr.size:
        actual_cost = ctx.cost("shade_canopy", cr.size)
        if spent + actual_cost <= budget_usd:
            placements.append(Placement("shade_canopy", cr, cc))
            spent += actual_cost
        else:
            affordable_n = ctx.affordable("shade_canopy", budget_usd - spent)
            if affordable_n > 0:
                placements.append(
                    Placement("shade_canopy", cr[:affordable_n], cc[:affordable_n])
                )

    return placements