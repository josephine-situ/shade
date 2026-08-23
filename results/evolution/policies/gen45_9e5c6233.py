from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "max-canopy-density synergy v3"
DESCRIPTION = (
    "Maximise heat_relief_c via maximum-density medium tree corridors on a "
    "heat×population synergy surface (75% budget at 2 m minimum spacing), "
    "followed by shade canopies on hottest open ground (25% budget). "
    "Priority-tract boost 0.35 preserves equity_ratio > 1. "
    "Zero albedo interventions (avoids UTCI penalty from reflected shortwave)."
)

PRIORITY_BOOST = 0.35
FRAC_MED = 0.75
FRAC_CANOPY = 0.25
SPACING_MED_M = 2.0


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def priority_surface(ctx: PlanningContext) -> np.ndarray:
    mask = ctx.exposure
    heat_n = _norm(ctx.heat_ta3pm, mask)
    pop_n = _norm(ctx.population, mask)
    vuln_n = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours, mask)
    uhii_n = _norm(ctx.heat_uhii, mask)
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))
    score = (
        0.50 * synergy
        + 0.20 * heat_n
        + 0.15 * pop_n
        + 0.10 * vuln_n
        + 0.03 * hours_n
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
    rows, cols = np.nonzero(candidates)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    order = np.argsort(-score[rows, cols])[:limit]
    return rows[order], cols[order]


def _stamp_box(arr: np.ndarray, rows: np.ndarray, cols: np.ndarray,
               radius: int, H: int, W: int) -> None:
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        r0 = max(0, r - radius)
        r1 = min(H, r + radius + 1)
        c0 = max(0, c - radius)
        c1 = min(W, c + radius + 1)
        arr[r0:r1, c0:c1] = True


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score = priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0
    used = np.zeros(ctx.shape, dtype=bool)
    H, W = ctx.shape

    # ── Medium trees: 75% budget, minimum 2m spacing ──────────────────
    med_budget = budget_usd * FRAC_MED
    n_med_max = ctx.affordable("tree_medium", med_budget)
    spacing_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    cand_med = ctx.plantable & ~used

    tr, tc = _greedy_spaced(score, cand_med, spacing_px, n_med_max)
    if tr.size:
        placements.append(Placement("tree_medium", tr, tc))
        spent += ctx.cost("tree_medium", tr.size)
        used[tr, tc] = True

    # ── Shade canopies: remaining budget ───────────────────────────────
    remaining = budget_usd - spent
    if remaining <= 0.0:
        return placements

    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True
    crown_px = max(int(round(3.5 / ctx.res_m)), 1)
    if tr.size:
        _stamp_box(covered, tr, tc, crown_px, H, W)

    open_ground = ctx.buildable & ~covered & ~used
    n_can = ctx.affordable("shade_canopy", remaining)
    n_can = min(n_can, int(open_ground.sum()))
    cr, cc = _top_pixels(score, open_ground, n_can)
    if cr.size:
        actual = ctx.cost("shade_canopy", cr.size)
        if spent + actual <= budget_usd + 0.01:
            placements.append(Placement("shade_canopy", cr, cc))
        else:
            n_fit = ctx.affordable("shade_canopy", budget_usd - spent)
            if n_fit > 0:
                placements.append(Placement("shade_canopy", cr[:n_fit], cc[:n_fit]))

    return placements