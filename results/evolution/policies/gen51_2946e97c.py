from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-quad-shade corridors v2"
DESCRIPTION = (
    "Maximise heat_relief_c via optimised four-phase shade deployment. "
    "(1) Medium trees at 4m spacing on highest heat×population synergy "
    "corridors (70% budget) — optimal spacing for corridor coverage without "
    "excessive crown overlap; (2) Small trees filling crown gaps at 2m "
    "spacing (15% budget); (3) Shade canopies on remaining hot open "
    "buildable ground (10% budget); (4) Any remaining budget → more medium "
    "trees then small trees for maximum UTCI impact. "
    "Priority surface: geometric mean synergy(heat×pop) 0.50, heat_ta3pm "
    "0.20, population 0.12, vulnerability 0.10, heat_hours 0.05, UHII 0.03. "
    "Priority-tract boost 0.35 ensures strong equity_ratio. "
    "Reflective surfaces avoided entirely (albedo trap)."
)

PRIORITY_BOOST = 0.35

FRAC_MED    = 0.70
FRAC_SML    = 0.15
FRAC_CANOPY = 0.10
# Remaining 5% flows to phase 4 (more trees)

SPACING_MED_M = 4.0   # wider than parent's 3m → better corridor coverage
SPACING_SML_M = 2.0

CROWN_MED_M = 4.0     # slightly larger crown estimate for exclusion
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
    Synergy-based composite priority surface for tree/canopy placement.
    Geometric mean of heat × population rewards pixels that are simultaneously
    hot AND populated — these yield maximum person-degC relief per intervention.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    # Synergy: geometric mean rewards co-occurrence of heat AND population
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.50 * synergy
        + 0.20 * heat_n
        + 0.12 * pop_n
        + 0.10 * vuln_n
        + 0.05 * hours_n
        + 0.03 * uhii_n
    )

    # Strong boost for top-quartile vulnerability tracts
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


def _stamp_crown(
    covered: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    crown_px: int,
    shape: tuple[int, int],
) -> None:
    """Mark a box of crown_px radius around each (r, c) as covered."""
    H, W = shape
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        r0 = max(0, r - crown_px)
        r1 = min(H, r + crown_px + 1)
        c0 = max(0, c - crown_px)
        c1 = min(W, c + crown_px + 1)
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
    remaining = budget_usd - spent
    affordable_n = ctx.affordable(action, remaining)
    if affordable_n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), spent
    rows = rows[:affordable_n]
    cols = cols[:affordable_n]
    spent += ctx.cost(action, rows.size)
    return rows, cols, spent


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score = priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0

    # Track which pixels are used (no double-booking)
    used    = np.zeros(ctx.shape, dtype=bool)
    # Track canopy coverage (existing + new plantings) for canopy phase
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # pre-existing canopy

    # Pixel dimensions
    crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
    crown_sml_px = max(int(round(CROWN_SML_M / ctx.res_m)), 1)
    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    spacing_sml_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)

    # ── Phase 1: Medium street trees (70% budget, 4m spacing) ──────────────
    # Wider 4m spacing vs parent's 3m: avoids excessive crown overlap,
    # spreads interventions across more corridors for greater population reach.
    med_budget = budget_usd * FRAC_MED
    n_med_max  = ctx.affordable("tree_medium", med_budget)
    cand_med   = ctx.plantable & ~used
    n_med      = min(n_med_max, int(cand_med.sum()))

    mr, mc = _greedy_spaced(score, cand_med, spacing_med_px, n_med)
    if mr.size:
        mr, mc, spent = _safe_place(ctx, "tree_medium", mr, mc, spent, budget_usd)
        if mr.size:
            placements.append(Placement("tree_medium", mr, mc))
            used[mr, mc] = True
            _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)

    # ── Phase 2: Small street trees (15% budget, 2m spacing, gap-fill) ─────
    # Fill canopy gaps between medium trees — tight spacing to cover
    # every plantable pixel not already shaded by medium trees.
    sml_budget = min(budget_usd * FRAC_SML, budget_usd - spent)
    sr = sc = np.array([], dtype=int)
    if sml_budget >= ctx.unit_cost("tree_small"):
        cand_sml = ctx.plantable & ~used & ~covered
        n_sml_max = ctx.affordable("tree_small", sml_budget)
        n_sml     = min(n_sml_max, int(cand_sml.sum()))

        sr, sc = _greedy_spaced(score, cand_sml, spacing_sml_px, n_sml)
        if sr.size:
            sr, sc, spent = _safe_place(ctx, "tree_small", sr, sc, spent, budget_usd)
            if sr.size:
                placements.append(Placement("tree_small", sr, sc))
                used[sr, sc] = True
                _stamp_crown(covered, sr, sc, crown_sml_px, ctx.shape)

    # ── Phase 3: Shade canopies (10% budget, hottest open buildable ground) ─
    # Target remaining hot pedestrian areas not covered by tree canopy.
    canopy_budget = min(budget_usd * FRAC_CANOPY, budget_usd - spent)
    cr = cc = np.array([], dtype=int)
    if canopy_budget >= ctx.unit_cost("shade_canopy"):
        open_ground  = ctx.buildable & ~covered & ~used
        n_canopy_max = ctx.affordable("shade_canopy", canopy_budget)
        n_canopy     = min(n_canopy_max, int(open_ground.sum()))

        cr, cc = _top_pixels(score, open_ground, n_canopy)
        if cr.size:
            cr, cc, spent = _safe_place(ctx, "shade_canopy", cr, cc, spent, budget_usd)
            if cr.size:
                placements.append(Placement("shade_canopy", cr, cc))
                used[cr, cc] = True
                # Canopy pixels themselves count as covered
                covered[cr, cc] = True

    # ── Phase 4: Remaining budget → more medium trees, then small trees ─────
    # Greedily use any leftover budget for maximum UTCI impact.
    remaining = budget_usd - spent

    # 4a: More medium trees if possible
    if remaining >= ctx.unit_cost("tree_medium"):
        cand_med2  = ctx.plantable & ~used
        n_med2_max = ctx.affordable("tree_medium", remaining)
        n_med2     = min(n_med2_max, int(cand_med2.sum()))

        if n_med2 > 0:
            mr2, mc2 = _greedy_spaced(score, cand_med2, spacing_med_px, n_med2)
            if mr2.size:
                mr2, mc2, spent = _safe_place(ctx, "tree_medium", mr2, mc2, spent, budget_usd)
                if mr2.size:
                    placements.append(Placement("tree_medium", mr2, mc2))
                    used[mr2, mc2] = True
                    _stamp_crown(covered, mr2, mc2, crown_med_px, ctx.shape)

    # 4b: More small trees if possible
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("tree_small"):
        cand_sml2  = ctx.plantable & ~used & ~covered
        n_sml2_max = ctx.affordable("tree_small", remaining)
        n_sml2     = min(n_sml2_max, int(cand_sml2.sum()))

        if n_sml2 > 0:
            sr2, sc2 = _greedy_spaced(score, cand_sml2, spacing_sml_px, n_sml2)
            if sr2.size:
                sr2, sc2, spent = _safe_place(ctx, "tree_small", sr2, sc2, spent, budget_usd)
                if sr2.size:
                    placements.append(Placement("tree_small", sr2, sc2))
                    used[sr2, sc2] = True
                    _stamp_crown(covered, sr2, sc2, crown_sml_px, ctx.shape)

    # 4c: Final remaining budget → shade canopies on any open hot ground
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("shade_canopy"):
        open_ground2 = ctx.buildable & ~covered & ~used
        n_extra      = ctx.affordable("shade_canopy", remaining)
        n_extra      = min(n_extra, int(open_ground2.sum()))

        if n_extra > 0:
            er, ec = _top_pixels(score, open_ground2, n_extra)
            if er.size:
                er, ec, spent = _safe_place(ctx, "shade_canopy", er, ec, spent, budget_usd)
                if er.size:
                    placements.append(Placement("shade_canopy", er, ec))

    return placements