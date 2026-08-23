from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-corridor tight-grid v1"
DESCRIPTION = (
    "Maximise population-weighted UTCI relief (heat_relief_c) via heat×population "
    "geometric mean synergy scoring with tighter spatial targeting. "
    "Phase 1: Medium trees at 4m spacing (62% budget) on highest heat×pop synergy "
    "corridors — tighter than parent for denser shade. Phase 2: Small trees at 3m "
    "spacing (13% budget) fill remaining plantable gaps. Phase 3: Shade canopies "
    "consume all remaining budget on hottest open pedestrian ground not under tree "
    "crowns. Priority surface: 0.40 heat×pop synergy + 0.25 heat_ta3pm + "
    "0.20 population + 0.10 heat_hours + 0.05 vulnerability. No reflective "
    "surfaces (albedo trap). Budget tracking is precise to avoid overspend."
)

# Budget allocation
MEDIUM_BUDGET_FRAC = 0.62
SMALL_BUDGET_FRAC  = 0.13
# Remainder (~25%) goes to shade canopies

# Tree spacing
MEDIUM_SPACING_M = 4.0   # tighter than parent's 5m → denser shade corridors
SMALL_SPACING_M  = 3.0   # tighter than parent's 4m → better gap-fill

# Crown radii for canopy exclusion
CROWN_MED_M   = 3.5
CROWN_SMALL_M = 2.0

# Priority boost for top-quartile vulnerability tracts
PRIORITY_BOOST = 0.08   # modest — fitness > equity in this cell


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1] over masked pixels; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def _priority_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Composite priority surface using heat×population geometric mean synergy.
    Targets pixels that are BOTH hot AND populated for maximum person-degC relief.
    Scores only where ctx.exposure is True.
    """
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,  mask)
    pop_n   = _norm(ctx.population,  mask)
    hours_n = _norm(ctx.heat_hours,  mask)
    vuln_n  = _norm(ctx.vulnerability, mask)

    # Geometric mean synergy: rewards pixels that are simultaneously hot + populated
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.40 * synergy
        + 0.25 * heat_n
        + 0.20 * pop_n
        + 0.10 * hours_n
        + 0.05 * vuln_n
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _canopy_surface(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for shade canopy placement.
    Emphasizes heat×pop synergy for maximum person-degC impact.
    """
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,  mask)
    pop_n   = _norm(ctx.population,  mask)
    uhii_n  = _norm(ctx.heat_uhii,   mask)
    hours_n = _norm(ctx.heat_hours,  mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.50 * synergy
        + 0.20 * heat_n
        + 0.15 * pop_n
        + 0.10 * uhii_n
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
    Greedy spaced placement: pick highest-scoring candidate,
    suppress spacing_px neighbourhood, repeat until limit or exhausted.
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


def _safe_trim(
    ctx: PlanningContext,
    action: str,
    rows: np.ndarray,
    cols: np.ndarray,
    budget_remaining: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Trim rows/cols so the placement fits within budget_remaining."""
    if rows.size == 0:
        return rows, cols
    n = ctx.affordable(action, budget_remaining)
    if n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    return rows[:n], cols[:n]


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    tree_score   = _priority_surface(ctx)
    canopy_score = _canopy_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0
    used  = np.zeros(ctx.shape, dtype=bool)

    # ── Phase 1: Medium street trees (62% budget, 4m spacing) ────────────
    # Tight 4m spacing creates dense shade corridors on hot+populated streets.
    med_budget    = budget_usd * MEDIUM_BUDGET_FRAC
    spacing_med   = max(int(round(MEDIUM_SPACING_M / ctx.res_m)), 1)
    n_med_max     = ctx.affordable("tree_medium", med_budget)
    cand_med      = ctx.plantable & ~used
    n_med         = min(n_med_max, int(cand_med.sum()))

    mr, mc = _greedy_spaced(tree_score, cand_med, spacing_med, n_med)
    if mr.size:
        mr, mc = _safe_trim(ctx, "tree_medium", mr, mc, budget_usd - spent)
    if mr.size:
        placements.append(Placement("tree_medium", mr, mc))
        spent += ctx.cost("tree_medium", mr.size)
        used[mr, mc] = True

    # ── Phase 2: Small street trees (13% budget, 3m spacing, gap-fill) ───
    # Dense 3m spacing fills gaps between medium trees cheaply.
    small_budget  = min(budget_usd * SMALL_BUDGET_FRAC, budget_usd - spent)
    sr = sc = np.array([], dtype=int)
    if small_budget >= ctx.unit_cost("tree_small"):
        spacing_small = max(int(round(SMALL_SPACING_M / ctx.res_m)), 1)
        n_small_max   = ctx.affordable("tree_small", small_budget)
        cand_small    = ctx.plantable & ~used
        n_small       = min(n_small_max, int(cand_small.sum()))

        sr, sc = _greedy_spaced(tree_score, cand_small, spacing_small, n_small)
        if sr.size:
            sr, sc = _safe_trim(ctx, "tree_small", sr, sc, budget_usd - spent)
        if sr.size:
            placements.append(Placement("tree_small", sr, sc))
            spent += ctx.cost("tree_small", sr.size)
            used[sr, sc] = True

    # ── Phase 3: Shade canopies (remaining budget, open hot pedestrian ground)
    # Exclude pixels already under tree crowns or existing canopy to avoid waste.
    remaining = budget_usd - spent
    if remaining < ctx.unit_cost("shade_canopy"):
        return placements

    # Build crown exclusion mask
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True  # existing canopy
    if mr.size:
        crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
        _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)
    if sr.size:
        crown_small_px = max(int(round(CROWN_SMALL_M / ctx.res_m)), 1)
        _stamp_crown(covered, sr, sc, crown_small_px, ctx.shape)

    # Open buildable ground: not covered, not already used
    open_ground  = ctx.buildable & ~covered & ~used
    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
    if cr.size:
        cr, cc = _safe_trim(ctx, "shade_canopy", cr, cc, budget_usd - spent)
    if cr.size:
        placements.append(Placement("shade_canopy", cr, cc))
        spent += ctx.cost("shade_canopy", cr.size)
        used[cr, cc] = True

    # ── Phase 4: Additional medium trees with leftover budget ─────────────
    # If there's still budget, plant more medium trees where possible.
    remaining2 = budget_usd - spent
    if remaining2 >= ctx.unit_cost("tree_medium"):
        cand_extra   = ctx.plantable & ~used
        n_extra_max  = ctx.affordable("tree_medium", remaining2)
        n_extra      = min(n_extra_max, int(cand_extra.sum()))
        if n_extra > 0:
            # Use denser spacing for leftovers (no spacing constraint = 1px)
            er2, ec2 = _greedy_spaced(tree_score, cand_extra, 1, n_extra)
            if er2.size:
                er2, ec2 = _safe_trim(ctx, "tree_medium", er2, ec2, budget_usd - spent)
            if er2.size:
                placements.append(Placement("tree_medium", er2, ec2))
                spent += ctx.cost("tree_medium", er2.size)
                used[er2, ec2] = True

    # ── Phase 5: Final small trees with any remaining scraps ──────────────
    remaining3 = budget_usd - spent
    if remaining3 >= ctx.unit_cost("tree_small"):
        cand_final  = ctx.plantable & ~used
        n_final_max = ctx.affordable("tree_small", remaining3)
        n_final     = min(n_final_max, int(cand_final.sum()))
        if n_final > 0:
            fr, fc = _greedy_spaced(tree_score, cand_final, 1, n_final)
            if fr.size:
                fr, fc = _safe_trim(ctx, "tree_small", fr, fc, budget_usd - spent)
            if fr.size:
                placements.append(Placement("tree_small", fr, fc))

    return placements