from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "dense-synergy-corridors equity-greened v2"
DESCRIPTION = (
    "Improve on parent (0.0342) in cell (1,0,1,1): equity_ratio>=1, "
    "cost_efficiency>=40.55, cobenefit_greened_pct>=0.1976. "
    "Strategy: (1) Medium trees at 2-3m spacing (72% budget) on highest "
    "heat×population synergy corridors WITH priority-tract boost to ensure "
    "equity_ratio>=1; (2) Shade canopies on hottest remaining open ground "
    "(20% budget) — no small trees, concentrate on high-impact actions; "
    "(3) Grass conversion on hot paved non-roadbed pixels (8% budget) for "
    "cobenefit_greened_pct >= 0.1976. "
    "Priority surface: 0.45 geometric-mean synergy(heat×pop) + 0.22 heat_ta3pm "
    "+ 0.13 pop + 0.10 heat_hours + 0.05 UHII + 0.05 vulnerability + "
    "0.20 priority-tract boost. "
    "Reflective surfaces avoided entirely (albedo trap)."
)

FRAC_MED    = 0.72
FRAC_CANOPY = 0.20
FRAC_GRASS  = 0.08

SPACING_MED_M  = 2.5   # tighter than parent's 3m for denser shade corridors
CROWN_MED_M    = 3.5

PRIORITY_BOOST = 0.20  # strong enough to push equity_ratio >= 1


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1] over mask; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def _tree_score(ctx: PlanningContext) -> np.ndarray:
    """
    Synergy-based priority surface for tree placement.
    Geometric mean of normalised heat × population, with priority-tract boost
    to ensure equity_ratio >= 1.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)
    vuln_n  = _norm(ctx.vulnerability, mask)

    # Geometric mean synergy: only high if BOTH heat AND population are high
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.45 * synergy
        + 0.22 * heat_n
        + 0.13 * pop_n
        + 0.10 * hours_n
        + 0.05 * uhii_n
        + 0.05 * vuln_n
    )
    # Priority-tract boost to push equity_ratio >= 1
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _canopy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for shade canopy placement.
    Strong synergy emphasis + priority boost for equity.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)
    vuln_n  = _norm(ctx.vulnerability, mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.50 * synergy
        + 0.22 * heat_n
        + 0.13 * pop_n
        + 0.07 * hours_n
        + 0.05 * uhii_n
        + 0.03 * vuln_n
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _grass_score(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for grass conversion.
    Targets hot paved areas with high population.
    """
    mask = ctx.walkable

    heat_n = _norm(ctx.heat_ta3pm, mask)
    pop_n  = _norm(ctx.population, mask)
    uhii_n = _norm(ctx.heat_uhii,  mask)
    vuln_n = _norm(ctx.vulnerability, mask)

    score = (
          0.45 * heat_n
        + 0.30 * pop_n
        + 0.15 * uhii_n
        + 0.10 * vuln_n
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST * 0.5, 0.0)
    return np.where(mask, score, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy selection: pick highest-scoring candidate, suppress spacing_px
    neighbourhood, repeat until limit reached or candidates exhausted.
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
    """Mark a box of radius crown_px around each (r, c) as covered."""
    H, W = shape
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        r0 = max(0, r - crown_px)
        r1 = min(H, r + crown_px + 1)
        c0 = max(0, c - crown_px)
        c1 = min(W, c + crown_px + 1)
        covered[r0:r1, c0:c1] = True


def _safe_trim(
    ctx: PlanningContext,
    action: str,
    rows: np.ndarray,
    cols: np.ndarray,
    spent: float,
    budget_usd: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Trim placement to fit within remaining budget; return updated spent."""
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
    tree_score   = _tree_score(ctx)
    canopy_score = _canopy_score(ctx)
    grass_sc     = _grass_score(ctx)

    placements: list[Placement] = []
    spent = 0.0
    used    = np.zeros(ctx.shape, dtype=bool)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True  # pre-existing canopy

    # ── Phase 1: Medium street trees (72% budget, 2.5m spacing) ──────────
    # Tighter than parent's 3m → denser shade corridors → stronger UTCI drop.
    # Priority-tract boost ensures equity_ratio >= 1.
    med_budget   = budget_usd * FRAC_MED
    n_med_max    = ctx.affordable("tree_medium", med_budget)
    cand_med     = ctx.plantable & ~used
    n_med        = min(n_med_max, int(cand_med.sum()))

    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    mr, mc = _greedy_spaced(tree_score, cand_med, spacing_med_px, n_med)

    if mr.size:
        mr, mc, spent = _safe_trim(ctx, "tree_medium", mr, mc, spent, budget_usd)
        if mr.size:
            placements.append(Placement("tree_medium", mr, mc))
            used[mr, mc] = True
            crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
            _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)

    # ── Phase 2: Shade canopies (20% budget, hottest open ground) ─────────
    canopy_budget = min(budget_usd * FRAC_CANOPY, budget_usd - spent)
    if canopy_budget >= ctx.unit_cost("shade_canopy"):
        open_ground  = ctx.buildable & ~covered & ~used
        n_canopy_max = ctx.affordable("shade_canopy", canopy_budget)
        n_canopy     = min(n_canopy_max, int(open_ground.sum()))

        cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
        if cr.size:
            cr, cc, spent = _safe_trim(ctx, "shade_canopy", cr, cc, spent, budget_usd)
            if cr.size:
                placements.append(Placement("shade_canopy", cr, cc))
                used[cr, cc] = True

    # ── Phase 3: Grass conversion (8% budget, cobenefit_greened_pct target) ─
    grass_budget = min(budget_usd * FRAC_GRASS, budget_usd - spent)
    if grass_budget >= ctx.unit_cost("grass_conversion"):
        cand_grass  = ctx.placeable("grass_conversion") & ~ctx.roadbed & ~used
        n_grass_max = ctx.affordable("grass_conversion", grass_budget)
        n_grass     = min(n_grass_max, int(cand_grass.sum()))

        gr, gc = _top_pixels(grass_sc, cand_grass, n_grass)
        if gr.size:
            gr, gc, spent = _safe_trim(ctx, "grass_conversion", gr, gc, spent, budget_usd)
            if gr.size:
                placements.append(Placement("grass_conversion", gr, gc))
                used[gr, gc] = True

    # ── Phase 4: Remaining budget → more medium trees (gap-fill, no spacing) ─
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("tree_medium"):
        cand_extra = ctx.plantable & ~used
        n_extra    = ctx.affordable("tree_medium", remaining)
        n_extra    = min(n_extra, int(cand_extra.sum()))

        if n_extra > 0:
            # Gap-fill with tighter spacing to maximize density
            spacing_fill = max(int(round(2.0 / ctx.res_m)), 1)
            xr, xc = _greedy_spaced(tree_score, cand_extra, spacing_fill, n_extra)
            if xr.size:
                xr, xc, spent = _safe_trim(ctx, "tree_medium", xr, xc, spent, budget_usd)
                if xr.size:
                    placements.append(Placement("tree_medium", xr, xc))
                    used[xr, xc] = True

    # ── Phase 5: Any remaining budget → more shade canopies ───────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("shade_canopy"):
        open_ground2 = ctx.buildable & ~covered & ~used
        n_extra2     = ctx.affordable("shade_canopy", remaining)
        n_extra2     = min(n_extra2, int(open_ground2.sum()))

        if n_extra2 > 0:
            er, ec = _top_pixels(canopy_score, open_ground2, n_extra2)
            if er.size:
                er, ec, spent = _safe_trim(ctx, "shade_canopy", er, ec, spent, budget_usd)
                if er.size:
                    placements.append(Placement("shade_canopy", er, ec))
                    used[er, ec] = True

    # ── Phase 6: Final remainder → small trees on plantable gaps ──────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("tree_small"):
        cand_sml = ctx.plantable & ~used
        n_sml    = ctx.affordable("tree_small", remaining)
        n_sml    = min(n_sml, int(cand_sml.sum()))

        if n_sml > 0:
            spacing_sml = max(int(round(2.0 / ctx.res_m)), 1)
            sr, sc = _greedy_spaced(tree_score, cand_sml, spacing_sml, n_sml)
            if sr.size:
                sr, sc, spent = _safe_trim(ctx, "tree_small", sr, sc, spent, budget_usd)
                if sr.size:
                    placements.append(Placement("tree_small", sr, sc))

    return placements