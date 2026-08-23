from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "synergy-equity corridors v2"
DESCRIPTION = (
    "Maximise heat_relief_c in cell (1,1,0,1) via synergy-based priority scoring "
    "and tight tree spacing. Phase 1: Medium trees at 3m spacing (60% budget) on "
    "highest heat×population synergy corridors for dense canopy shade. Phase 2: "
    "Small trees at 2m spacing gap-filling (18% budget). Phase 3: Shade canopies "
    "on remaining hot open buildable ground (15% budget). Phase 4: Grass conversion "
    "on hot paved non-roadbed areas (7% budget) to maintain cobenefit_greened_pct. "
    "Priority surface: synergy(heat×pop) 0.50, heat_ta3pm 0.20, population 0.12, "
    "vulnerability 0.10, heat_hours 0.05, UHII 0.03. Priority-tract boost 0.30 "
    "for equity_ratio > 1. Reflective surfaces avoided (albedo trap)."
)

FRAC_MED    = 0.60
FRAC_SML    = 0.18
FRAC_CANOPY = 0.15
FRAC_GRASS  = 0.07

SPACING_MED_M = 3.0
SPACING_SML_M = 2.0

CROWN_MED_M = 3.5
CROWN_SML_M = 2.0

PRIORITY_BOOST = 0.30


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1] over mask; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def _tree_priority(ctx: PlanningContext) -> np.ndarray:
    """
    Synergy-based composite priority surface.
    Geometric mean of heat × population targets pixels that are BOTH hot
    AND populated for maximum person-degC relief per tree placed.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    # Geometric mean synergy: high only if BOTH heat AND population are high
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


def _canopy_priority(ctx: PlanningContext) -> np.ndarray:
    """
    Canopy-specific priority surface: stronger synergy weight since canopies
    provide immediate shade over high-density pedestrian areas.
    """
    mask = ctx.exposure

    heat_n  = _norm(ctx.heat_ta3pm,    mask)
    pop_n   = _norm(ctx.population,    mask)
    vuln_n  = _norm(ctx.vulnerability, mask)
    hours_n = _norm(ctx.heat_hours,    mask)
    uhii_n  = _norm(ctx.heat_uhii,     mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.55 * synergy
        + 0.20 * heat_n
        + 0.12 * pop_n
        + 0.08 * vuln_n
        + 0.05 * hours_n
    )
    score = score + np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _grass_priority(ctx: PlanningContext) -> np.ndarray:
    """
    Grass conversion priority: target hot paved non-roadbed areas with
    population presence for cooling + greening cobenefit.
    """
    mask = ctx.walkable

    heat_n = _norm(ctx.heat_ta3pm, mask)
    pop_n  = _norm(ctx.population, mask)
    uhii_n = _norm(ctx.heat_uhii,  mask)
    vuln_n = _norm(ctx.vulnerability, mask)

    score = (
          0.45 * heat_n
        + 0.25 * pop_n
        + 0.20 * uhii_n
        + 0.10 * vuln_n
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
    Greedy spaced placement: pick highest-scoring candidate, suppress
    a spacing_px-radius neighbourhood, repeat until limit reached.
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


def _safe_afford(
    ctx: PlanningContext,
    action: str,
    rows: np.ndarray,
    cols: np.ndarray,
    spent: float,
    budget_usd: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Trim arrays to fit remaining budget and return updated spend."""
    if rows.size == 0:
        return rows, cols, spent
    remaining = budget_usd - spent
    n_affordable = ctx.affordable(action, remaining)
    if n_affordable <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), spent
    rows = rows[:n_affordable]
    cols = cols[:n_affordable]
    spent += ctx.cost(action, rows.size)
    return rows, cols, spent


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    tree_score   = _tree_priority(ctx)
    canopy_score = _canopy_priority(ctx)
    grass_score  = _grass_priority(ctx)

    placements: list[Placement] = []
    spent = 0.0
    used    = np.zeros(ctx.shape, dtype=bool)
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # pre-existing canopy

    # ── Phase 1: Medium street trees (60% budget, 3m spacing) ───────────
    # Dense synergy-driven corridors — the main driver of heat_relief_c
    med_budget    = budget_usd * FRAC_MED
    n_med_max     = ctx.affordable("tree_medium", med_budget)
    cand_med      = ctx.plantable & ~used
    n_med         = min(n_med_max, int(cand_med.sum()))

    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)
    mr, mc = _greedy_spaced(tree_score, cand_med, spacing_med_px, n_med)

    if mr.size:
        mr, mc, spent = _safe_afford(ctx, "tree_medium", mr, mc, spent, budget_usd)
        if mr.size:
            placements.append(Placement("tree_medium", mr, mc))
            used[mr, mc] = True
            crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
            _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)

    # ── Phase 2: Small street trees (18% budget, 2m spacing, gap-fill) ──
    # Fill canopy gaps between medium trees for continuous shade coverage
    sml_budget = min(budget_usd * FRAC_SML, budget_usd - spent)
    sr = sc = np.array([], dtype=int)

    if sml_budget >= ctx.unit_cost("tree_small"):
        n_sml_max  = ctx.affordable("tree_small", sml_budget)
        # Exclude pixels already under medium tree crowns
        cand_sml   = ctx.plantable & ~used & ~covered
        n_sml      = min(n_sml_max, int(cand_sml.sum()))

        spacing_sml_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)
        sr, sc = _greedy_spaced(tree_score, cand_sml, spacing_sml_px, n_sml)

        if sr.size:
            sr, sc, spent = _safe_afford(ctx, "tree_small", sr, sc, spent, budget_usd)
            if sr.size:
                placements.append(Placement("tree_small", sr, sc))
                used[sr, sc] = True
                crown_sml_px = max(int(round(CROWN_SML_M / ctx.res_m)), 1)
                _stamp_crown(covered, sr, sc, crown_sml_px, ctx.shape)

    # ── Phase 3: Shade canopies (15% budget, hottest open buildable) ─────
    # Immediate shade on remaining open ground not covered by trees
    canopy_budget = min(budget_usd * FRAC_CANOPY, budget_usd - spent)

    if canopy_budget >= ctx.unit_cost("shade_canopy"):
        open_ground  = ctx.buildable & ~covered & ~used
        n_canopy_max = ctx.affordable("shade_canopy", canopy_budget)
        n_canopy     = min(n_canopy_max, int(open_ground.sum()))

        cr, cc = _top_pixels(canopy_score, open_ground, n_canopy)
        if cr.size:
            cr, cc, spent = _safe_afford(ctx, "shade_canopy", cr, cc, spent, budget_usd)
            if cr.size:
                placements.append(Placement("shade_canopy", cr, cc))
                used[cr, cc] = True
                # Mark canopy footprints as covered too
                covered[cr, cc] = True

    # ── Phase 4: Grass conversion (7% budget, boost cobenefit_greened_pct)
    # Depave hot paved areas to green surface for cooling + greening
    grass_budget = min(budget_usd * FRAC_GRASS, budget_usd - spent)

    if grass_budget >= ctx.unit_cost("grass_conversion"):
        cand_grass  = ctx.placeable("grass_conversion") & ~ctx.roadbed & ~used
        n_grass_max = ctx.affordable("grass_conversion", grass_budget)
        n_grass     = min(n_grass_max, int(cand_grass.sum()))

        gr, gc = _top_pixels(grass_score, cand_grass, n_grass)
        if gr.size:
            gr, gc, spent = _safe_afford(ctx, "grass_conversion", gr, gc, spent, budget_usd)
            if gr.size:
                placements.append(Placement("grass_conversion", gr, gc))
                used[gr, gc] = True

    # ── Phase 5: Remaining budget → additional shade canopies ────────────
    # Absorb any leftover budget into more canopy shade
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("shade_canopy"):
        open_ground2 = ctx.buildable & ~covered & ~used
        n_extra      = ctx.affordable("shade_canopy", remaining)
        n_extra      = min(n_extra, int(open_ground2.sum()))

        if n_extra > 0:
            er, ec = _top_pixels(canopy_score, open_ground2, n_extra)
            if er.size:
                er, ec, spent = _safe_afford(ctx, "shade_canopy", er, ec, spent, budget_usd)
                if er.size:
                    placements.append(Placement("shade_canopy", er, ec))
                    used[er, ec] = True

    # ── Phase 6: Any final remainder → more small trees ──────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("tree_small"):
        cand_extra = ctx.plantable & ~used
        n_extra2   = ctx.affordable("tree_small", remaining)
        n_extra2   = min(n_extra2, int(cand_extra.sum()))

        if n_extra2 > 0:
            spacing_px = max(int(round(2.0 / ctx.res_m)), 1)
            xr, xc = _greedy_spaced(tree_score, cand_extra, spacing_px, n_extra2)
            if xr.size:
                xr, xc, spent = _safe_afford(ctx, "tree_small", xr, xc, spent, budget_usd)
                if xr.size:
                    placements.append(Placement("tree_small", xr, xc))

    return placements