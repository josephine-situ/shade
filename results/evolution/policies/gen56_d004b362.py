from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "dense-shade-synergy v2"
DESCRIPTION = (
    "Improved champion for cell (0,1,1,1). Maximises population-weighted UTCI "
    "relief by aggressive shade deployment. Phase 1: Medium trees at 3m spacing "
    "(55% budget) targeting heat×pop×heat_hours synergy corridors. "
    "Phase 2: Shade canopies on all remaining hot open buildable ground (25% budget) "
    "targeting highest synergy open areas. Phase 3: Grass conversion on hot paved "
    "non-roadbed pixels (10% budget) for cobenefit_greened_pct. "
    "Phase 4: Small trees to gap-fill remaining plantable spots (5% budget). "
    "Phase 5: Any remainder to more shade canopies. "
    "Priority surface: 0.40 heat×pop synergy + 0.30 heat_ta3pm + 0.15 population "
    "+ 0.10 heat_hours + 0.05 vulnerability. No priority boost to keep equity_ratio<1 "
    "and focus purely on heat+population targeting. Reflective surfaces avoided."
)

# Budget fractions
FRAC_MEDIUM = 0.55
FRAC_CANOPY = 0.25
FRAC_GRASS  = 0.10
FRAC_SMALL  = 0.05
# Remainder goes to more canopies

# Tighter tree spacing for denser canopy cover
MEDIUM_SPACING_M = 3.0
SMALL_SPACING_M  = 2.0

# Crown radii for exclusion zones
CROWN_MED_M   = 4.0
CROWN_SMALL_M = 2.0

# No priority boost - keep equity_ratio < 1 for target cell (0,1,1,1)
PRIORITY_BOOST = 0.0


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1] over masked pixels; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def _synergy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Composite priority surface maximising person-degC UTCI relief.
    Heat×population geometric-mean synergy is the dominant term.
    No priority boost to avoid equity_ratio >= 1.
    """
    mask = ctx.exposure
    heat_n   = _norm(ctx.heat_ta3pm,    mask)
    pop_n    = _norm(ctx.population,    mask)
    hours_n  = _norm(ctx.heat_hours,    mask)
    vuln_n   = _norm(ctx.vulnerability, mask)
    uhii_n   = _norm(ctx.heat_uhii,     mask)

    # Geometric mean synergy: high only when BOTH heat and population are high
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    # Triple synergy: heat × population × heat_hours
    triple_synergy = np.cbrt(np.clip(heat_n * pop_n * hours_n, 0.0, None))

    score = (
          0.35 * synergy
        + 0.20 * triple_synergy
        + 0.25 * heat_n
        + 0.12 * pop_n
        + 0.05 * hours_n
        + 0.03 * uhii_n
    )
    return np.where(mask, score, -np.inf)


def _canopy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Priority surface for shade canopy placement.
    Heavy weight on heat×population synergy for maximum person-degC relief.
    """
    mask = ctx.exposure
    heat_n   = _norm(ctx.heat_ta3pm,    mask)
    pop_n    = _norm(ctx.population,    mask)
    hours_n  = _norm(ctx.heat_hours,    mask)
    uhii_n   = _norm(ctx.heat_uhii,     mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.45 * synergy
        + 0.30 * heat_n
        + 0.15 * pop_n
        + 0.10 * hours_n
    )
    return np.where(mask, score, -np.inf)


def _grass_score(ctx: PlanningContext) -> np.ndarray:
    """Score for grass conversion: target hot paved non-roadbed areas."""
    mask = ctx.walkable
    heat_n  = _norm(ctx.heat_ta3pm,   mask)
    pop_n   = _norm(ctx.population,   mask)
    uhii_n  = _norm(ctx.heat_uhii,    mask)
    hours_n = _norm(ctx.heat_hours,   mask)

    score = (
          0.45 * heat_n
        + 0.25 * pop_n
        + 0.20 * uhii_n
        + 0.10 * hours_n
    )
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
    if remaining <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), spent
    affordable_n = ctx.affordable(action, remaining)
    if affordable_n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), spent
    rows = rows[:affordable_n]
    cols = cols[:affordable_n]
    spent += ctx.cost(action, rows.size)
    return rows, cols, spent


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score      = _synergy_score(ctx)
    can_score  = _canopy_score(ctx)
    grass_sc   = _grass_score(ctx)
    placements: list[Placement] = []
    spent = 0.0
    used  = np.zeros(ctx.shape, dtype=bool)

    # Track canopy/crown coverage for shade canopy placement exclusion
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[ctx.cdsm > 0.0] = True   # existing canopy

    # ── Phase 1: Medium street trees (55% budget, 3m spacing) ─────────────
    medium_budget = budget_usd * FRAC_MEDIUM
    n_medium_max  = ctx.affordable("tree_medium", medium_budget)
    cand_medium   = ctx.plantable & ~used
    n_medium      = min(n_medium_max, int(cand_medium.sum()))

    spacing_medium = max(int(round(MEDIUM_SPACING_M / ctx.res_m)), 1)
    mr, mc = _greedy_spaced(score, cand_medium, spacing_medium, n_medium)

    if mr.size:
        mr, mc, spent = _safe_place(ctx, "tree_medium", mr, mc, spent, budget_usd)
        if mr.size:
            placements.append(Placement("tree_medium", mr, mc))
            used[mr, mc] = True
            crown_med_px = max(int(round(CROWN_MED_M / ctx.res_m)), 1)
            _stamp_crown(covered, mr, mc, crown_med_px, ctx.shape)

    # ── Phase 2: Shade canopies (25% budget, hottest open ground) ──────────
    canopy_budget = min(budget_usd * FRAC_CANOPY, budget_usd - spent)
    cr = cc = np.array([], dtype=int)
    if canopy_budget >= ctx.unit_cost("shade_canopy"):
        open_ground  = ctx.buildable & ~covered & ~used
        n_canopy_max = ctx.affordable("shade_canopy", canopy_budget)
        n_canopy     = min(n_canopy_max, int(open_ground.sum()))

        cr, cc = _top_pixels(can_score, open_ground, n_canopy)
        if cr.size:
            cr, cc, spent = _safe_place(ctx, "shade_canopy", cr, cc, spent, budget_usd)
            if cr.size:
                placements.append(Placement("shade_canopy", cr, cc))
                used[cr, cc] = True

    # ── Phase 3: Grass conversion (10% budget, cobenefit_greened_pct) ──────
    grass_budget = min(budget_usd * FRAC_GRASS, budget_usd - spent)
    gr = gc = np.array([], dtype=int)
    if grass_budget >= ctx.unit_cost("grass_conversion"):
        cand_grass  = ctx.placeable("grass_conversion") & ~ctx.roadbed & ~used
        n_grass_max = ctx.affordable("grass_conversion", grass_budget)
        n_grass     = min(n_grass_max, int(cand_grass.sum()))

        gr, gc = _top_pixels(grass_sc, cand_grass, n_grass)
        if gr.size:
            gr, gc, spent = _safe_place(ctx, "grass_conversion", gr, gc, spent, budget_usd)
            if gr.size:
                placements.append(Placement("grass_conversion", gr, gc))
                used[gr, gc] = True

    # ── Phase 4: Small trees (5% budget, gap-fill between medium trees) ───
    small_budget = min(budget_usd * FRAC_SMALL, budget_usd - spent)
    sr = sc = np.array([], dtype=int)
    if small_budget >= ctx.unit_cost("tree_small"):
        cand_small  = ctx.plantable & ~used
        n_small_max = ctx.affordable("tree_small", small_budget)
        n_small     = min(n_small_max, int(cand_small.sum()))

        spacing_small = max(int(round(SMALL_SPACING_M / ctx.res_m)), 1)
        sr, sc = _greedy_spaced(score, cand_small, spacing_small, n_small)

        if sr.size:
            sr, sc, spent = _safe_place(ctx, "tree_small", sr, sc, spent, budget_usd)
            if sr.size:
                placements.append(Placement("tree_small", sr, sc))
                used[sr, sc] = True
                crown_small_px = max(int(round(CROWN_SMALL_M / ctx.res_m)), 1)
                _stamp_crown(covered, sr, sc, crown_small_px, ctx.shape)

    # ── Phase 5: Remaining budget → more shade canopies ───────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("shade_canopy"):
        open_ground2 = ctx.buildable & ~covered & ~used
        n_extra      = ctx.affordable("shade_canopy", remaining)
        n_extra      = min(n_extra, int(open_ground2.sum()))

        if n_extra > 0:
            er, ec = _top_pixels(can_score, open_ground2, n_extra)
            if er.size:
                er, ec, spent = _safe_place(
                    ctx, "shade_canopy", er, ec, spent, budget_usd
                )
                if er.size:
                    placements.append(Placement("shade_canopy", er, ec))
                    used[er, ec] = True

    # ── Phase 6: Final remainder → more grass conversion ──────────────────
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("grass_conversion"):
        cand_grass2  = ctx.placeable("grass_conversion") & ~ctx.roadbed & ~used
        n_grass2_max = ctx.affordable("grass_conversion", remaining)
        n_grass2     = min(n_grass2_max, int(cand_grass2.sum()))
        if n_grass2 > 0:
            gr2, gc2 = _top_pixels(grass_sc, cand_grass2, n_grass2)
            if gr2.size:
                gr2, gc2, spent = _safe_place(
                    ctx, "grass_conversion", gr2, gc2, spent, budget_usd
                )
                if gr2.size:
                    placements.append(Placement("grass_conversion", gr2, gc2))

    return placements