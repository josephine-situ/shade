from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "population-heat dense shade v3"
DESCRIPTION = (
    "Maximise population-weighted UTCI drop (heat_relief_c) by deploying medium "
    "trees at 4 m spacing on a population×heat priority surface, filling gaps "
    "with small trees at 3 m spacing, then covering remaining hot pedestrian "
    "ground with shade canopies. Budget: 68% medium trees, 14% small trees, "
    "18% shade canopies. Priority surface weights population (0.35) and afternoon "
    "heat (0.40) most heavily since fitness IS population-weighted UTCI, plus "
    "heat_hours (0.10), vulnerability (0.10), UHII (0.05). Tighter spacing "
    "maximises shade coverage on high-population corridors. Priority-tract boost "
    "0.20 (raised). Albedo actions avoided."
)

WEIGHTS = {
    "heat_ta3pm":    0.40,
    "heat_hours":    0.10,
    "population":    0.35,
    "vulnerability": 0.10,
    "uhii":          0.05,
}

PRIORITY_BOOST       = 0.20   # stronger boost for top-quartile tracts
TREE_MED_BUDGET_FRAC = 0.68
TREE_SML_BUDGET_FRAC = 0.14
CANOPY_BUDGET_FRAC   = 0.18

TREE_MED_SPACING_M   = 4.0   # tighter than parent (was 5m)
TREE_SML_SPACING_M   = 3.0   # tighter small tree spacing


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
    Composite priority heavily weighted toward population×heat signal,
    since heat_relief_c is population-weighted UTCI drop.
    """
    mask = ctx.exposure
    score = (
          WEIGHTS["heat_ta3pm"]    * _norm(ctx.heat_ta3pm,    mask)
        + WEIGHTS["heat_hours"]    * _norm(ctx.heat_hours,    mask)
        + WEIGHTS["population"]    * _norm(ctx.population,    mask)
        + WEIGHTS["vulnerability"] * _norm(ctx.vulnerability, mask)
        + WEIGHTS["uhii"]          * _norm(ctx.heat_uhii,     mask)
    )
    # Multiplicative interaction: population × heat boosts highest-impact zones
    pop_norm   = _norm(ctx.population,  mask)
    heat_norm  = _norm(ctx.heat_ta3pm,  mask)
    # Add cross-term to strongly favour dense+hot pixels
    score += 0.10 * pop_norm * heat_norm

    score += np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy selection: pick highest-scoring candidate, block a spacing_px-radius
    box around it, repeat until `limit` reached or candidates exhausted.
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
    """Select the top `limit` candidate pixels by score, no spacing constraint."""
    rows, cols = np.nonzero(candidates)
    if rows.size == 0 or limit <= 0:
        return np.array([], dtype=int), np.array([], dtype=int)
    order = np.argsort(-score[rows, cols])[:limit]
    return rows[order], cols[order]


def _mark_crown(
    covered: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
    crown_px: int,
    shape: tuple[int, int],
) -> None:
    """Stamp a box crown_px wide around each (r, c) into covered."""
    H, W = shape
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        r0 = max(0, r - crown_px);  r1 = min(H, r + crown_px + 1)
        c0 = max(0, c - crown_px);  c1 = min(W, c + crown_px + 1)
        covered[r0:r1, c0:c1] = True


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score      = priority_surface(ctx)
    placements: list[Placement] = []
    spent      = 0.0
    used       = np.zeros(ctx.shape, dtype=bool)

    # Pre-mark existing canopy so we don't waste canopy budget under it
    existing_canopy = ctx.cdsm > 0.0

    # ── 1. Medium street trees (68% budget, 4 m / 2 px spacing) ─────────
    medium_budget = budget_usd * TREE_MED_BUDGET_FRAC
    n_medium_max  = ctx.affordable("tree_medium", medium_budget)
    cand_medium   = ctx.plantable & ~used

    spacing_med_px = max(int(round(TREE_MED_SPACING_M / ctx.res_m)), 1)
    n_medium = min(n_medium_max, int(cand_medium.sum()))

    mr, mc = _greedy_spaced(score, cand_medium, spacing_med_px, n_medium)

    if mr.size:
        placements.append(Placement("tree_medium", mr, mc))
        spent += ctx.cost("tree_medium", mr.size)
        used[mr, mc] = True

    # ── 2. Small street trees (14% budget, 3 m spacing, fills gaps) ─────
    sml_budget = min(budget_usd * TREE_SML_BUDGET_FRAC, budget_usd - spent)
    n_sml_max  = ctx.affordable("tree_small", sml_budget)
    cand_sml   = ctx.plantable & ~used

    spacing_sml_px = max(int(round(TREE_SML_SPACING_M / ctx.res_m)), 1)
    n_sml = min(n_sml_max, int(cand_sml.sum()))

    sr, sc = _greedy_spaced(score, cand_sml, spacing_sml_px, n_sml)

    if sr.size:
        placements.append(Placement("tree_small", sr, sc))
        spent += ctx.cost("tree_small", sr.size)
        used[sr, sc] = True

    # ── 3. Shade canopies (18% budget → remaining, hottest open ground) ──
    remaining = budget_usd - spent
    if remaining <= ctx.unit_cost("shade_canopy"):
        return placements

    # Mark tree crown footprints
    covered = np.zeros(ctx.shape, dtype=bool)
    covered[existing_canopy] = True  # also exclude existing canopy from canopy placement

    # Medium tree crown ~3.5 m radius → ~2 px at 2 m resolution
    med_crown_px = max(int(round(3.5 / ctx.res_m)), 1)
    if mr.size:
        _mark_crown(covered, mr, mc, med_crown_px, ctx.shape)

    # Small tree crown ~2 m radius → 1 px at 2 m resolution
    sm_crown_px = max(int(round(2.0 / ctx.res_m)), 1)
    if sr.size:
        _mark_crown(covered, sr, sc, sm_crown_px, ctx.shape)

    # Open buildable ground not shaded by trees or existing canopy
    open_ground = ctx.buildable & ~covered & ~used

    n_canopy_max = ctx.affordable("shade_canopy", remaining)
    n_canopy     = min(n_canopy_max, int(open_ground.sum()))

    cr, cc = _top_pixels(score, open_ground, n_canopy)
    if cr.size:
        placements.append(Placement("shade_canopy", cr, cc))
        spent += ctx.cost("shade_canopy", cr.size)

    # ── 4. Use any remaining budget on additional small trees ────────────
    remaining2 = budget_usd - spent
    if remaining2 > ctx.unit_cost("tree_small"):
        cand_extra = ctx.plantable & ~used
        n_extra_max = ctx.affordable("tree_small", remaining2)
        n_extra = min(n_extra_max, int(cand_extra.sum()))
        if n_extra > 0:
            # Use tightest feasible spacing for gap-fill
            er, ec = _greedy_spaced(score, cand_extra, 1, n_extra)
            if er.size:
                placements.append(Placement("tree_small", er, ec))

    return placements