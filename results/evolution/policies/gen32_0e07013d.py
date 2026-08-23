from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "dense-corridor trees-canopy-grass v4"
DESCRIPTION = (
    "Maximise population-weighted UTCI relief via four-phase shading + greening: "
    "(1) Medium trees at 5m spacing using 60% budget on hottest populated corridors; "
    "(2) Small trees at 4m spacing using 13% budget for gap-fill; "
    "(3) Shade canopies on hottest remaining open ground using 20% budget; "
    "(4) Grass conversion on hot paved areas using remaining 7% budget to boost "
    "cobenefit_greened_pct above threshold for cell (0,1,1,1). "
    "Priority surface uses heat×population synergy (geometric mean) combined with "
    "linear terms. No reflective surfaces. Targets cell (0,1,1,1)."
)

# Budget fractions
FRAC_MEDIUM = 0.60
FRAC_SMALL  = 0.13
FRAC_CANOPY = 0.20
FRAC_GRASS  = 0.07  # grass conversion to push cobenefit_greened_pct above 0.1976


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
    Synergy-weighted priority surface.
    Core insight: UTCI fitness = population-weighted relief, so pixels that are
    simultaneously hot AND populated get the highest priority via geometric mean.
    """
    mask = ctx.exposure
    heat_n  = _norm(ctx.heat_ta3pm,  mask)
    pop_n   = _norm(ctx.population,  mask)
    hours_n = _norm(ctx.heat_hours,  mask)
    uhii_n  = _norm(ctx.heat_uhii,   mask)
    vuln_n  = _norm(ctx.vulnerability, mask)

    # Synergy term: sqrt(heat * pop) rewards pixels that are both hot and populated
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
        0.40 * synergy     # hot AND populated = max UTCI benefit
        + 0.20 * heat_n    # pure heat component
        + 0.20 * pop_n     # pure population component
        + 0.10 * hours_n   # cumulative heat exposure duration
        + 0.07 * uhii_n    # UHI intensity
        + 0.03 * vuln_n    # slight vulnerability weight
    )
    # Small priority-tract boost (keep equity_ratio near 1, not above)
    score += np.where(ctx.priority, 0.05, 0.0)
    return np.where(mask, score, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy selection by score with spatial exclusion zone.
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


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score = priority_surface(ctx)
    placements: list[Placement] = []
    spent = 0.0
    used  = np.zeros(ctx.shape, dtype=bool)

    # ── Phase 1: Medium street trees (60% budget, 5m spacing) ─────────────
    # Medium trees deliver the strongest per-tree UTCI benefit.
    med_budget   = budget_usd * FRAC_MEDIUM
    n_med_max    = ctx.affordable("tree_medium", med_budget)
    cand_med     = ctx.plantable & ~used
    n_med        = min(n_med_max, int(cand_med.sum()))

    spacing_med  = max(int(round(5.0 / ctx.res_m)), 1)
    mr, mc = _greedy_spaced(score, cand_med, spacing_med, n_med)

    if mr.size:
        placements.append(Placement("tree_medium", mr, mc))
        spent += ctx.cost("tree_medium", mr.size)
        used[mr, mc] = True

    # ── Phase 2: Small street trees (13% budget, 4m spacing, gap-fill) ───
    small_budget = min(budget_usd * FRAC_SMALL, budget_usd - spent)
    sr = sc = np.array([], dtype=int)
    if small_budget >= ctx.unit_cost("tree_small"):
        cand_small  = ctx.plantable & ~used
        n_small_max = ctx.affordable("tree_small", small_budget)
        n_small     = min(n_small_max, int(cand_small.sum()))

        spacing_sm  = max(int(round(4.0 / ctx.res_m)), 1)
        sr, sc = _greedy_spaced(score, cand_small, spacing_sm, n_small)

        if sr.size:
            placements.append(Placement("tree_small", sr, sc))
            spent += ctx.cost("tree_small", sr.size)
            used[sr, sc] = True

    # ── Phase 3: Shade canopies (20% budget, hottest open buildable ground) ─
    canopy_budget = min(budget_usd * FRAC_CANOPY, budget_usd - spent)
    cr = cc = np.array([], dtype=int)
    if canopy_budget >= ctx.unit_cost("shade_canopy"):
        # Exclude areas already covered by new tree crowns
        covered = np.zeros(ctx.shape, dtype=bool)
        if mr.size:
            _stamp_crown(covered, mr, mc,
                         max(int(round(3.5 / ctx.res_m)), 1), ctx.shape)
        if sr.size:
            _stamp_crown(covered, sr, sc,
                         max(int(round(2.0 / ctx.res_m)), 1), ctx.shape)

        open_ground = ctx.buildable & (ctx.cdsm <= 0.0) & ~covered & ~used

        n_canopy_max = ctx.affordable("shade_canopy", canopy_budget)
        n_canopy     = min(n_canopy_max, int(open_ground.sum()))

        cr, cc = _top_pixels(score, open_ground, n_canopy)
        if cr.size:
            actual = ctx.cost("shade_canopy", cr.size)
            if spent + actual <= budget_usd:
                placements.append(Placement("shade_canopy", cr, cc))
                spent += actual
                used[cr, cc] = True
            else:
                n_fit = ctx.affordable("shade_canopy", budget_usd - spent)
                if n_fit > 0:
                    placements.append(Placement("shade_canopy", cr[:n_fit], cc[:n_fit]))
                    spent += ctx.cost("shade_canopy", n_fit)
                    used[cr[:n_fit], cc[:n_fit]] = True

    # ── Phase 4: Grass conversion (remaining ~7% budget, hot paved areas) ──
    # Converts paved to grass: reduces surface temp, adds green cobenefit.
    # This pushes cobenefit_greened_pct above 0.1976 for cell (0,1,1,1).
    remaining = budget_usd - spent
    if remaining >= ctx.unit_cost("grass_conversion"):
        # grass_conversion goes on paved pixels (lc=1), not roadbed
        grass_cand = ctx.placeable("grass_conversion") & ~used

        if grass_cand.any():
            n_grass_max = ctx.affordable("grass_conversion", remaining)
            n_grass     = min(n_grass_max, int(grass_cand.sum()))

            # Prioritise hottest paved areas with some population
            # Use a combined heat+population score for grass placement
            mask = ctx.exposure
            heat_n = _norm(ctx.heat_ta3pm, mask)
            pop_n  = _norm(ctx.population, mask)
            grass_score = np.where(
                mask,
                0.60 * heat_n + 0.30 * pop_n + 0.10 * _norm(ctx.heat_hours, mask),
                -np.inf
            )
            # Also boost priority tracts slightly
            grass_score += np.where(ctx.priority, 0.05, 0.0)
            grass_score  = np.where(mask, grass_score, -np.inf)

            gr, gc = _top_pixels(grass_score, grass_cand, n_grass)
            if gr.size:
                actual = ctx.cost("grass_conversion", gr.size)
                if spent + actual <= budget_usd:
                    placements.append(Placement("grass_conversion", gr, gc))
                    spent += actual
                else:
                    n_fit = ctx.affordable("grass_conversion", budget_usd - spent)
                    if n_fit > 0:
                        placements.append(
                            Placement("grass_conversion", gr[:n_fit], gc[:n_fit])
                        )

    return placements