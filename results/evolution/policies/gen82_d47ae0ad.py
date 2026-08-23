from __future__ import annotations

import numpy as np
from policy_api import Placement, PlanningContext

POLICY_NAME = "dense-synergy-corridors equity-boost v1"
DESCRIPTION = (
    "Maximise population-weighted UTCI relief in cell (1,1,1,0). "
    "Three-phase dense shade deployment using geometric-mean heat×population "
    "synergy scoring. Phase 1: Medium trees at 3.5m spacing (65% budget) on "
    "hottest+most-populated pedestrian corridors. Phase 2: Small trees at 2.5m "
    "spacing (25% budget) for dense gap-fill. Phase 3: Shade canopies on the "
    "hottest remaining open buildable ground (10% budget). "
    "Priority surface uses 0.45 heat×pop synergy + 0.25 heat_ta3pm + "
    "0.15 population + 0.10 heat_hours + 0.05 vulnerability, with a 0.20 "
    "priority-tract boost to keep equity_ratio >= 1. No reflective surfaces, "
    "no grass conversion (keeps cobenefit_greened_pct below 0.1976 threshold)."
)

# Budget fractions
FRAC_MED    = 0.65
FRAC_SML    = 0.25
FRAC_CANOPY = 0.10

# Spacing
SPACING_MED_M = 3.5
SPACING_SML_M = 2.5

# Crown exclusion radii
CROWN_MED_M   = 3.5
CROWN_SML_M   = 2.0

# Priority tract boost
PRIORITY_BOOST = 0.20


def _norm(arr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Min-max normalise arr to [0, 1] over masked region; 0.5 if constant."""
    vals = arr[mask]
    if vals.size == 0:
        return np.zeros_like(arr, dtype="float64")
    lo, hi = float(np.min(vals)), float(np.max(vals))
    if hi - lo < 1e-9:
        return np.where(mask, 0.5, 0.0).astype("float64")
    return np.clip((arr.astype("float64") - lo) / (hi - lo), 0.0, 1.0)


def _synergy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Composite priority surface using heat×population geometric mean synergy.
    Targets pixels that are simultaneously hot AND populated for maximum
    person-degC relief per intervention. Mild vulnerability boost maintains
    equity_ratio >= 1 without sacrificing overall fitness.
    """
    mask = ctx.exposure

    heat_n   = _norm(ctx.heat_ta3pm,    mask)
    pop_n    = _norm(ctx.population,    mask)
    hours_n  = _norm(ctx.heat_hours,    mask)
    vuln_n   = _norm(ctx.vulnerability, mask)
    uhii_n   = _norm(ctx.heat_uhii,     mask)

    # Geometric mean synergy: high only when BOTH heat AND population are high
    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.45 * synergy
        + 0.25 * heat_n
        + 0.15 * pop_n
        + 0.10 * hours_n
        + 0.03 * vuln_n
        + 0.02 * uhii_n
    )
    # Priority-tract boost to maintain equity_ratio >= 1
    score += np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _canopy_score(ctx: PlanningContext) -> np.ndarray:
    """
    Score for shade canopy placement: emphasise heat×pop synergy but also
    weight vulnerability more to strengthen equity in residual coverage.
    """
    mask = ctx.exposure

    heat_n   = _norm(ctx.heat_ta3pm,    mask)
    pop_n    = _norm(ctx.population,    mask)
    hours_n  = _norm(ctx.heat_hours,    mask)
    vuln_n   = _norm(ctx.vulnerability, mask)
    uhii_n   = _norm(ctx.heat_uhii,     mask)

    synergy = np.sqrt(np.clip(heat_n * pop_n, 0.0, None))

    score = (
          0.40 * synergy
        + 0.22 * heat_n
        + 0.18 * vuln_n
        + 0.12 * pop_n
        + 0.05 * hours_n
        + 0.03 * uhii_n
    )
    score += np.where(ctx.priority, PRIORITY_BOOST, 0.0)
    return np.where(mask, score, -np.inf)


def _greedy_spaced(
    score: np.ndarray,
    candidates: np.ndarray,
    spacing_px: int,
    limit: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedy spaced placement: pick highest-scoring candidate pixel,
    suppress spacing_px-radius neighbourhood, repeat until limit reached.
    Efficiently handles large candidate sets via sorted indexing.
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
    radius_px: int,
    shape: tuple[int, int],
) -> None:
    """Mark a box of radius_px around each (r, c) as covered."""
    H, W = shape
    for i in range(len(rows)):
        r, c = int(rows[i]), int(cols[i])
        r0 = max(0, r - radius_px);  r1 = min(H, r + radius_px + 1)
        c0 = max(0, c - radius_px);  c1 = min(W, c + radius_px + 1)
        covered[r0:r1, c0:c1] = True


def _safe_place(
    ctx: PlanningContext,
    action: str,
    rows: np.ndarray,
    cols: np.ndarray,
    spent: float,
    budget_usd: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Trim placement to fit within remaining budget, return updated spent."""
    if rows.size == 0:
        return rows, cols, spent
    remaining    = budget_usd - spent
    affordable_n = ctx.affordable(action, remaining)
    if affordable_n <= 0:
        return np.array([], dtype=int), np.array([], dtype=int), spent
    rows = rows[:affordable_n]
    cols = cols[:affordable_n]
    spent += ctx.cost(action, rows.size)
    return rows, cols, spent


def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]:
    score_tree   = _synergy_score(ctx)
    score_canopy = _canopy_score(ctx)
    placements: list[Placement] = []
    spent = 0.0
    used  = np.zeros(ctx.shape, dtype=bool)

    # ── Phase 1: Medium street trees (65% budget, 3.5m spacing) ──────────
    # Medium trees have the highest per-tree UTCI benefit.
    # 3.5m spacing is denser than parent's 4m, maximizing shade corridor coverage.
    med_budget     = budget_usd * FRAC_MED
    n_med_max      = ctx.affordable("tree_medium", med_budget)
    cand_med       = ctx.plantable & ~used
    n_med          = min(n_med_max, int(cand_med.sum()))
    spacing_med_px = max(int(round(SPACING_MED_M / ctx.res_m)), 1)

    mr, mc = _greedy_spaced(score_tree, cand_med, spacing_med_px, n_med)
    if mr.size:
        mr, mc, spent = _safe_place(ctx, "tree_medium", mr, mc, spent, budget_usd)
        if mr.size:
            placements.append(Placement("tree_medium", mr, mc))
            used[mr, mc] = True

    # ── Phase 2: Small street trees (25% budget, 2.5m spacing) ────────────
    # Dense gap-fill between medium trees. 2.5m spacing maximises tree count
    # in remaining plantable space for broad shade coverage.
    # Avoids double-planting on used pixels.
    sml_budget     = min(budget_usd * FRAC_SML, budget_usd - spent)
    sr = sc        = np.array([], dtype=int)
    if sml_budget >= ctx.unit_cost("tree_small"):
        cand_sml       = ctx.plantable & ~used
        n_sml_max      = ctx.affordable("tree_small", sml_budget)
        n_sml          = min(n_sml_max, int(cand_sml.sum()))
        spacing_sml_px = max(int(round(SPACING_SML_M / ctx.res_m)), 1)

        sr, sc = _greedy_spaced(score_tree, cand_sml, spacing_sml_px, n_sml)
        if sr.size:
            sr, sc, spent = _safe_place(ctx, "tree_small", sr, sc, spent, budget_usd)
            if sr.size:
                placements.append(Placement("tree_small", sr, sc))
                used[sr, sc] = True

    # ── Phase 3: Shade canopies (remaining ~10% budget + any leftover) ────
    # Target hottest open buildable ground not already covered by tree crowns.
    # Shade canopies provide direct UTCI relief on exposed pedestrian ground.
    remaining = budget_usd - spent
    if remaining < ctx.unit_cost("shade_canopy"):
        return placements

    # Build coverage mask: existing canopy + newly planted tree crowns
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

    cr, cc = _top_pixels(score_canopy, open_ground, n_canopy)
    if cr.size:
        cr, cc, spent = _safe_place(ctx, "shade_canopy", cr, cc, spent, budget_usd)
        if cr.size:
            placements.append(Placement("shade_canopy", cr, cc))
            used[cr, cc] = True

    # ── Phase 4: Sweep any remaining budget into more medium trees ─────────
    # If budget remains (e.g. from spacing constraints limiting tree count),
    # plant more medium trees on any remaining plantable ground at 5m spacing.
    remaining2 = budget_usd - spent
    if remaining2 >= ctx.unit_cost("tree_medium"):
        cand_extra = ctx.plantable & ~used
        if cand_extra.any():
            n_extra_max    = ctx.affordable("tree_medium", remaining2)
            n_extra        = min(n_extra_max, int(cand_extra.sum()))
            spacing_ex_px  = max(int(round(5.0 / ctx.res_m)), 1)

            er, ec = _greedy_spaced(score_tree, cand_extra, spacing_ex_px, n_extra)
            if er.size:
                er, ec, spent = _safe_place(ctx, "tree_medium", er, ec, spent, budget_usd)
                if er.size:
                    placements.append(Placement("tree_medium", er, ec))
                    used[er, ec] = True

    # ── Phase 5: Final sweep — small trees on any remaining plantable pixels
    remaining3 = budget_usd - spent
    if remaining3 >= ctx.unit_cost("tree_small"):
        cand_final = ctx.plantable & ~used
        if cand_final.any():
            n_final_max   = ctx.affordable("tree_small", remaining3)
            n_final       = min(n_final_max, int(cand_final.sum()))
            spacing_fin_px = max(int(round(3.0 / ctx.res_m)), 1)

            fr, fc = _greedy_spaced(score_tree, cand_final, spacing_fin_px, n_final)
            if fr.size:
                fr, fc, spent = _safe_place(ctx, "tree_small", fr, fc, spent, budget_usd)
                if fr.size:
                    placements.append(Placement("tree_small", fr, fc))

    return placements