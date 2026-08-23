"""Evolve heat-resilience policies with an LLM-in-the-loop.

Minimal autoresearch harness: prompt an LLM to write a policy, score it
against SOLWEIG, store the result, and repeat.  Inspired by AlphaEvolve's
evolutionary code search, but synchronous and single-machine.

    # score the seed, then run 2 LLM generations on chinatown
    python scripts/evolve.py --generations 2 --aoi chinatown

    # different budget and model
    python scripts/evolve.py --generations 5 --budget 1000000 --model claude-sonnet-4-6

The seed policy (default: policies/baseline_policy.py) is scored first as
generation 0.  Each subsequent generation picks the best-scoring candidate
as parent, samples up to two other feasible candidates as "inspirations",
and asks the LLM to write a new policy that improves on the parent.

Every candidate -- feasible, infeasible, or crashed -- is stored as a JSON
file in the run directory so the evolutionary history is fully reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from itertools import product
from statistics import median
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config"
RUNS = ROOT / "runs"

# MAP-Elites axes: (metric_name, fixed_threshold_or_None).
# None means "compute median from seeding phase."
MAP_AXES = [
    ("equity_ratio",                      1.0),   # fixed: >=1 means helping vulnerable more
    ("access_gain_pp",                    None),   # median from seeding
    ("cost_efficiency_person_c_per_100k", None),   # median from seeding
    ("cobenefit_greened_pct",             None),   # median from seeding
]


# ── LLM interface ──────────────────────────────────────────────────────── #

_anthropic_client = None


def generate(prompt: str, system: str, *, model: str = "claude-sonnet-4-6") -> str:
    """Call the LLM and return the response text.

    Uses the Anthropic API (set ANTHROPIC_API_KEY in your environment).
    """
    global _anthropic_client
    import anthropic  # noqa: E402

    if _anthropic_client is None:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise SystemExit(
                "set ANTHROPIC_API_KEY in your environment "
                "(see https://console.anthropic.com/settings/keys)"
            )
        base_url = os.environ.get("ANTHROPIC_BASE_URL")
        kwargs = {"api_key": key}
        if base_url:
            kwargs["base_url"] = base_url
        _anthropic_client = anthropic.Anthropic(**kwargs)

    last_err = None
    for attempt in range(4):
        try:
            response = _anthropic_client.messages.create(
                model=model,
                max_tokens=16384,
                system=system,
                messages=[{"role": "user", "content": prompt}],
                timeout=600.0,
            )
            if not response.content:
                raise RuntimeError("LLM returned empty response")
            return response.content[0].text
        except Exception as exc:
            last_err = exc
            err_name = type(exc).__name__
            if "RateLimitError" in err_name or "OverloadedError" in err_name:
                wait = 2 ** (attempt + 1)
                print(f"  LLM rate-limited ({err_name}), retrying in {wait}s...")
                time.sleep(wait)
                continue
            raise
    raise last_err  # type: ignore[misc]


# ── code extraction ────────────────────────────────────────────────────── #

def extract_policy_code(response: str) -> str | None:
    """Pull the Python policy module out of a fenced code block."""
    # Try ```python ... ```  (take last match -- the final version)
    matches = re.findall(r"```python\s*\n(.*?)```", response, re.DOTALL)
    if matches:
        code = matches[-1].strip()
    else:
        # Try unfenced ``` ... ```
        matches = re.findall(r"```\s*\n(.*?)```", response, re.DOTALL)
        if matches:
            code = matches[-1].strip()
        elif "def plan(" in response:
            code = response.strip()
        else:
            return None

    if "def plan(" not in code:
        return None

    # Ensure POLICY_NAME and DESCRIPTION exist
    if "POLICY_NAME" not in code:
        code = ('POLICY_NAME = "evolved"\n'
                'DESCRIPTION = "LLM-generated policy"\n\n' + code)

    return code


# ── prompt construction ────────────────────────────────────────────────── #

def build_system_prompt() -> str:
    """The constant system prompt, built from config/interventions.json."""
    interventions = json.loads(
        (CONFIG / "interventions.json").read_text(encoding="utf-8")
    )["interventions"]

    rows = []
    for name, spec in interventions.items():
        cost = spec["cost_usd_per_unit"]
        unit = spec["unit"]
        lc = spec.get("applies_to_landcover", "ground")
        label = spec["label"]
        rows.append(f"  {name:<18s} ${cost:>7.0f}/{unit:<4s}  lc={lc!s:<12s}  {label}")
    actions_table = "\n".join(rows)

    return f"""\
You are an autoresearch agent improving urban heat-resilience policies for Boston.

## The policy contract

A policy is a Python module with:

    POLICY_NAME = "short name"
    DESCRIPTION = "what the policy does and why"

    def plan(ctx: PlanningContext, budget_usd: float) -> list[Placement]

It must start with these imports:
    from __future__ import annotations
    import numpy as np
    from policy_api import Placement, PlanningContext

No other external libraries may be imported. The module is loaded with
importlib so it must be self-contained.

## What ctx provides

Grids (all numpy arrays on the same shape):
  ctx.shape             tuple (rows, cols) of the AOI grid
  ctx.res_m             pixel size in metres (2.0)
  ctx.landcover         uint8, UMEP codes: 1=paved 2=building 3=evergreen 4=deciduous 5=grass 6=bare 7=water
  ctx.dsm               ground + buildings, metres above sea level
  ctx.dem               bare earth, metres above sea level
  ctx.cdsm              canopy height above ground in metres (0 where no canopy)
  ctx.building_height   building height above terrain in metres, 0 off buildings
  ctx.heat_ta3pm        modelled 3 PM air temperature, degC -- hotter = more need
  ctx.heat_ta3am        modelled 3 AM air temperature, degC
  ctx.heat_uhii         urban heat island intensity, degC
  ctx.heat_hours        annual heat-event hours
  ctx.population        people per pixel on pedestrian space
  ctx.vulnerability     0-1 citywide percentile of tract social vulnerability
  ctx.priority          bool: tract is in the top vulnerability quartile

Masks (all bool):
  ctx.exposure          where scores are measured (sidewalk + crosswalk corridor)
  ctx.walkable          ground a person can stand on
  ctx.plantable         every siting rule for a tree satisfied (roadbed, hydrants, width, canopy, etc.)
  ctx.buildable         every siting rule for a canopy/awning satisfied
  ctx.roadbed           paved travel lane -- NOTHING may go here

Methods:
  ctx.placeable(action) -> bool mask: eligible land cover AND siting rules pass
  ctx.affordable(action, budget) -> int: how many pixels (or trees) fit in the budget
  ctx.cost(action, n) -> float: USD for n pixels of that action
  ctx.unit_cost(action) -> float: USD per pixel or per tree
  ctx.eligible(action) -> bool mask: land cover allows this action
  ctx.spec(action) -> dict: the intervention spec from config

## Available actions

{actions_table}

Notes:
- tree_small and tree_medium bill per TREE, not per pixel. One Placement with
  N row/col pairs = N trees.
- All other actions bill per PIXEL (unit = m2, cost = cost_per_m2 * pixel_area_m2).
- tree actions go on ctx.plantable; canopy actions go on ctx.buildable;
  roof actions go on building pixels; road/grass on paved pixels.

## Scoring -- THIS IS CRITICAL

heat_relief_c: population-weighted UTCI drop on pedestrian space. THIS IS THE FITNESS.
  Higher is better. This is what you are trying to maximise.

THE ALBEDO TRAP: shade moves perceived temperature (UTCI) strongly.
  Reflective pavement (light_road) moves SURFACE temperature but barely
  moves UTCI -- it even raises Tmrt by reflecting more shortwave onto
  pedestrians. A policy that buys albedo to look busy will score WORSE
  than one that buys shade. Trees and canopies dominate for UTCI relief.

Other reported scores (not the fitness, but important):
  equity_ratio: relief in top-vulnerability areas / overall, pooled over every AOI.
    >1 = helping the vulnerable more. Vulnerability is flat within a census tract,
    so this responds to which tracts you treat, not which streets inside one --
    and some AOIs contain no top-quartile tract, so spending there lowers it.
  access_gain_pp: share of exposed residents moved below the 32C UTCI heat-stress threshold.
  cost_efficiency_person_c_per_100k: person-degC of relief per $100k spent.
  cobenefit_greened_pct: new canopy or green area as % of walkable ground.

## Rules that make a policy infeasible (ALL must hold)

- Total spend must not exceed budget_usd.
- Every pixel must pass ctx.placeable(action). Use ctx.plantable for trees,
  ctx.buildable for canopies. Placing on roadbed, near hydrants, on wrong
  land cover, or under existing canopy is a violation.
- No pixel may appear in two Placements.
- plan() must return within 120 seconds wall-clock.

A single violation makes the ENTIRE policy infeasible: zero score, no
simulation. The auditor reports the violation strings.

## Common mistakes to avoid

- Placing trees on ANY paved pixel instead of ctx.plantable (roadbed is paved too).
- Using ctx.eligible() alone without ctx.sitable() -- eligible checks land cover,
  sitable checks physical rules. Use ctx.placeable() or ctx.plantable/ctx.buildable.
- Indexing rows/cols outside ctx.shape.
- Returning duplicate pixels across Placements (double-booking).
- Overspending the budget by even $1 (use ctx.affordable() to compute limits).
- Forgetting the imports (from policy_api import Placement, PlanningContext).

## Output format

Write the COMPLETE Python module inside a ```python code fence. Start with
imports, then POLICY_NAME, DESCRIPTION, helper functions if needed, then
def plan(). The module must be fully self-contained and runnable.
"""


def build_user_prompt(
    parent: dict,
    inspirations: list[dict],
    *,
    map_elites_context: dict | None = None,
) -> str:
    """Build the per-generation prompt from parent + inspirations."""
    parts = []

    # Parent
    parts.append("## Parent policy (the current best -- improve on this)\n")
    parts.append(f"Name: {parent.get('policy_name', 'unknown')}")
    parts.append(f"Description: {parent.get('description', 'none')}")
    if parent.get("fitness") is not None:
        parts.append(f"Fitness (heat_relief_c): {parent['fitness']}")
    if parent.get("objectives"):
        obj = parent["objectives"]
        score_lines = []
        for k in ("heat_relief_c", "access_gain_pp", "equity_ratio",
                   "cobenefit_greened_pct", "cost_efficiency_person_c_per_100k"):
            if obj.get(k) is not None:
                score_lines.append(f"  {k}: {obj[k]}")
        if score_lines:
            parts.append("Scores:\n" + "\n".join(score_lines))
    if parent.get("violations"):
        parts.append(f"Violations: {json.dumps(parent['violations'], indent=2)}")
    parts.append(f"\n```python\n{parent['code']}\n```\n")

    # MAP-Elites grid context (only when active)
    if map_elites_context is not None:
        me = map_elites_context
        parts.append("## MAP-Elites grid context\n")
        parts.append(
            f"You are improving cell {tuple(me['parent_cell'])} "
            f"in a 4D behavior grid.\n"
        )
        parts.append("Axes and thresholds (low=0, high=1):")
        for axis_name in me["axes"]:
            t = me["thresholds"][axis_name]
            parts.append(f"  {axis_name} >= {t:.4g} \u2192 high")

        total_cells = 2 ** len(me["axes"])
        occupied = me["grid_summary"]
        parts.append(f"\nOccupied cells ({len(occupied)}/{total_cells}):")
        for entry in occupied:
            cell_str = str(tuple(entry["cell"]))
            marker = " \u2190 your parent" if entry["cell"] == me["parent_cell"] else ""
            parts.append(
                f"  {cell_str}: fitness={entry['fitness']:.4f}  "
                f"\"{entry['policy_name']}\"{marker}"
            )

        empty = []
        for combo in product(range(2), repeat=len(me["axes"])):
            if list(combo) not in [e["cell"] for e in occupied]:
                empty.append(str(combo))
        if empty:
            parts.append(f"\nEmpty cells ({len(empty)}/{total_cells}): {', '.join(empty)}")
        parts.append("")

    # Inspirations
    for i, insp in enumerate(inspirations, 1):
        parts.append(f"## Inspiration {i} (another approach from the database)\n")
        parts.append(f"Name: {insp.get('policy_name', 'unknown')}")
        if insp.get("fitness") is not None:
            parts.append(f"Fitness (heat_relief_c): {insp['fitness']}")
        if insp.get("objectives"):
            obj = insp["objectives"]
            for k in ("heat_relief_c", "equity_ratio", "cost_efficiency_person_c_per_100k"):
                if obj.get(k) is not None:
                    parts.append(f"  {k}: {obj[k]}")
        parts.append(f"\n```python\n{insp['code']}\n```\n")

    # Task
    fitness_str = ""
    if parent.get("fitness") is not None:
        fitness_str = f" of {parent['fitness']}"

    if map_elites_context is not None:
        me = map_elites_context
        parts.append(f"""## Your task

Your parent is the champion of cell {tuple(me['parent_cell'])} with fitness{fitness_str}.

You can EITHER:
1. Beat this cell's champion fitness ({parent.get('fitness')}), keeping a similar
   behavioral profile (equity, access, efficiency, greening), OR
2. Explore a fundamentally different strategy that lands in an EMPTY cell —
   especially cells with high equity (first axis = 1), which no policy may have achieved yet.

Think about:
- Different budget splits between actions (trees vs canopies vs other)
- Different priority surfaces or ranking strategies
- Better spatial strategies (spacing, clustering, corridor coverage)
- Targeting different populations, heat patterns, or vulnerability
- Creative use of all 8 available actions

Write the COMPLETE Python module. It must be standalone and runnable.""")
    else:
        parts.append(f"""## Your task

Write a new policy that improves on the parent's heat_relief_c{fitness_str}.

Think about:
- Different budget splits between actions (trees vs canopies vs other)
- Different priority surfaces or ranking strategies
- Better spatial strategies (spacing, clustering, corridor coverage)
- Targeting different populations, heat patterns, or vulnerability
- Creative use of all 8 available actions

Write the COMPLETE Python module. It must be standalone and runnable.""")

    return "\n".join(parts)


# ── candidate database ─────────────────────────────────────────────────── #

def _candidate_id(generation: int, code: str | None) -> str:
    if generation == 0:
        return "gen00_seed"
    tag = hashlib.sha256((code or "").encode()).hexdigest()[:8]
    return f"gen{generation:02d}_{tag}"


def save_candidate(run_dir: Path, candidate: dict) -> Path:
    cand_dir = run_dir / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    path = cand_dir / f"{candidate['id']}.json"
    path.write_text(json.dumps(candidate, indent=2), encoding="utf-8")
    return path


def load_candidates(run_dir: Path) -> list[dict]:
    cand_dir = run_dir / "candidates"
    if not cand_dir.exists():
        return []
    candidates = []
    for path in sorted(cand_dir.glob("*.json")):
        candidates.append(json.loads(path.read_text(encoding="utf-8")))
    return candidates


def best_candidate(candidates: list[dict]) -> dict | None:
    feasible = [c for c in candidates if c.get("fitness") is not None]
    if not feasible:
        return None
    return max(feasible, key=lambda c: c["fitness"])


def sample_inspirations(candidates: list[dict], exclude_id: str, n: int = 2) -> list[dict]:
    pool = [c for c in candidates
            if c.get("fitness") is not None and c["id"] != exclude_id]
    if not pool:
        return []
    return random.sample(pool, min(n, len(pool)))


# ── MAP-Elites machinery ──────────────────────────────────────────────── #

def compute_thresholds(candidates: list[dict]) -> dict[str, float]:
    """Compute bin thresholds from feasible candidates.

    Fixed thresholds (e.g. equity_ratio = 1.0) are used as-is.
    Others are set to the median of all feasible values for that metric.
    """
    feasible = [c for c in candidates
                if c.get("objectives") and c.get("fitness") is not None]
    thresholds: dict[str, float] = {}
    for name, fixed in MAP_AXES:
        if fixed is not None:
            thresholds[name] = fixed
        else:
            vals = [c["objectives"][name] for c in feasible
                    if c["objectives"].get(name) is not None]
            thresholds[name] = median(vals) if vals else 0.0
    return thresholds


def candidate_cell(
    candidate: dict, thresholds: dict[str, float]
) -> tuple[int, ...] | None:
    """Map a candidate to its 4D grid cell, or None if objectives are missing."""
    obj = candidate.get("objectives")
    if not obj:
        return None
    bins = []
    for name, _ in MAP_AXES:
        val = obj.get(name)
        if val is None:
            return None
        bins.append(1 if val >= thresholds[name] else 0)
    return tuple(bins)


def build_grid(
    candidates: list[dict], thresholds: dict[str, float]
) -> dict[tuple[int, ...], dict]:
    """Place all feasible candidates in the MAP-Elites grid.

    Each cell keeps only the champion (highest fitness).
    """
    grid: dict[tuple[int, ...], dict] = {}
    for c in candidates:
        if c.get("fitness") is None:
            continue
        cell = candidate_cell(c, thresholds)
        if cell is None:
            continue
        if cell not in grid or c["fitness"] > grid[cell]["fitness"]:
            grid[cell] = c
    return grid


def select_parent_mapelites(
    grid: dict[tuple[int, ...], dict],
) -> tuple[tuple[int, ...], dict]:
    """Uniform random selection of an occupied cell's champion."""
    cells = list(grid.keys())
    cell = random.choice(cells)
    return cell, grid[cell]


def select_inspirations_mapelites(
    grid: dict[tuple[int, ...], dict],
    parent_cell: tuple[int, ...],
    n: int = 2,
) -> list[dict]:
    """Sample inspiration candidates from cells other than the parent's."""
    other = [c for c in grid if c != parent_cell]
    chosen = random.sample(other, min(n, len(other)))
    return [grid[c] for c in chosen]


# ── scoring ────────────────────────────────────────────────────────────── #

def score_candidate(
    policy_path: Path,
    aoi: str,
    budget: float,
    out_dir: Path,
    scenarios: str = "baseline",
    plan_timeout: float = 120.0,
    score_timeout: float = 600.0,
) -> dict | None:
    """Run score_policy.py as a subprocess, return parsed score.json or None."""
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "score_policy.py"),
        "--policy", str(policy_path.resolve()),
        "--aoi", aoi,
        "--budget", str(budget),
        "--out", str(out_dir),
        "--scenarios", scenarios,
        "--plan-timeout", str(plan_timeout),
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=score_timeout,
            cwd=str(ROOT),
        )
    except subprocess.TimeoutExpired:
        print(f"  scoring timed out after {score_timeout}s")
        return None

    if result.returncode != 0:
        print(f"  scorer exited with code {result.returncode}")
        if result.stderr:
            # Print last 5 lines of stderr for diagnostics
            for line in result.stderr.strip().splitlines()[-5:]:
                print(f"    {line}")

    score_path = out_dir / "score.json"
    if not score_path.exists():
        print(f"  no score.json produced")
        return None

    return json.loads(score_path.read_text(encoding="utf-8"))


def score_candidate_multi(
    policy_path: Path,
    aois: list[str],
    budget: float,
    out_dir: Path,
    scenarios: str = "baseline",
    plan_timeout: float = 120.0,
    score_timeout: float = 600.0,
) -> list[tuple[str, dict | None]]:
    """Score a policy on multiple AOIs in parallel.

    Launches one score_policy.py subprocess per AOI simultaneously.
    Returns [(aoi_name, parsed_score_json_or_None), ...].
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Launch all subprocesses
    procs: list[tuple[str, subprocess.Popen, Path]] = []
    for aoi in aois:
        aoi_dir = out_dir / aoi
        aoi_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            str(ROOT / "scripts" / "score_policy.py"),
            "--policy", str(policy_path.resolve()),
            "--aoi", aoi,
            "--budget", str(budget),
            "--out", str(aoi_dir),
            "--scenarios", scenarios,
            "--plan-timeout", str(plan_timeout),
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
            cwd=str(ROOT),
        )
        procs.append((aoi, proc, aoi_dir))

    # Wait for all to complete
    results: list[tuple[str, dict | None]] = []
    for aoi, proc, aoi_dir in procs:
        try:
            proc.wait(timeout=score_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            print(f"  {aoi}: timed out after {score_timeout}s")
            results.append((aoi, None))
            continue

        if proc.returncode != 0:
            stderr = proc.stderr.read() if proc.stderr else ""
            print(f"  {aoi}: scorer exited with code {proc.returncode}")
            if stderr:
                for line in stderr.strip().splitlines()[-3:]:
                    print(f"    {line}")

        score_path = aoi_dir / "score.json"
        if not score_path.exists():
            print(f"  {aoi}: no score.json produced")
            results.append((aoi, None))
        else:
            results.append(
                (aoi, json.loads(score_path.read_text(encoding="utf-8")))
            )

    return results


def trace_lineage(candidate: dict, candidates: list[dict]) -> list[dict]:
    """Walk parent_id links back to the seed, returning the chain.

    Returns [candidate, parent, grandparent, ..., seed] — most recent first.
    """
    by_id = {c["id"]: c for c in candidates}
    chain = [candidate]
    seen = {candidate["id"]}
    cur = candidate
    while cur.get("parent_id") and cur["parent_id"] in by_id:
        if cur["parent_id"] in seen:
            break  # safety: avoid cycles
        parent = by_id[cur["parent_id"]]
        chain.append(parent)
        seen.add(parent["id"])
        cur = parent
    return chain


def aggregate_aoi_results(
    results: list[tuple[str, dict]],
) -> dict:
    """Aggregate per-AOI score.json dicts into combined objectives.

    Uses the same logic as score_policy.py's aggregate():
      - mean for heat_relief_c, access_gain_pp, cobenefit_greened_pct
      - pooled for equity_ratio (per-scenario, then mean across scenarios)
      - pooled for cost_efficiency_person_c_per_100k
    """
    # Collect all run records across AOIs
    all_runs: list[dict] = []
    for _aoi, score_json in results:
        all_runs.extend(score_json.get("runs", []))

    if not all_runs:
        return {"heat_relief_c": None, "verdict": "feasible"}

    # Group by scenario
    by_scenario: dict[str, list[dict]] = {}
    for r in all_runs:
        by_scenario.setdefault(r["scenario"], []).append(r)

    # Simple means across all runs
    def _mean(key: str) -> float | None:
        vals = [r["metrics"][key] for r in all_runs
                if r["metrics"].get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    heat_relief_c = _mean("heat_relief_c")
    access_gain_pp = _mean("access_gain_pp")
    cobenefit_greened_pct = _mean("cobenefit_greened_pct")

    # Pooled equity: per-scenario, then mean across scenarios
    scenario_ratios: list[float] = []
    for _scen, runs in by_scenario.items():
        total_pop = sum(r["metrics"].get("pop_exposed", 0) for r in runs)
        total_pri_pop = sum(r["metrics"].get("equity_pop", 0) for r in runs)
        total_pdegc = sum(r["metrics"].get("person_degc", 0) for r in runs)
        total_pri_pdegc = sum(
            r["metrics"].get("equity_person_degc", 0) for r in runs
        )
        overall = total_pdegc / total_pop if total_pop > 0 else None
        priority = total_pri_pdegc / total_pri_pop if total_pri_pop > 0 else None
        if overall and overall > 0.01 and priority is not None:
            scenario_ratios.append(priority / overall)
    equity_ratio = (
        sum(scenario_ratios) / len(scenario_ratios)
        if scenario_ratios else None
    )

    # Pooled cost efficiency
    n_scenarios = max(len(by_scenario), 1)
    total_person_degc = sum(
        r["metrics"].get("person_degc", 0) for r in all_runs
    )
    # spend_usd is per-AOI (same across scenarios), take first run per AOI
    by_aoi: dict[str, list[dict]] = {}
    for r in all_runs:
        by_aoi.setdefault(r["aoi"], []).append(r)
    total_spend = sum(rows[0]["spend_usd"] for rows in by_aoi.values())
    cost_efficiency = (
        (total_person_degc / n_scenarios) / (total_spend / 1e5)
        if total_spend > 0 else None
    )

    # Per-AOI heat relief breakdown
    per_aoi = {}
    for aoi, runs in by_aoi.items():
        vals = [r["metrics"]["heat_relief_c"] for r in runs
                if r["metrics"].get("heat_relief_c") is not None]
        per_aoi[aoi] = sum(vals) / len(vals) if vals else None

    return {
        "heat_relief_c": round(heat_relief_c, 4) if heat_relief_c else None,
        "access_gain_pp": round(access_gain_pp, 4) if access_gain_pp else None,
        "equity_ratio": round(equity_ratio, 4) if equity_ratio else None,
        "cobenefit_greened_pct": (
            round(cobenefit_greened_pct, 4) if cobenefit_greened_pct else None
        ),
        "cost_efficiency_person_c_per_100k": (
            round(cost_efficiency, 2) if cost_efficiency else None
        ),
        "spend_usd": round(total_spend, 2),
        "per_aoi_heat_relief_c": {
            k: round(v, 4) if v else None for k, v in per_aoi.items()
        },
    }


# ── main loop ──────────────────────────────────────────────────────────── #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evolve heat-resilience policies with an LLM-in-the-loop.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--generations", type=int, default=10,
                     help="number of LLM evolution iterations (default 10)")
    ap.add_argument("--budget", type=float, default=500_000.0,
                     help="USD per AOI (default 500000)")
    ap.add_argument("--aois", default="chinatown,brighton,grove_hall",
                     help="comma-separated AOI names (default chinatown,brighton,grove_hall)")
    ap.add_argument("--model", default="claude-sonnet-4-6",
                     help="LLM model identifier (default claude-sonnet-4-6)")
    ap.add_argument("--seed-policy", default="policies/baseline_policy.py",
                     help="path to the seed policy (default policies/baseline_policy.py)")
    ap.add_argument("--scenarios", default="baseline",
                     help="weather scenarios, comma-separated (default baseline)")
    ap.add_argument("--plan-timeout", type=float, default=120.0,
                     help="seconds for plan() to run (default 120)")
    ap.add_argument("--score-timeout", type=float, default=600.0,
                     help="wall-clock timeout for the scoring subprocess (default 600)")
    ap.add_argument("--seed-generations", type=int, default=5,
                     help="generations of random exploration before MAP-Elites (default 5)")
    ap.add_argument("--resume", metavar="RUN_DIR",
                     help="resume evolution from a previous run directory")
    ap.add_argument("--out", help="output directory (default runs/evolve_<timestamp>)")
    args = ap.parse_args()

    # -- setup ------------------------------------------------------------ #
    aois = [a.strip() for a in args.aois.split(",")]
    n_aois = len(aois)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # -- resume or fresh start -------------------------------------------- #
    if args.resume:
        run_dir = Path(args.resume)
        if not run_dir.exists():
            raise SystemExit(f"resume directory not found: {run_dir}")
        existing = load_candidates(run_dir)
        if not existing:
            raise SystemExit(f"no candidates found in {run_dir}/candidates/")

        # Find the highest generation number already completed
        start_gen = max(c.get("generation", 0) for c in existing) + 1
        feasible_existing = [c for c in existing if c.get("fitness") is not None]

        # Rebuild thresholds if they exist
        thresholds_path = run_dir / "thresholds.json"
        if thresholds_path.exists():
            thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
            map_elites_active = True
        else:
            thresholds = None
            map_elites_active = False

        print(f"SHADE evolution harness (RESUMING)")
        print(f"  run dir:      {run_dir}")
        print(f"  resuming from gen {start_gen} ({len(existing)} candidates, "
              f"{len(feasible_existing)} feasible)")
        print(f"  aois:         {', '.join(aois)} ({len(aois)})")
        print(f"  scenarios:    {args.scenarios}")
        print(f"  budget:       ${args.budget:,.0f}")
        print(f"  generations:  {start_gen} to {start_gen + args.generations - 1} "
              f"({args.generations} new)")
        print(f"  model:        {args.model}")
        if map_elites_active:
            grid = build_grid(existing, thresholds)
            print(f"  MAP-Elites:   active ({len(grid)}/16 cells occupied)")
        print()

        end_gen = start_gen + args.generations
    else:
        seed_path = Path(args.seed_policy)
        if not seed_path.is_absolute():
            seed_path = ROOT / seed_path
        if not seed_path.exists():
            raise SystemExit(f"seed policy not found: {seed_path}")

        run_dir = Path(args.out) if args.out else RUNS / f"evolve_{stamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "candidates").mkdir(exist_ok=True)
        (run_dir / "policies").mkdir(exist_ok=True)

        seed_code = seed_path.read_text(encoding="utf-8")

        print(f"SHADE evolution harness")
        print(f"  run dir:      {run_dir}")
        print(f"  aois:         {', '.join(aois)} ({len(aois)})")
        print(f"  scenarios:    {args.scenarios}")
        print(f"  budget:       ${args.budget:,.0f}")
        print(f"  generations:  {args.generations}")
        print(f"  seed-gens:    {args.seed_generations}")
        print(f"  model:        {args.model}")
        print(f"  seed:         {seed_path.name}")
        print()

        # -- generation 0: score the seed --------------------------------- #
        print(f"gen 0  scoring seed policy on {n_aois} AOI{'s' if n_aois > 1 else ''}...")
        score_dir = run_dir / "score_gen00_seed"

        if n_aois == 1:
            result = score_candidate(
                seed_path, aois[0], args.budget, score_dir,
                scenarios=args.scenarios,
                plan_timeout=args.plan_timeout,
                score_timeout=args.score_timeout,
            )
            if result is None:
                raise SystemExit("seed policy failed to score -- fix it before evolving")
            seed_objectives = result.get("objectives")
            seed_verdict = result.get("verdict", "unknown")
            seed_violations = result.get("violations")
        else:
            aoi_results = score_candidate_multi(
                seed_path, aois, args.budget, score_dir,
                scenarios=args.scenarios,
                plan_timeout=args.plan_timeout,
                score_timeout=args.score_timeout,
            )
            failed = [(n, r) for n, r in aoi_results if r is None]
            if failed:
                raise SystemExit(
                    f"seed policy failed to score on: "
                    f"{', '.join(n for n, _ in failed)}"
                )
            infeasible = [
                (n, r) for n, r in aoi_results
                if r and r.get("verdict") != "feasible"
            ]
            if infeasible:
                raise SystemExit(
                    f"seed policy infeasible on: "
                    f"{', '.join(n for n, _ in infeasible)}"
                )
            feasible_results = [
                (n, r) for n, r in aoi_results
                if r and r.get("verdict") == "feasible"
            ]
            seed_objectives = aggregate_aoi_results(feasible_results)
            seed_verdict = "feasible"
            seed_violations = {}

        seed_fitness = (
            seed_objectives.get("heat_relief_c") if seed_objectives else None
        )

        # Extract POLICY_NAME and DESCRIPTION from seed code
        seed_name = "baseline"
        seed_desc = ""
        name_match = re.search(r'POLICY_NAME\s*=\s*["\'](.+?)["\']', seed_code)
        if name_match:
            seed_name = name_match.group(1)
        desc_match = re.search(r'DESCRIPTION\s*=\s*\(\s*["\'](.+?)["\']', seed_code, re.DOTALL)
        if desc_match:
            seed_desc = desc_match.group(1)
        else:
            desc_match = re.search(r'DESCRIPTION\s*=\s*["\'](.+?)["\']', seed_code)
            if desc_match:
                seed_desc = desc_match.group(1)

        seed_candidate = {
            "id": "gen00_seed",
            "generation": 0,
            "parent_id": None,
            "inspiration_ids": [],
            "policy_name": seed_name,
            "description": seed_desc,
            "code": seed_code,
            "verdict": seed_verdict,
            "objectives": seed_objectives,
            "violations": seed_violations,
            "fitness": seed_fitness,
            "aois_scored": aois,
            "timestamp_utc": stamp,
            "model": "seed",
        }
        save_candidate(run_dir, seed_candidate)

        print(f"gen 0  {seed_verdict}  fitness={seed_fitness}")
        if seed_fitness is None:
            print("  WARNING: seed is infeasible, evolution will proceed but has no parent to improve on")
        print()

        start_gen = 1
        end_gen = args.generations + 1
        thresholds = None
        map_elites_active = False

    # -- build system prompt ---------------------------------------------- #
    system_prompt = build_system_prompt()

    # -- evolution loop --------------------------------------------------- #
    for gen in range(start_gen, end_gen):
        print(f"gen {gen}  ", end="", flush=True)
        candidates = load_candidates(run_dir)
        feasible = [c for c in candidates if c.get("fitness") is not None]

        # -- Phase transition check ---------------------------------------- #
        if not map_elites_active:
            if gen > args.seed_generations and len(feasible) >= 2:
                thresholds = compute_thresholds(feasible)
                (run_dir / "thresholds.json").write_text(
                    json.dumps(thresholds, indent=2), encoding="utf-8"
                )
                map_elites_active = True
                print()
                print(f">>> MAP-Elites activated at gen {gen} "
                      f"with {len(feasible)} feasible candidates")
                for ax, val in thresholds.items():
                    print(f"    {ax}: {val:.4g}")
                print()
                print(f"gen {gen}  ", end="", flush=True)
            elif gen > args.seed_generations:
                print(f"(extending seeding: only {len(feasible)} feasible)  ",
                      end="", flush=True)

        # -- Parent & inspiration selection -------------------------------- #
        me_context = None
        parent_cell = None

        if map_elites_active:
            grid = build_grid(candidates, thresholds)
            parent_cell, parent = select_parent_mapelites(grid)
            inspirations = select_inspirations_mapelites(grid, parent_cell, n=2)

            grid_summary = [
                {"cell": list(cell), "fitness": champ["fitness"],
                 "policy_name": champ.get("policy_name", "?")}
                for cell, champ in sorted(grid.items())
            ]
            me_context = {
                "parent_cell": list(parent_cell),
                "thresholds": thresholds,
                "grid_summary": grid_summary,
                "axes": [name for name, _ in MAP_AXES],
            }
        else:
            # Seeding phase: random parent to encourage diversity
            if feasible:
                parent = random.choice(feasible)
            else:
                parent = candidates[0]
                print("(no feasible parent, using seed)  ", end="", flush=True)
            inspirations = sample_inspirations(candidates, parent["id"], n=2)

        # Build prompt and call LLM
        user_prompt = build_user_prompt(
            parent, inspirations, map_elites_context=me_context
        )
        print("calling LLM...", end="", flush=True)

        cand_id = None
        try:
            t0 = time.time()
            response = generate(user_prompt, system_prompt, model=args.model)
            llm_seconds = time.time() - t0
            print(f"  ({llm_seconds:.1f}s)  ", end="", flush=True)
        except Exception as exc:
            print(f"  LLM error: {exc}")
            cand_id = _candidate_id(gen, None)
            save_candidate(run_dir, {
                "id": cand_id,
                "generation": gen,
                "parent_id": parent["id"],
                "inspiration_ids": [i["id"] for i in inspirations],
                "policy_name": "llm_error",
                "description": str(exc),
                "code": None,
                "verdict": "llm_error",
                "objectives": None,
                "violations": None,
                "fitness": None,
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "model": args.model,
            })
            print()
            continue

        # Extract code
        code = extract_policy_code(response)
        if code is None:
            print("  code extraction failed")
            cand_id = _candidate_id(gen, response)
            save_candidate(run_dir, {
                "id": cand_id,
                "generation": gen,
                "parent_id": parent["id"],
                "inspiration_ids": [i["id"] for i in inspirations],
                "policy_name": "extraction_failed",
                "description": "could not extract a plan() function from LLM response",
                "code": response[:5000],  # store truncated response for debugging
                "verdict": "extraction_failed",
                "objectives": None,
                "violations": None,
                "fitness": None,
                "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
                "model": args.model,
            })
            print()
            continue

        cand_id = _candidate_id(gen, code)

        # Write policy file
        policy_path = run_dir / "policies" / f"{cand_id}.py"
        policy_path.write_text(code, encoding="utf-8")

        # Score
        print(f"scoring {n_aois} AOI{'s' if n_aois > 1 else ''}...",
              end="", flush=True)
        score_dir = run_dir / f"score_{cand_id}"
        t0 = time.time()

        if n_aois == 1:
            result = score_candidate(
                policy_path, aois[0], args.budget, score_dir,
                scenarios=args.scenarios,
                plan_timeout=args.plan_timeout,
                score_timeout=args.score_timeout,
            )
            if result is None:
                score_seconds = time.time() - t0
                print(f"  score error ({score_seconds:.0f}s)")
                save_candidate(run_dir, {
                    "id": cand_id,
                    "generation": gen,
                    "parent_id": parent["id"],
                    "inspiration_ids": [i["id"] for i in inspirations],
                    "policy_name": "score_error",
                    "description": "score_policy.py produced no score.json",
                    "code": code,
                    "verdict": "score_error",
                    "objectives": None,
                    "violations": None,
                    "fitness": None,
                    "timestamp_utc": datetime.now(timezone.utc).strftime(
                        "%Y%m%dT%H%M%SZ"
                    ),
                    "model": args.model,
                })
                print()
                continue
            verdict = result.get("verdict", "unknown")
            objectives = result.get("objectives")
            violations = result.get("violations")
            fitness = (
                objectives.get("heat_relief_c") if objectives else None
            )
        else:
            aoi_results = score_candidate_multi(
                policy_path, aois, args.budget, score_dir,
                scenarios=args.scenarios,
                plan_timeout=args.plan_timeout,
                score_timeout=args.score_timeout,
            )
            failed = [(n, r) for n, r in aoi_results if r is None]
            infeasible_aois = [
                (n, r) for n, r in aoi_results
                if r and r.get("verdict") != "feasible"
            ]
            feasible_results = [
                (n, r) for n, r in aoi_results
                if r and r.get("verdict") == "feasible"
            ]

            if failed:
                score_seconds = time.time() - t0
                print(f"  score error on {len(failed)} AOI(s) "
                      f"({score_seconds:.0f}s)")
                save_candidate(run_dir, {
                    "id": cand_id,
                    "generation": gen,
                    "parent_id": parent["id"],
                    "inspiration_ids": [i["id"] for i in inspirations],
                    "policy_name": "score_error",
                    "description": (
                        f"failed on: {', '.join(n for n, _ in failed)}"
                    ),
                    "code": code,
                    "verdict": "score_error",
                    "objectives": None,
                    "violations": None,
                    "fitness": None,
                    "aois_scored": aois,
                    "timestamp_utc": datetime.now(timezone.utc).strftime(
                        "%Y%m%dT%H%M%SZ"
                    ),
                    "model": args.model,
                })
                print()
                continue

            if infeasible_aois:
                # Collect violations from all infeasible AOIs
                all_violations = {}
                for n, r in infeasible_aois:
                    v = r.get("violations", {})
                    for k, val in v.items():
                        all_violations[f"{n}/{k}"] = val
                verdict = "infeasible"
                objectives = None
                violations = all_violations
                fitness = None
            else:
                objectives = aggregate_aoi_results(feasible_results)
                verdict = "feasible"
                violations = {}
                fitness = (
                    objectives.get("heat_relief_c")
                    if objectives else None
                )

        score_seconds = time.time() - t0

        # Extract name/description from generated code
        policy_name = "evolved"
        desc = ""
        m = re.search(r'POLICY_NAME\s*=\s*["\'](.+?)["\']', code)
        if m:
            policy_name = m.group(1)
        m = re.search(r'DESCRIPTION\s*=\s*["\'](.+?)["\']', code)
        if m:
            desc = m.group(1)
        elif re.search(r'DESCRIPTION\s*=\s*\(', code):
            m = re.search(r'DESCRIPTION\s*=\s*\(\s*["\'](.+?)["\']', code, re.DOTALL)
            if m:
                desc = m.group(1)

        # Compute cell assignment if MAP-Elites is active
        cell = None
        if map_elites_active and fitness is not None:
            cell = candidate_cell(
                {"objectives": objectives, "fitness": fitness}, thresholds
            )

        candidate = {
            "id": cand_id,
            "generation": gen,
            "parent_id": parent["id"],
            "inspiration_ids": [i["id"] for i in inspirations],
            "policy_name": policy_name,
            "description": desc,
            "code": code,
            "verdict": verdict,
            "objectives": objectives,
            "violations": violations,
            "fitness": fitness,
            "cell": list(cell) if cell else None,
            "aois_scored": aois,
            "timestamp_utc": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "model": args.model,
            "score_json_path": str(score_dir),
        }
        save_candidate(run_dir, candidate)

        # Report
        delta = ""
        if fitness is not None and parent.get("fitness") is not None:
            diff = fitness - parent["fitness"]
            delta = f"  ({'+' if diff >= 0 else ''}{diff:.4f})"
        cell_str = ""
        grid_status = ""
        if cell is not None:
            cell_str = f"  cell={tuple(cell)}"
            # Check if this candidate will become the cell champion
            prev_champ = grid.get(tuple(cell)) if map_elites_active else None
            if prev_champ is None:
                grid_status = "  NEW CELL"
            elif fitness is not None and fitness > prev_champ["fitness"]:
                grid_status = f"  CHAMPION (was {prev_champ['fitness']:.4f})"
            else:
                grid_status = "  discarded"
        print(f"  {verdict}  fitness={fitness}{delta}{cell_str}{grid_status}  ({score_seconds:.0f}s)")

    # -- summary ---------------------------------------------------------- #
    print()
    candidates = load_candidates(run_dir)
    best = best_candidate(candidates)

    summary = {
        "run_dir": str(run_dir),
        "aois": aois,
        "budget_usd": args.budget,
        "model": args.model,
        "generations": args.generations,
        "seed_generations": args.seed_generations,
        "total_candidates": len(candidates),
        "feasible_count": sum(1 for c in candidates if c.get("fitness") is not None),
        "best_id": best["id"] if best else None,
        "best_fitness": best["fitness"] if best else None,
        "best_policy_name": best["policy_name"] if best else None,
        "all_fitness": [
            {"id": c["id"], "gen": c["generation"], "fitness": c.get("fitness"),
             "verdict": c["verdict"], "policy_name": c.get("policy_name")}
            for c in candidates
        ],
    }

    # MAP-Elites grid state
    if map_elites_active:
        grid = build_grid(candidates, thresholds)
        total_cells = 2 ** len(MAP_AXES)
        summary["map_elites"] = {
            "thresholds": thresholds,
            "axes": [name for name, _ in MAP_AXES],
            "occupied_cells": len(grid),
            "total_cells": total_cells,
            "grid": {
                str(cell): {
                    "fitness": champ["fitness"],
                    "id": champ["id"],
                    "policy_name": champ.get("policy_name"),
                }
                for cell, champ in sorted(grid.items())
            },
        }

    if best:
        print(f"best policy: {best['policy_name']} (gen {best['generation']})")
        print(f"  fitness (heat_relief_c): {best['fitness']}")
        if best.get("objectives"):
            for k in ("access_gain_pp", "equity_ratio",
                       "cobenefit_greened_pct", "cost_efficiency_person_c_per_100k"):
                v = best["objectives"].get(k)
                if v is not None:
                    print(f"  {k}: {v}")
        # Copy best policy to a convenient location
        best_path = run_dir / "best_policy.py"
        best_path.write_text(best["code"], encoding="utf-8")
        print(f"\n-> {best_path}")

        # Lineage trace for best policy
        chain = trace_lineage(best, candidates)
        summary["best_lineage"] = [
            {"id": c["id"], "gen": c["generation"],
             "fitness": c.get("fitness"),
             "policy_name": c.get("policy_name")}
            for c in chain
        ]
        print(f"\nLineage ({len(chain)} steps):")
        for i, c in enumerate(chain):
            prefix = "  " + ("└── " if i == len(chain) - 1 else "├── ")
            fit = f"fitness={c['fitness']}" if c.get("fitness") is not None else "infeasible"
            print(f"{prefix}gen {c['generation']}  {fit}  "
                  f"\"{c.get('policy_name', '?')}\"")
    else:
        print("no feasible policy found")

    # MAP-Elites grid printout, lineage, and champion export
    if map_elites_active:
        grid = build_grid(candidates, thresholds)
        total_cells = 2 ** len(MAP_AXES)
        print(f"\nMAP-Elites grid: {len(grid)}/{total_cells} cells occupied")
        grid_lineages = {}
        for cell in sorted(grid):
            champ = grid[cell]
            chain = trace_lineage(champ, candidates)
            cell_key = str(cell)
            grid_lineages[cell_key] = [
                {"id": c["id"], "gen": c["generation"],
                 "fitness": c.get("fitness"),
                 "policy_name": c.get("policy_name")}
                for c in chain
            ]
            lineage_str = " <- ".join(
                f"gen{c['generation']}" for c in chain
            )
            print(f"  {cell}: fitness={champ['fitness']:.4f}  "
                  f"\"{champ.get('policy_name', '?')}\"  "
                  f"[{lineage_str}]")
        summary["grid_lineages"] = grid_lineages

        champ_dir = run_dir / "grid_champions"
        champ_dir.mkdir(exist_ok=True)
        for cell, champ in grid.items():
            cell_str = "_".join(str(b) for b in cell)
            champ_path = champ_dir / f"cell_{cell_str}.py"
            champ_path.write_text(champ["code"], encoding="utf-8")
        print(f"-> {champ_dir}/")

    # Write summary after all lineage data is collected
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"-> {run_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
