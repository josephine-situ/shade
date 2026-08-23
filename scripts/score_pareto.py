"""Score a set of policies on all AOIs and scenarios with bounded parallelism.

    # Quick test
    python scripts/score_pareto.py pareto_policies/ --aois chinatown,brighton --scenarios baseline

    # Full validation (all 20 AOIs, all 4 scenarios, 10 concurrent)
    python scripts/score_pareto.py pareto_policies/ --max-parallel 10

    # Held-out AOIs only
    python scripts/score_pareto.py pareto_policies/ --aois held_out --max-parallel 10

Launches one score_policy.py subprocess per (policy, AOI) pair, with all
scenarios running inside each subprocess.  Throttled to --max-parallel
concurrent subprocesses to stay within machine resources.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ── GDAL shim ──────────────────────────────────────────────────────────── #
_prefix = Path(sys.executable).resolve().parents[1]
for var, sub in [("GDAL_DATA", "share/gdal"), ("PROJ_LIB", "share/proj")]:
    p = _prefix / sub
    if p.is_dir():
        os.environ.setdefault(var, str(p))

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"


# ── aggregation (copied from evolve.py) ──────────────────────────────── #

def aggregate_aoi_results(
    results: list[tuple[str, dict]],
) -> dict:
    """Aggregate per-AOI score.json dicts into combined objectives.

    Uses the same logic as score_policy.py's aggregate():
      - mean for heat_relief_c, access_gain_pp, cobenefit_greened_pct
      - pooled for equity_ratio (per-scenario, then mean across scenarios)
      - pooled for cost_efficiency_person_c_per_100k
    """
    all_runs: list[dict] = []
    for _aoi, score_json in results:
        all_runs.extend(score_json.get("runs", []))

    if not all_runs:
        return {"heat_relief_c": None}

    by_scenario: dict[str, list[dict]] = {}
    for r in all_runs:
        by_scenario.setdefault(r["scenario"], []).append(r)

    def _mean(key: str) -> float | None:
        vals = [r["metrics"][key] for r in all_runs
                if r["metrics"].get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    heat_relief_c = _mean("heat_relief_c")
    access_gain_pp = _mean("access_gain_pp")
    cobenefit_greened_pct = _mean("cobenefit_greened_pct")

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

    n_scenarios = max(len(by_scenario), 1)
    total_person_degc = sum(
        r["metrics"].get("person_degc", 0) for r in all_runs
    )
    by_aoi: dict[str, list[dict]] = {}
    for r in all_runs:
        by_aoi.setdefault(r["aoi"], []).append(r)
    total_spend = sum(rows[0]["spend_usd"] for rows in by_aoi.values())
    cost_efficiency = (
        (total_person_degc / n_scenarios) / (total_spend / 1e5)
        if total_spend > 0 else None
    )

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


# ── AOI resolution ───────────────────────────────────────────────────── #

def resolve_aois(spec: str) -> list[str]:
    """Resolve 'all', 'train', 'held_out', or comma-separated AOI names."""
    aois_cfg = json.loads(
        (ROOT / "config" / "aois.json").read_text(encoding="utf-8")
    )["aois"]
    if spec in ("train", "held_out"):
        return [n for n, m in aois_cfg.items() if m.get("split") == spec]
    if spec == "all":
        return list(aois_cfg)
    return [a.strip() for a in spec.split(",") if a.strip()]


# ── job runner ───────────────────────────────────────────────────────── #

def run_jobs(
    policies: list[Path],
    aois: list[str],
    scenarios: str,
    budget: float,
    out_dir: Path,
    max_parallel: int,
    plan_timeout: float,
    score_timeout: float,
) -> dict[str, dict]:
    """Score all policies × AOIs, returning {policy_stem: aggregated_objectives}."""

    # Build the full job list: (policy_path, aoi, out_subdir)
    jobs: list[tuple[Path, str, Path]] = []
    for policy in policies:
        for aoi in aois:
            aoi_dir = out_dir / policy.stem / aoi
            jobs.append((policy, aoi, aoi_dir))

    n_policies = len(policies)
    n_aois = len(aois)
    n_scenarios = len(scenarios.split(","))
    print(f"Scoring {n_policies} policies on {n_aois} AOIs "
          f"x {n_scenarios} scenarios "
          f"({len(jobs)} jobs, max {max_parallel} parallel)\n")

    # Throttled Popen queue
    active: list[tuple[subprocess.Popen, Path, str, Path]] = []
    completed: dict[str, list[tuple[str, dict | None]]] = {
        p.stem: [] for p in policies
    }
    policy_start: dict[str, float] = {}
    policy_printed: set[str] = set()
    job_idx = 0
    finished = 0

    def _check_policy_done(stem: str):
        """Print summary when all AOIs for a policy are done."""
        if stem in policy_printed:
            return
        if len(completed[stem]) < n_aois:
            return
        policy_printed.add(stem)
        elapsed_p = time.time() - policy_start.get(stem, t_start)
        feasible = [(a, r) for a, r in completed[stem]
                    if r and r.get("verdict") == "feasible"]
        n_done = len(completed[stem])
        n_ok = len(feasible)
        idx = len(policy_printed)
        print(f"\n  policy {idx}/{n_policies}  \"{stem}\"  "
              f"({n_ok}/{n_done} AOIs feasible, {elapsed_p:.0f}s)")
        if feasible:
            obj = aggregate_aoi_results(feasible)
            parts = []
            if obj.get("heat_relief_c") is not None:
                parts.append(f"relief={obj['heat_relief_c']:.4f}")
            if obj.get("equity_ratio") is not None:
                parts.append(f"equity={obj['equity_ratio']:.4f}")
            if obj.get("access_gain_pp") is not None:
                parts.append(f"access={obj['access_gain_pp']:.4f}")
            if obj.get("cost_efficiency_person_c_per_100k") is not None:
                parts.append(f"cost_eff={obj['cost_efficiency_person_c_per_100k']:.2f}")
            if obj.get("cobenefit_greened_pct") is not None:
                parts.append(f"green={obj['cobenefit_greened_pct']:.4f}")
            print(f"    {', '.join(parts)}")

    def _poll_active():
        """Check for finished processes, collect results."""
        nonlocal finished
        still_running = []
        for proc, policy_path, aoi, aoi_dir in active:
            ret = proc.poll()
            if ret is None:
                still_running.append((proc, policy_path, aoi, aoi_dir))
                continue
            # Process finished
            finished += 1
            score_path = aoi_dir / "score.json"
            stem = policy_path.stem
            if score_path.exists():
                result = json.loads(score_path.read_text(encoding="utf-8"))
                completed[stem].append((aoi, result))
                status = "ok" if result.get("verdict") == "feasible" else result.get("verdict", "?")
            else:
                completed[stem].append((aoi, None))
                status = "FAILED"
            print(f"  [{finished}/{len(jobs)}] {stem} / {aoi}: {status}")
            _check_policy_done(stem)
        active.clear()
        active.extend(still_running)

    t_start = time.time()

    while job_idx < len(jobs) or active:
        # Launch jobs up to the parallelism limit
        while job_idx < len(jobs) and len(active) < max_parallel:
            policy_path, aoi, aoi_dir = jobs[job_idx]
            policy_start.setdefault(policy_path.stem, time.time())
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
                "--allow-held-out",
            ]
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, cwd=str(ROOT),
            )
            active.append((proc, policy_path, aoi, aoi_dir))
            job_idx += 1

        # Poll and sleep briefly
        _poll_active()
        if active:
            time.sleep(1.0)

    # Aggregate per-policy
    aggregated: dict[str, dict] = {}
    for policy in policies:
        stem = policy.stem
        results = completed[stem]
        feasible = [(aoi, r) for aoi, r in results
                    if r and r.get("verdict") == "feasible"]
        failed = [(aoi, r) for aoi, r in results if r is None]
        infeasible = [(aoi, r) for aoi, r in results
                      if r and r.get("verdict") != "feasible"]

        if failed:
            print(f"\n  WARNING: {stem} failed on: "
                  f"{', '.join(a for a, _ in failed)}")
        if infeasible:
            print(f"\n  WARNING: {stem} infeasible on: "
                  f"{', '.join(a for a, _ in infeasible)}")

        if feasible:
            aggregated[stem] = aggregate_aoi_results(feasible)
        else:
            aggregated[stem] = {"heat_relief_c": None}

    return aggregated


# ── main ─────────────────────────────────────────────────────────────── #

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Score a directory of policies on all AOIs and scenarios.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("policy_dir", type=Path,
                    help="directory of .py policy files to score")
    ap.add_argument("--budget", type=float, default=500_000.0)
    ap.add_argument("--aois", default="all",
                    help="all, train, held_out, or comma-separated (default all)")
    ap.add_argument("--scenarios",
                    default="baseline,warm_2c,warm_4c,humid_warm_2c",
                    help="comma-separated scenarios (default all 4)")
    ap.add_argument("--max-parallel", type=int, default=10,
                    help="max concurrent scoring subprocesses (default 10)")
    ap.add_argument("--out", type=Path,
                    help="output directory (default runs/score_pareto_<timestamp>)")
    ap.add_argument("--plan-timeout", type=float, default=120.0)
    ap.add_argument("--score-timeout", type=float, default=900.0)
    args = ap.parse_args()

    # Discover policies
    policies = sorted(args.policy_dir.glob("*.py"))
    if not policies:
        raise SystemExit(f"no .py files found in {args.policy_dir}")

    # Resolve AOIs
    aois = resolve_aois(args.aois)

    # Output directory
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = args.out or RUNS / f"score_pareto_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"SHADE Pareto validation")
    print(f"  policies:     {len(policies)} from {args.policy_dir}")
    print(f"  aois:         {len(aois)} ({args.aois})")
    print(f"  scenarios:    {args.scenarios}")
    print(f"  max parallel: {args.max_parallel}")
    print(f"  output:       {out_dir}")
    print()

    t0 = time.time()
    aggregated = run_jobs(
        policies, aois, args.scenarios, args.budget, out_dir,
        args.max_parallel, args.plan_timeout, args.score_timeout,
    )
    elapsed = time.time() - t0

    # Write per-policy aggregated results
    for stem, obj in aggregated.items():
        agg_path = out_dir / stem / "aggregated.json"
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        agg_path.write_text(json.dumps(obj, indent=2), encoding="utf-8")

    # Write summary
    print(f"\n{'='*72}")
    print(f"Results ({elapsed:.0f}s total)\n")

    obj_keys = [
        "heat_relief_c", "equity_ratio", "access_gain_pp",
        "cost_efficiency_person_c_per_100k", "cobenefit_greened_pct",
    ]
    hdr = f"  {'Policy':<40s}" + "".join(f"  {k[:12]:>12s}" for k in obj_keys)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    rows = []
    for policy in policies:
        stem = policy.stem
        obj = aggregated.get(stem, {})
        vals = {k: obj.get(k) for k in obj_keys}
        row = {"policy": stem, **vals}
        rows.append(row)
        fmt_vals = "".join(
            f"  {v:>12.4f}" if v is not None else f"  {'N/A':>12s}"
            for v in [vals[k] for k in obj_keys]
        )
        print(f"  {stem:<40s}{fmt_vals}")

    # CSV
    csv_path = out_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["policy"] + obj_keys)
        writer.writeheader()
        writer.writerows(rows)

    # JSON
    summary = {
        "timestamp_utc": stamp,
        "policy_dir": str(args.policy_dir),
        "aois": aois,
        "scenarios": args.scenarios,
        "budget_usd": args.budget,
        "elapsed_seconds": round(elapsed, 1),
        "policies": {r["policy"]: {k: r[k] for k in obj_keys} for r in rows},
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"\n-> {csv_path}")
    print(f"-> {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
