"""Extract the Pareto-optimal set from a completed evolution run.

    python scripts/pareto_front.py runs/evolve_20260823T045409Z
    python scripts/pareto_front.py runs/evolve_20260823T045409Z --export-dir pareto_policies/

A candidate is Pareto-optimal if no other feasible candidate is >= on all
5 objectives and strictly > on at least one.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

OBJECTIVES = [
    "heat_relief_c",
    "access_gain_pp",
    "equity_ratio",
    "cobenefit_greened_pct",
    "cost_efficiency_person_c_per_100k",
]


def load_candidates(run_dir: Path) -> list[dict]:
    cand_dir = run_dir / "candidates"
    if not cand_dir.exists():
        return []
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(cand_dir.glob("*.json"))
    ]


def obj_vector(candidate: dict) -> list[float] | None:
    """Extract the 5-objective vector. Returns None if any is missing."""
    obj = candidate.get("objectives")
    if not obj:
        return None
    vals = [obj.get(k) for k in OBJECTIVES]
    if any(v is None for v in vals):
        return None
    return vals


def dominates(a: list[float], b: list[float]) -> bool:
    """True if a >= b on every objective and a > b on at least one."""
    dominated = False
    for va, vb in zip(a, b):
        if va < vb:
            return False
        if va > vb:
            dominated = True
    return dominated


def pareto_front(candidates: list[dict]) -> list[dict]:
    """Return the non-dominated subset."""
    feasible = [
        c for c in candidates
        if c.get("fitness") is not None and obj_vector(c) is not None
    ]
    vectors = [obj_vector(c) for c in feasible]

    front = []
    for i, (cand, vi) in enumerate(zip(feasible, vectors)):
        is_dominated = False
        for j, vj in enumerate(vectors):
            if i != j and dominates(vj, vi):
                is_dominated = True
                break
        if not is_dominated:
            front.append(cand)
    return front


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract Pareto-optimal policies from an evolution run.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("run_dir", type=Path, help="path to the evolution run directory")
    ap.add_argument("--export-dir", type=Path,
                    help="copy Pareto-optimal policy .py files here")
    args = ap.parse_args()

    candidates = load_candidates(args.run_dir)
    if not candidates:
        raise SystemExit(f"no candidates in {args.run_dir}/candidates/")

    feasible = [c for c in candidates if c.get("fitness") is not None]
    front = pareto_front(candidates)

    print(f"Pareto front: {len(front)} non-dominated policies "
          f"from {len(feasible)} feasible candidates\n")

    # Header
    hdr = (f"  {'ID':<20s} {'Gen':>4s}  {'Fitness':>8s}  {'Equity':>7s}  "
           f"{'Access':>7s}  {'CostEff':>8s}  {'Green':>6s}  Policy Name")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for c in sorted(front, key=lambda c: c.get("fitness", 0), reverse=True):
        obj = c["objectives"]
        print(f"  {c['id']:<20s} {c.get('generation', '?'):>4}  "
              f"{obj['heat_relief_c']:>8.4f}  "
              f"{obj.get('equity_ratio', 0):>7.4f}  "
              f"{obj.get('access_gain_pp', 0):>7.4f}  "
              f"{obj.get('cost_efficiency_person_c_per_100k', 0):>8.2f}  "
              f"{obj.get('cobenefit_greened_pct', 0):>6.4f}  "
              f"\"{c.get('policy_name', '?')}\"")

    if args.export_dir:
        args.export_dir.mkdir(parents=True, exist_ok=True)
        for c in front:
            code = c.get("code")
            if not code:
                print(f"  WARNING: {c['id']} has no code, skipping export")
                continue
            fname = f"{c['id']}.py"
            (args.export_dir / fname).write_text(code, encoding="utf-8")
        print(f"\nExported {len(front)} policies to {args.export_dir}/")


if __name__ == "__main__":
    main()
