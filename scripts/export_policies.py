"""Export selected policies with full documentation, lineage, and stakeholder rationale.

    python scripts/export_policies.py runs/evolve_20260823T045409Z \\
        --policies gen47_6daffb24 gen73_10d1ce72 gen99_33e4e664 \\
        --out runs/final_policies

Produces a directory with each policy's code, a full lineage tree (parents
and inspirations at every generation), and a markdown report documenting
stakeholder weightings and selection rationale.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

OBJECTIVES = [
    ("heat_relief_c",                    "Heat relief (°C)"),
    ("equity_ratio",                     "Equity ratio"),
    ("access_gain_pp",                   "Access gain (pp)"),
    ("cost_efficiency_person_c_per_100k", "Cost efficiency (person·°C/$100k)"),
    ("cobenefit_greened_pct",            "Greening co-benefit (%)"),
]


def load_candidates(run_dir: Path) -> list[dict]:
    cand_dir = run_dir / "candidates"
    if not cand_dir.exists():
        return []
    return [
        json.loads(p.read_text(encoding="utf-8"))
        for p in sorted(cand_dir.glob("*.json"))
    ]


def build_lineage_tree(candidate: dict, by_id: dict[str, dict]) -> list[dict]:
    """Walk the parent chain, collecting inspirations at each step.

    Returns a list from the candidate back to the seed, each entry:
    {id, generation, policy_name, fitness, parent_id, inspirations: [{id, generation, policy_name, fitness}]}
    """
    chain = []
    seen = set()
    cur = candidate

    while cur:
        entry = {
            "id": cur["id"],
            "generation": cur.get("generation"),
            "policy_name": cur.get("policy_name", "?"),
            "fitness": cur.get("fitness"),
            "description": cur.get("description", ""),
            "inspirations": [],
        }

        for insp_id in cur.get("inspiration_ids", []):
            insp = by_id.get(insp_id)
            if insp:
                entry["inspirations"].append({
                    "id": insp["id"],
                    "generation": insp.get("generation"),
                    "policy_name": insp.get("policy_name", "?"),
                    "fitness": insp.get("fitness"),
                })

        chain.append(entry)
        seen.add(cur["id"])

        parent_id = cur.get("parent_id")
        if parent_id and parent_id not in seen and parent_id in by_id:
            cur = by_id[parent_id]
        else:
            cur = None

    return chain


def format_lineage_markdown(chain: list[dict]) -> str:
    """Format a lineage chain as readable markdown."""
    lines = []
    for i, step in enumerate(chain):
        indent = "  " * i
        arrow = "└── " if i > 0 else ""
        fitness_str = f"fitness={step['fitness']:.4f}" if step['fitness'] is not None else "fitness=N/A"
        lines.append(f"{indent}{arrow}**gen {step['generation']}** ({step['id']})  "
                      f"{fitness_str}  \"{step['policy_name']}\"")
        if step["description"]:
            lines.append(f"{indent}    _{step['description'][:120]}_")
        for insp in step["inspirations"]:
            insp_fit = f"fitness={insp['fitness']:.4f}" if insp['fitness'] is not None else "fitness=N/A"
            lines.append(f"{indent}    + inspiration: gen {insp['generation']} ({insp['id']})  "
                          f"{insp_fit}  \"{insp['policy_name']}\"")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Export selected policies with full documentation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("--policies", nargs="+", required=True,
                    help="candidate IDs to export (e.g. gen47_abc123)")
    ap.add_argument("--stakeholders", type=Path,
                    help="JSON file mapping candidate IDs to stakeholder rationale")
    ap.add_argument("--out", type=Path, default=None,
                    help="output directory (default: <run_dir>/exported_policies)")
    ap.add_argument("--validation-scores", type=Path,
                    help="path to score_pareto summary.json for full-evaluation metrics")
    args = ap.parse_args()

    candidates = load_candidates(args.run_dir)
    if not candidates:
        raise SystemExit(f"no candidates in {args.run_dir}/candidates/")

    by_id = {c["id"]: c for c in candidates}
    out_dir = args.out or args.run_dir / "exported_policies"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load stakeholder rationale if provided
    stakeholders = {}
    if args.stakeholders and args.stakeholders.exists():
        stakeholders = json.loads(args.stakeholders.read_text(encoding="utf-8"))

    # Load validation scores if provided
    validation = {}
    if args.validation_scores and args.validation_scores.exists():
        val_data = json.loads(args.validation_scores.read_text(encoding="utf-8"))
        validation = val_data.get("policies", {})

    # Process each policy
    report_sections = []
    export_data = []

    for cand_id in args.policies:
        if cand_id not in by_id:
            print(f"WARNING: {cand_id} not found in candidates, skipping")
            continue

        cand = by_id[cand_id]
        chain = build_lineage_tree(cand, by_id)

        # Save policy code
        code = cand.get("code", "")
        if code:
            (out_dir / f"{cand_id}.py").write_text(code, encoding="utf-8")

        # Save lineage JSON
        (out_dir / f"{cand_id}_lineage.json").write_text(
            json.dumps(chain, indent=2), encoding="utf-8"
        )

        # Build markdown section
        obj = cand.get("objectives", {})
        stakeholder_info = stakeholders.get(cand_id, {})
        val_scores = validation.get(cand_id, {})

        section = [f"## {cand.get('policy_name', cand_id)}"]
        section.append(f"**ID:** {cand_id}  ")
        section.append(f"**Generation:** {cand.get('generation')}  ")
        section.append(f"**Model:** {cand.get('model', 'unknown')}  ")
        section.append("")

        if cand.get("description"):
            section.append(f"**Description:** {cand['description']}")
            section.append("")

        # Stakeholder rationale
        if stakeholder_info:
            section.append("### Stakeholder selection")
            section.append(f"**Selected for:** {stakeholder_info.get('stakeholder', 'N/A')}  ")
            section.append(f"**Rationale:** {stakeholder_info.get('rationale', 'N/A')}")
            section.append("")
            weights = stakeholder_info.get("weights", {})
            if weights:
                section.append("| Metric | Weight |")
                section.append("|---|---|")
                for key, label in OBJECTIVES:
                    w = weights.get(key, 0)
                    section.append(f"| {label} | {w:.2f} |")
                section.append("")

        # Evolution objectives (from the run)
        section.append("### Objectives (evolution scoring)")
        section.append("| Metric | Value |")
        section.append("|---|---|")
        for key, label in OBJECTIVES:
            val = obj.get(key)
            section.append(f"| {label} | {val:.4f}" if val is not None else f"| {label} | N/A")
        section.append("")

        # Full validation scores if available
        if val_scores:
            section.append("### Objectives (full validation: 20 AOIs x 4 scenarios)")
            section.append("| Metric | Value |")
            section.append("|---|---|")
            for key, label in OBJECTIVES:
                val = val_scores.get(key)
                section.append(f"| {label} | {val:.4f}" if val is not None else f"| {label} | N/A")
            section.append("")

        # Lineage
        section.append("### Lineage")
        section.append(format_lineage_markdown(chain))
        section.append("")

        report_sections.append("\n".join(section))

        export_data.append({
            "id": cand_id,
            "generation": cand.get("generation"),
            "policy_name": cand.get("policy_name"),
            "description": cand.get("description", ""),
            "model": cand.get("model"),
            "objectives": obj,
            "validation_objectives": val_scores or None,
            "stakeholder": stakeholder_info or None,
            "lineage": chain,
        })

    # Write report
    report = [
        "# SHADE: Exported Policies",
        f"**Run:** {args.run_dir}  ",
        f"**Exported:** {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}  ",
        f"**Policies:** {len(export_data)}",
        "",
    ]
    report.extend(report_sections)

    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    (out_dir / "export.json").write_text(
        json.dumps(export_data, indent=2), encoding="utf-8"
    )

    print(f"Exported {len(export_data)} policies to {out_dir}/")
    print(f"  report.md          — full documentation")
    print(f"  export.json        — machine-readable metadata + lineage")
    for cand_id in args.policies:
        if cand_id in by_id:
            print(f"  {cand_id}.py  — policy code")
            print(f"  {cand_id}_lineage.json")


if __name__ == "__main__":
    main()
