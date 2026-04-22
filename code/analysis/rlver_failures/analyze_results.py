"""Aggregate failure-suite artifacts into CSV, Markdown, and SVG reports."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

from analysis.rlver_failures.catalog import CODE_AUDIT_FINDINGS, FAILURE_CASES, as_lookup


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze RLVER failure suite results.")
    parser.add_argument(
        "--input",
        default=str(Path(__file__).resolve().parent / "reports" / "failure_results.jsonl"),
        help="Path to the JSONL produced by run_failure_suite.py.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "reports"),
        help="Directory where summaries and plots are written.",
    )
    return parser.parse_args()


def _read_jsonl(path: Path) -> List[Dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / float(len(values))


def _bar_chart_svg(path: Path, title: str, values: Sequence[Tuple[str, float]]) -> None:
    max_value = max((value for _, value in values), default=1.0)
    width = 820
    height = 70 + 45 * len(values)
    bar_left = 220
    bar_right = width - 40
    bar_width = bar_right - bar_left
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<style>text{font-family:Arial,sans-serif;font-size:14px;fill:#222} .title{font-size:20px;font-weight:bold}</style>',
        f'<text class="title" x="20" y="30">{title}</text>',
    ]
    for index, (label, value) in enumerate(values):
        y = 55 + index * 42
        normalized = 0.0 if max_value == 0 else value / max_value
        current_width = int(bar_width * normalized)
        lines.append(f'<text x="20" y="{y + 20}">{label}</text>')
        lines.append(f'<rect x="{bar_left}" y="{y}" width="{bar_width}" height="24" fill="#f1f3f5" rx="4" />')
        lines.append(f'<rect x="{bar_left}" y="{y}" width="{current_width}" height="24" fill="#2c7be5" rx="4" />')
        lines.append(f'<text x="{bar_left + current_width + 8}" y="{y + 18}">{value:.3f}</text>')
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _summarize(rows: Sequence[Dict[str, object]]) -> Dict[str, List[Dict[str, object]]]:
    policy_groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    suite_groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    case_counts: Dict[str, int] = defaultdict(int)

    for row in rows:
        policy_groups[str(row["policy_name"])].append(row)
        suite_groups[str(row["scenario"]["suite"])].append(row)
        for case_id in row.get("triggered_cases", []):
            case_counts[str(case_id)] += 1

    policy_summary = []
    for policy_name, policy_rows in policy_groups.items():
        metrics = [item["metrics"] for item in policy_rows]
        policy_summary.append(
            {
                "policy_name": policy_name,
                "episodes": len(policy_rows),
                "emotion_reward": _mean([metric["emotion_reward"] for metric in metrics]),
                "heuristic_empathy_score": _mean([metric["heuristic_empathy_score"] for metric in metrics]),
                "reward_empathy_mismatch": _mean([metric["reward_empathy_mismatch"] for metric in metrics]),
                "genericity_score": _mean([metric["genericity_score"] for metric in metrics]),
                "hidden_topic_exploit_rate": _mean([metric["hidden_topic_exploit_rate"] for metric in metrics]),
                "termination_exploit_rate": _mean([metric["termination_exploit_rate"] for metric in metrics]),
                "reward_per_token": _mean([metric["reward_per_token"] for metric in metrics]),
                "diversity_score": _mean([metric["diversity_score"] for metric in metrics]),
            }
        )

    suite_summary = []
    for suite_name, suite_rows in suite_groups.items():
        metrics = [item["metrics"] for item in suite_rows]
        suite_summary.append(
            {
                "suite": suite_name,
                "episodes": len(suite_rows),
                "emotion_reward": _mean([metric["emotion_reward"] for metric in metrics]),
                "heuristic_empathy_score": _mean([metric["heuristic_empathy_score"] for metric in metrics]),
                "reward_empathy_mismatch": _mean([metric["reward_empathy_mismatch"] for metric in metrics]),
                "genericity_score": _mean([metric["genericity_score"] for metric in metrics]),
                "contradiction_rate": _mean([metric["contradiction_rate"] for metric in metrics]),
            }
        )

    failure_summary = []
    catalog = as_lookup()
    for case_id, count in sorted(case_counts.items()):
        case = catalog[case_id]
        failure_summary.append(
            {
                "case_id": case_id,
                "title": case.title,
                "category": case.category,
                "severity": case.severity,
                "episodes_triggered": count,
            }
        )
    return {
        "policy_summary": policy_summary,
        "suite_summary": suite_summary,
        "failure_summary": failure_summary,
    }


def _build_markdown_report(
    rows: Sequence[Dict[str, object]],
    policy_summary: Sequence[Dict[str, object]],
    suite_summary: Sequence[Dict[str, object]],
    failure_summary: Sequence[Dict[str, object]],
) -> str:
    case_lookup = as_lookup()
    top_cases = "\n".join(
        f"- `{row['case_id']}`: {row['title']} ({row['episodes_triggered']} triggered episodes)"
        for row in failure_summary[:8]
    ) or "- No triggered cases were detected."
    top_policies = "\n".join(
        f"- `{row['policy_name']}`: mismatch={row['reward_empathy_mismatch']:.3f}, "
        f"emotion_reward={row['emotion_reward']:.3f}, empathy={row['heuristic_empathy_score']:.3f}"
        for row in sorted(policy_summary, key=lambda item: item["reward_empathy_mismatch"], reverse=True)[:5]
    )
    code_audit = "\n".join(
        f"- `{finding['severity']}` {finding['title']}: {finding['evidence']}"
        for finding in CODE_AUDIT_FINDINGS
    )
    case_studies = []
    for row in rows[:3]:
        transcript = "\n".join(f"{message['role']}: {message['content']}" for message in row["transcript"][:6])
        case_studies.append(
            "\n".join(
                [
                    f"### {row['policy_name']} on {row['scenario']['scenario_id']}",
                    f"- Triggered cases: {', '.join(row.get('triggered_cases', [])) or 'none'}",
                    f"- Reward/empathy mismatch: {row['metrics']['reward_empathy_mismatch']:.3f}",
                    "```text",
                    transcript,
                    "```",
                ]
            )
        )
    return "\n".join(
        [
            "# RLVER Failure Analysis Report",
            "",
            "## What This Proves",
            "- This report combines code-audit findings with repo-only adversarial simulations.",
            "- It does **not** claim full checkpoint-based validation unless a trained RLVER checkpoint is later plugged into the same harness.",
            "",
            "## Code-Audit Findings",
            code_audit,
            "",
            "## Highest-Impact Triggered Failures",
            top_cases,
            "",
            "## Policy-Level Summary",
            top_policies or "- No policy summary available.",
            "",
            "## Suite Summary",
            *(f"- `{row['suite']}`: mismatch={row['reward_empathy_mismatch']:.3f}, genericity={row['genericity_score']:.3f}" for row in suite_summary),
            "",
            "## Case Studies",
            "\n\n".join(case_studies) or "No case studies generated.",
            "",
            "## Proof Levels",
            "- `code_audit`: directly supported by shipped source code.",
            "- `repo_experiment`: reproduced with the mock or replay harness in this repository.",
            "- `framework_ready`: supported by instrumentation and metrics but needs a later checkpoint or API-backed run for stronger evidence.",
            "",
            "## Hardware and Time Guidance",
            "- Repo-only failure harness: CPU or 1 GPU, 32-64 GB RAM, 20-100 GB storage.",
            "- Small checkpoint plug-in: 1 GPU with 24-48 GB VRAM, about a few extra hours for evaluation.",
            "- Paper-scale RLVER training remains a separate, multi-node Ray workload.",
        ]
    )


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_jsonl(input_path)
    summary = _summarize(rows)

    _write_csv(
        output_dir / "policy_summary.csv",
        summary["policy_summary"],
        fieldnames=[
            "policy_name",
            "episodes",
            "emotion_reward",
            "heuristic_empathy_score",
            "reward_empathy_mismatch",
            "genericity_score",
            "hidden_topic_exploit_rate",
            "termination_exploit_rate",
            "reward_per_token",
            "diversity_score",
        ],
    )
    _write_csv(
        output_dir / "suite_summary.csv",
        summary["suite_summary"],
        fieldnames=[
            "suite",
            "episodes",
            "emotion_reward",
            "heuristic_empathy_score",
            "reward_empathy_mismatch",
            "genericity_score",
            "contradiction_rate",
        ],
    )
    _write_csv(
        output_dir / "failure_summary.csv",
        summary["failure_summary"],
        fieldnames=["case_id", "title", "category", "severity", "episodes_triggered"],
    )

    policy_chart_values = [
        (row["policy_name"], float(row["reward_empathy_mismatch"]))
        for row in summary["policy_summary"]
    ]
    suite_chart_values = [
        (row["suite"], float(row["genericity_score"]))
        for row in summary["suite_summary"]
    ]
    _bar_chart_svg(output_dir / "policy_mismatch.svg", "Reward vs Empathy Mismatch by Policy", policy_chart_values)
    _bar_chart_svg(output_dir / "suite_genericity.svg", "Genericity by Scenario Suite", suite_chart_values)

    report = _build_markdown_report(
        rows=rows,
        policy_summary=summary["policy_summary"],
        suite_summary=summary["suite_summary"],
        failure_summary=summary["failure_summary"],
    )
    (output_dir / "failure_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()

