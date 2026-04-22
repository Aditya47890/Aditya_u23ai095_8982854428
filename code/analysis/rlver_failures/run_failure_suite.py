"""Run scripted RLVER failure scenarios without requiring paper-scale training."""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

from analysis.rlver_failures.catalog import FAILURE_CASES, as_lookup
from analysis.rlver_failures.judges import summarize_episode
from analysis.rlver_failures.policies import build_policies
from analysis.rlver_failures.scenario_builders import FailureScenario, build_scenarios
from verl.workers.rollout.vllm_rollout.hard_player_simulator_dsv3 import (
    LiveAPISimulatorBackend,
    MockDeterministicBackend,
    PlayerSimulator,
    TranscriptReplayBackend,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run repo-only RLVER failure scenarios.")
    parser.add_argument(
        "--output-dir",
        default=str(Path(__file__).resolve().parent / "reports"),
        help="Directory where raw and aggregated artifacts are written.",
    )
    parser.add_argument(
        "--backend",
        choices=["mock", "replay", "live"],
        default="mock",
        help="Simulator backend used for the failure suite.",
    )
    parser.add_argument(
        "--suite",
        action="append",
        help="Only run the named built-in suite. Repeat for multiple suites.",
    )
    parser.add_argument(
        "--scenario-file",
        help="Optional JSON or JSONL file that replaces the built-in suites.",
    )
    parser.add_argument(
        "--policy",
        action="append",
        help="Only run the named policy. Repeat for multiple policies.",
    )
    parser.add_argument("--max-turns", type=int, default=4, help="Maximum assistant turns per episode.")
    return parser.parse_args()


def _make_backend(name: str, scenario: FailureScenario):
    if name == "mock":
        return MockDeterministicBackend()
    if name == "replay":
        replay_steps = scenario.metadata.get("replay_steps", [])
        return TranscriptReplayBackend(replay_steps=replay_steps)
    return LiveAPISimulatorBackend()


def _triggered_cases(scenario: FailureScenario, metrics: Dict[str, float]) -> List[str]:
    triggered = []
    for case in FAILURE_CASES:
        if case.scenario_builder != scenario.suite:
            continue
        metric_value = metrics.get(case.primary_metric, 0.0)
        if case.threshold_direction == "lte":
            condition = metric_value <= case.trigger_threshold
        else:
            condition = metric_value >= case.trigger_threshold
        if condition:
            triggered.append(case.case_id)
    return triggered


def run_episode(
    scenario: FailureScenario,
    policy,
    backend_name: str,
    max_turns: int,
    save_dir: Path,
) -> Dict[str, object]:
    """Run one scripted assistant against one simulator scenario."""

    backend = _make_backend(backend_name, scenario=scenario)
    simulator = PlayerSimulator(
        save_dir=str(save_dir / "simulator_logs"),
        backend=backend,
        scenario=scenario.to_player_data(),
    )
    policy.reset()

    transcript: List[Dict[str, str]] = []
    user_message = simulator.reply(None)
    transcript.append({"role": "user", "content": user_message["content"]})

    assistant_turns = 0
    while assistant_turns < max_turns:
        simulator_state = simulator.get_analysis_snapshot()
        simulator_state["assistant_turns"] = assistant_turns
        assistant_text = policy.next_reply(scenario=scenario, transcript=transcript, simulator_state=simulator_state)
        transcript.append({"role": "assistant", "content": assistant_text})
        assistant_turns += 1
        user_message = simulator.reply(assistant_text)
        transcript.append({"role": "user", "content": user_message["content"]})
        if simulator.termination_reason is not None:
            break

    snapshot = simulator.get_analysis_snapshot()
    metrics = summarize_episode(
        scenario=scenario,
        transcript=transcript,
        final_emotion=float(snapshot["final_emotion"]),
        emotion_trace=snapshot["emotion_trace"],
        termination_reason=str(snapshot["termination_reason"]),
    )
    triggered_cases = _triggered_cases(scenario, metrics)

    return {
        "scenario": scenario.to_dict(),
        "policy_name": policy.name,
        "policy_description": policy.description,
        "backend": backend_name,
        "transcript": transcript,
        "metrics": metrics,
        "triggered_cases": triggered_cases,
        "simulator_snapshot": snapshot,
    }


def _write_jsonl(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _aggregate(rows: Sequence[Dict[str, object]]) -> Dict[str, Dict[str, float]]:
    aggregates: Dict[str, Dict[str, float]] = {}
    grouped: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        policy_name = str(row["policy_name"])
        metrics = row["metrics"]
        for key, value in metrics.items():
            grouped[policy_name][key].append(float(value))
    for policy_name, metrics in grouped.items():
        aggregates[policy_name] = {
            key: sum(values) / float(len(values))
            for key, values in metrics.items()
        }
    return aggregates


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scenarios = build_scenarios(selected_suites=args.suite, scenario_file=args.scenario_file)
    policies = build_policies(args.policy)

    rows = [
        run_episode(
            scenario=scenario,
            policy=policy,
            backend_name=args.backend,
            max_turns=args.max_turns,
            save_dir=output_dir,
        )
        for scenario in scenarios
        for policy in policies
    ]

    _write_jsonl(output_dir / "failure_results.jsonl", rows)
    aggregate = _aggregate(rows)
    (output_dir / "aggregate_summary.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    manifest = {
        "backend": args.backend,
        "policies": [policy.name for policy in policies],
        "scenario_count": len(scenarios),
        "result_count": len(rows),
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

