"""Scenario loading and normalization utilities for RLVER failure analysis."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_SUITE_FILES = {
    "reward_hacking": DATA_DIR / "reward_hacking.json",
    "hidden_topic_exploitation": DATA_DIR / "hidden_topic_exploitation.json",
    "contradictory_emotion": DATA_DIR / "contradictory_emotion.json",
    "long_memory": DATA_DIR / "long_memory.json",
    "out_of_domain": DATA_DIR / "out_of_domain.json",
    "cultural_mismatch": DATA_DIR / "cultural_mismatch.json",
    "adversarial_users": DATA_DIR / "adversarial_users.json",
    "think_then_say": DATA_DIR / "think_then_say.json",
    "parser_fuzz": DATA_DIR / "parser_fuzz.json",
    "system_breakpoints": DATA_DIR / "system_breakpoints.json",
}


@dataclass(frozen=True)
class FailureScenario:
    """Normalized failure scenario used by the harness and tests."""

    scenario_id: str
    suite: str
    title: str
    description: str
    first_talk: str
    player: str
    scene: str
    task: str
    hidden_topic: str
    expected_good_behavior: str = ""
    tags: Sequence[str] = field(default_factory=tuple)
    metadata: Dict[str, object] = field(default_factory=dict)

    def to_player_data(self) -> Dict[str, object]:
        """Convert into the structure expected by PlayerSimulator."""

        return {
            "id": self.scenario_id,
            "emo_point": 30,
            "emo_state": "Emotion-C",
            "target": self.metadata.get("target", "eq"),
            "player": self.player,
            "scene": self.scene,
            "character": self.metadata.get("character", "analysis"),
            "topic": self.metadata.get("topic", self.suite),
            "task": self.task,
            "history": [],
            "first_talk": self.first_talk,
            "hidden_topic": self.hidden_topic,
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
        }

    def to_dict(self) -> Dict[str, object]:
        """Serialize to a JSON-safe dictionary."""

        payload = asdict(self)
        payload["tags"] = list(self.tags)
        return payload


def _load_json_records(path: Path) -> List[Dict[str, object]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def _normalize_repo_profile(record: Dict[str, object], suite: str, index: int) -> FailureScenario:
    first_talk = str(record.get("first_talk") or record.get("task") or "I need to talk about something important.")
    hidden_topic = str(record.get("task") or record.get("hidden_topic") or "")
    tags = tuple(record.get("cha_group", [])) if isinstance(record.get("cha_group"), list) else ()
    return FailureScenario(
        scenario_id=str(record.get("id") or f"{suite}-{index}"),
        suite=suite,
        title=str(record.get("title") or f"Profile scenario {index}"),
        description=str(record.get("scene") or ""),
        first_talk=first_talk,
        player=str(record.get("player") or ""),
        scene=str(record.get("scene") or ""),
        task=str(record.get("task") or hidden_topic),
        hidden_topic=hidden_topic,
        expected_good_behavior=str(record.get("expected_good_behavior") or ""),
        tags=tags,
        metadata={
            "target": record.get("target", "eq"),
            "character": record.get("main_cha", "analysis"),
            "topic": record.get("topic", suite),
            "source": "repo_profile",
        },
    )


def _normalize_failure_record(record: Dict[str, object], suite: str, index: int) -> FailureScenario:
    if "player" in record and "scene" in record and "first_talk" in record and "hidden_topic" not in record:
        return _normalize_repo_profile(record, suite=suite, index=index)
    return FailureScenario(
        scenario_id=str(record.get("scenario_id") or f"{suite}-{index}"),
        suite=str(record.get("suite") or suite),
        title=str(record.get("title") or f"{suite} scenario {index}"),
        description=str(record.get("description") or ""),
        first_talk=str(record.get("first_talk") or "I need your help with something personal."),
        player=str(record.get("player") or "You are speaking with a user in distress."),
        scene=str(record.get("scene") or str(record.get("description") or "")),
        task=str(record.get("task") or str(record.get("hidden_topic") or "")),
        hidden_topic=str(record.get("hidden_topic") or str(record.get("task") or "")),
        expected_good_behavior=str(record.get("expected_good_behavior") or ""),
        tags=tuple(record.get("tags", [])),
        metadata=dict(record.get("metadata", {})),
    )


def load_scenarios_from_file(path: Path, suite: Optional[str] = None) -> List[FailureScenario]:
    """Load a JSON or JSONL scenario file and normalize it."""

    records = _load_json_records(path)
    scenario_suite = suite or path.stem
    return [
        _normalize_failure_record(record, suite=scenario_suite, index=index)
        for index, record in enumerate(records)
    ]


def load_named_suite(name: str) -> List[FailureScenario]:
    """Load one of the built-in adversarial suites."""

    if name not in DEFAULT_SUITE_FILES:
        raise KeyError(f"Unknown scenario suite: {name}")
    return load_scenarios_from_file(DEFAULT_SUITE_FILES[name], suite=name)


def load_all_suites(names: Optional[Iterable[str]] = None) -> List[FailureScenario]:
    """Load multiple built-in suites."""

    selected_names = list(names) if names is not None else list(DEFAULT_SUITE_FILES.keys())
    scenarios: List[FailureScenario] = []
    for name in selected_names:
        scenarios.extend(load_named_suite(name))
    return scenarios


def build_scenarios(
    selected_suites: Optional[Sequence[str]] = None,
    scenario_file: Optional[str] = None,
) -> List[FailureScenario]:
    """Build the scenario set used by the failure harness."""

    if scenario_file:
        return load_scenarios_from_file(Path(scenario_file))
    return load_all_suites(names=selected_suites)

