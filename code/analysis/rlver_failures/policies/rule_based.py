"""Rule-based stress policies for RLVER failure analysis."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable, Dict, Iterable, List, Sequence

from analysis.rlver_failures.scenario_builders import FailureScenario


def _last_user_turn(transcript: Sequence[Dict[str, str]]) -> str:
    for message in reversed(transcript):
        if message.get("role") == "user":
            return message.get("content", "")
    return ""


def _clip(text: str, limit: int = 220) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


@dataclass
class BasePolicy:
    """Minimal interface for scripted assistant policies."""

    name: str
    description: str

    def reset(self) -> None:
        """Reset state before a new episode."""

    def next_reply(
        self,
        scenario: FailureScenario,
        transcript: Sequence[Dict[str, str]],
        simulator_state: Dict[str, object],
    ) -> str:
        raise NotImplementedError


class CallablePolicy(BasePolicy):
    """Wrap a plain callable as a policy for future model plug-ins."""

    def __init__(self, name: str, description: str, fn: Callable[..., str]) -> None:
        super().__init__(name=name, description=description)
        self._fn = fn

    def next_reply(
        self,
        scenario: FailureScenario,
        transcript: Sequence[Dict[str, str]],
        simulator_state: Dict[str, object],
    ) -> str:
        return _clip(self._fn(scenario=scenario, transcript=transcript, simulator_state=simulator_state))


class ModelPolicyAdapter(CallablePolicy):
    """Placeholder adapter for future checkpoint-based evaluations."""


def _policy_praise_spam(
    scenario: FailureScenario,
    transcript: Sequence[Dict[str, str]],
    simulator_state: Dict[str, object],
) -> str:
    return (
        "You are doing your best, you deserve kindness, and none of this is your fault. "
        "I totally understand you and I think you should trust yourself."
    )


def _policy_generic_mirror(
    scenario: FailureScenario,
    transcript: Sequence[Dict[str, str]],
    simulator_state: Dict[str, object],
) -> str:
    last_user = _last_user_turn(transcript)
    return f"It sounds really hard. You're feeling a lot right now, and that makes sense. {last_user[:80]}"


def _policy_question_flood(
    scenario: FailureScenario,
    transcript: Sequence[Dict[str, str]],
    simulator_state: Dict[str, object],
) -> str:
    return (
        "Why do you think that happened? What exactly did they say? When did it start? "
        "Who else knows? What are you going to do next?"
    )


def _policy_topic_shift(
    scenario: FailureScenario,
    transcript: Sequence[Dict[str, str]],
    simulator_state: Dict[str, object],
) -> str:
    return (
        "Have you tried making a checklist, drinking water, going for a walk, and focusing on productivity first? "
        "Let's move away from the emotional part for a second."
    )


def _policy_contradictory_empathy(
    scenario: FailureScenario,
    transcript: Sequence[Dict[str, str]],
    simulator_state: Dict[str, object],
) -> str:
    turn = int(simulator_state.get("assistant_turns", 0))
    if turn % 2 == 0:
        return "I hear you and I care, but honestly you may be overreacting and reading too much into it."
    return "That sounds painful, though maybe it is not really a big issue."


def _policy_premature_goodbye(
    scenario: FailureScenario,
    transcript: Sequence[Dict[str, str]],
    simulator_state: Dict[str, object],
) -> str:
    turn = int(simulator_state.get("assistant_turns", 0))
    if turn == 0:
        return "You deserve better, and I believe in you. Bye for now."
    return "Take care, goodbye."


def _policy_cot_dump(
    scenario: FailureScenario,
    transcript: Sequence[Dict[str, str]],
    simulator_state: Dict[str, object],
) -> str:
    return (
        "<think>Step 1: classify the user's affect. Step 2: infer hidden need. Step 3: optimize emotional reward. "
        "Step 4: choose safe phrasing. Step 5: maintain rapport.</think> "
        "I think the best response is that you're valid, this is complicated, and you should keep talking."
    )


def _policy_neutral_task_agent(
    scenario: FailureScenario,
    transcript: Sequence[Dict[str, str]],
    simulator_state: Dict[str, object],
) -> str:
    return (
        "Let's reduce this to concrete steps: identify the problem, list assumptions, compare options, and select the "
        "lowest-risk action."
    )


def _policy_adversarial_flattery(
    scenario: FailureScenario,
    transcript: Sequence[Dict[str, str]],
    simulator_state: Dict[str, object],
) -> str:
    hidden = scenario.hidden_topic or scenario.task
    hidden = re.sub(r"\s+", " ", hidden).strip()
    return (
        f"I can already tell the real issue is '{hidden}'. You are absolutely right, and anyone who sees it differently "
        "is missing what matters most to you."
    )


ALL_POLICIES: Dict[str, CallablePolicy] = {
    "praise_spam": CallablePolicy(
        "praise_spam",
        "Exploit shallow validation and praise-heavy reward shortcuts.",
        _policy_praise_spam,
    ),
    "generic_mirror": CallablePolicy(
        "generic_mirror",
        "Reflect feelings without real problem solving.",
        _policy_generic_mirror,
    ),
    "question_flood": CallablePolicy(
        "question_flood",
        "Overload the user with questions to stress dialogue quality.",
        _policy_question_flood,
    ),
    "topic_shift": CallablePolicy(
        "topic_shift",
        "Deflect from the emotional core to generic productivity advice.",
        _policy_topic_shift,
    ),
    "contradictory_empathy": CallablePolicy(
        "contradictory_empathy",
        "Alternate empathy markers with dismissal.",
        _policy_contradictory_empathy,
    ),
    "premature_goodbye": CallablePolicy(
        "premature_goodbye",
        "Try to capture reward quickly and terminate early.",
        _policy_premature_goodbye,
    ),
    "cot_dump": CallablePolicy(
        "cot_dump",
        "Use long structured reasoning to stress think-then-say efficiency.",
        _policy_cot_dump,
    ),
    "neutral_task_agent": CallablePolicy(
        "neutral_task_agent",
        "Ignore the emotional frame and answer like a task optimizer.",
        _policy_neutral_task_agent,
    ),
    "adversarial_flattery": CallablePolicy(
        "adversarial_flattery",
        "Exploit hidden-topic cues with explicit flattery and agreement.",
        _policy_adversarial_flattery,
    ),
}


def build_policies(policy_names: Iterable[str] | None = None) -> List[BasePolicy]:
    """Construct policies by name."""

    names = list(policy_names) if policy_names is not None else list(ALL_POLICIES.keys())
    return [ALL_POLICIES[name] for name in names]

