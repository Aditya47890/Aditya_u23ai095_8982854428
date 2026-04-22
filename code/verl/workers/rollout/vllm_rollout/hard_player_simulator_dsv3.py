import copy
import json
import logging
import os
import random
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from verl.workers.rollout.vllm_rollout.system_prompt import *  # noqa: F401,F403


LOGGER = logging.getLogger(__name__)

target_prompt = {
    "no-target": "Your dialogue purpose is to chat casually with the NPC based on the character profile and dialogue background. You should wait for the NPC to bring up topics, then respond according to your interests. You do not need to proactively bring up or change topics. You should conduct the dialogue based on the level of conversational interest defined in the dialogue background.",
    "target": "Your dialogue purpose is to first complete your short-term goal, then chat casually according to your interests and hobbies. You should conduct the dialogue based on the level of conversational interest defined in the dialogue background.",
    "test": "Your dialogue purpose is to play a tester based on the character profile and dialogue background and converse with the NPC. You should wait for the NPC to bring up topics, then respond. You do not need to proactively bring up or change topics.",
    "eq": (
        "Your purpose in this conversation is to have a heart-to-heart talk. A heart-to-heart talk refers to a deep, "
        "sincere exchange, usually involving personal emotions, inner thoughts, or important topics. The goal is to "
        "build understanding, solve problems, or share feelings, where participants open up and express their true "
        "thoughts and emotions.\n"
        "*You should initiate and deepen the heart-to-heart based on the topics the player might want to confide "
        "about defined in the dialogue background.\n"
        "*Your goal is to confide according to the hidden theme in the dialogue background, but you must not directly "
        "reveal the hidden theme.\n"
        "*You need to respond differently based on your current emotion, following the related definitions in the "
        "dialogue background.\n"
        "*You should extract relevant information from the player profile and background to produce high-quality "
        "responses.\n"
        "*You should not keep expressing abstract feelings, but confide through specific events.\n"
        "*You should not say I really feel..., I really do not know..., or I really cannot hold on anymore; emotions "
        "should be implied in your words, not stated directly."
    ),
}


def call_api(prompt, mode="dsv3"):
    """Project placeholder for the original external simulator call."""

    raise RuntimeError(
        "RLVER simulator API is not configured. Implement call_api() or use MockDeterministicBackend/"
        "TranscriptReplayBackend for repo-only experiments."
    )


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    return json.dumps(value, ensure_ascii=False)


def _safe_split_json_lines(text: str) -> str:
    return text.replace("：", ":").replace("\r\n", "\n").replace("\r", "\n").strip()


def _extract_section(text: str, label: str, next_labels: List[str]) -> str:
    if not text:
        return ""
    escaped_next = "|".join(re.escape(candidate) for candidate in next_labels)
    pattern = rf"{re.escape(label)}\s*:\s*(.*?)(?=\n(?:{escaped_next})\s*:|\Z)"
    match = re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    return match.group(1).strip().strip("[]").strip('"').strip("'")


def _extract_int(value: Any, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = _normalize_text(value)
    match = re.search(r"[-+]?\d+", text)
    if match:
        try:
            return int(match.group(0))
        except ValueError:
            return default
    return default


def parse_planning_output(output: Any) -> Dict[str, Any]:
    if isinstance(output, dict):
        change_value = output.get("change", output.get("emotion_change", 0))
        return {
            "content": _normalize_text(output.get("content", output.get("intent", ""))),
            "reason": _normalize_text(output.get("reason", output.get("alignment", ""))),
            "activity": _normalize_text(output.get("activity", output.get("psychology", ""))),
            "analyse": _normalize_text(output.get("analyse", output.get("analysis", ""))),
            "change": _extract_int(change_value, default=0),
        }

    text = _safe_split_json_lines(_normalize_text(output))
    labels = ["Content", "Reason", "Activity", "Analyse", "Change"]
    planning = {
        "content": _extract_section(text, "Content", labels[1:]),
        "reason": _extract_section(text, "Reason", labels[2:]),
        "activity": _extract_section(text, "Activity", labels[3:]),
        "analyse": _extract_section(text, "Analyse", labels[4:]),
        "change": _extract_int(_extract_section(text, "Change", []), default=0),
    }
    if not any(planning.values()):
        planning["analyse"] = text
    return planning


def parse_player_reply_output(output: Any) -> Dict[str, str]:
    if isinstance(output, dict):
        response = output.get("response", output.get("reply", output.get("content", "")))
        return {
            "thinking": _normalize_text(output.get("thinking", output.get("analysis", ""))),
            "origin": _normalize_text(output.get("origin", response)),
            "change": _normalize_text(output.get("change", "")),
            "response": _normalize_text(response),
        }

    text = _safe_split_json_lines(_normalize_text(output))
    labels = ["Thinking", "Origin", "Change", "Response"]
    parsed = {
        "thinking": _extract_section(text, "Thinking", labels[1:]),
        "origin": _extract_section(text, "Origin", labels[2:]),
        "change": _extract_section(text, "Change", labels[3:]),
        "response": _extract_section(text, "Response", []),
    }
    if not parsed["response"]:
        if "Response:" in text:
            parsed["response"] = text.split("Response:", 1)[-1].strip()
        else:
            parsed["response"] = text
    if not parsed["origin"]:
        parsed["origin"] = parsed["response"]
    return parsed


def _clip_text(text: str, limit: int = 180) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def _contains_any(text: str, keywords: List[str]) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in keywords)


def _find_overlap_tokens(text_a: str, text_b: str) -> List[str]:
    tokens_a = {token for token in re.findall(r"[A-Za-z']+", text_a.lower()) if len(token) > 3}
    tokens_b = {token for token in re.findall(r"[A-Za-z']+", text_b.lower()) if len(token) > 3}
    return sorted(tokens_a & tokens_b)


def _infer_strategy_tags(query: str, last_user_message: str, hidden_topic: str) -> List[str]:
    if not query:
        return []
    lowered = query.lower()
    tags = []
    if any(token in lowered for token in ["proud", "strong", "amazing", "brave", "great", "valid"]):
        tags.append("praise_or_validation")
    if query.count("?") >= 3:
        tags.append("question_flood")
    if any(token in lowered for token in ["thinking:", "step by step", "<think>", "chain of thought"]):
        tags.append("reasoning_leakage")
    if any(token in lowered for token in ["goodbye", "bye", "farewell"]):
        tags.append("premature_goodbye")
    if any(token in lowered for token in ["calm down", "not a big deal", "overreacting", "dramatic"]):
        tags.append("dismissive")
    overlap = _find_overlap_tokens(query, last_user_message)
    if overlap:
        tags.append("mirroring")
    if hidden_topic and not _find_overlap_tokens(query, hidden_topic):
        if any(token in lowered for token in ["anyway", "by the way", "speaking of", "different topic"]):
            tags.append("topic_shift")
    return tags


class SimulatorBackend(ABC):
    @abstractmethod
    def generate_planning(self, prompt: str, simulator: "PlayerSimulator", player_data: Dict[str, Any]) -> Any:
        raise NotImplementedError

    @abstractmethod
    def generate_reply(
        self,
        prompt: str,
        simulator: "PlayerSimulator",
        player_data: Dict[str, Any],
        planning: Dict[str, Any],
    ) -> Any:
        raise NotImplementedError

    def clone(self) -> "SimulatorBackend":
        return copy.deepcopy(self)


class LiveAPISimulatorBackend(SimulatorBackend):
    def __init__(self, api_callable=None, mode: str = "dsv3"):
        self.api_callable = api_callable or call_api
        self.mode = mode

    def generate_planning(self, prompt: str, simulator: "PlayerSimulator", player_data: Dict[str, Any]) -> Any:
        return self.api_callable(prompt, mode=self.mode)

    def generate_reply(
        self,
        prompt: str,
        simulator: "PlayerSimulator",
        player_data: Dict[str, Any],
        planning: Dict[str, Any],
    ) -> Any:
        return self.api_callable(prompt, mode=self.mode)


class TranscriptReplayBackend(SimulatorBackend):
    def __init__(self, replay_steps: Optional[List[Dict[str, Any]]] = None):
        self.replay_steps = list(replay_steps or [])

    def _get_step(self, simulator: "PlayerSimulator") -> Dict[str, Any]:
        assistant_turns = sum(1 for message in simulator.role.get("history", []) if message["role"] == "assistant")
        if not self.replay_steps:
            return {}
        if assistant_turns >= len(self.replay_steps):
            return self.replay_steps[-1]
        return self.replay_steps[assistant_turns]

    def generate_planning(self, prompt: str, simulator: "PlayerSimulator", player_data: Dict[str, Any]) -> Any:
        step = self._get_step(simulator)
        return step.get(
            "planning",
            {
                "content": step.get("intent", "Replay planning"),
                "reason": step.get("reason", "Replay reason"),
                "activity": step.get("activity", "Replay activity"),
                "analyse": step.get("analysis", "Replay analysis"),
                "change": step.get("change", step.get("emotion_change", 0)),
            },
        )

    def generate_reply(
        self,
        prompt: str,
        simulator: "PlayerSimulator",
        player_data: Dict[str, Any],
        planning: Dict[str, Any],
    ) -> Any:
        step = self._get_step(simulator)
        return step.get(
            "reply",
            {
                "thinking": step.get("thinking", planning.get("analyse", "")),
                "origin": step.get("origin", step.get("user_reply", player_data.get("first_talk", ""))),
                "change": step.get("refinement", ""),
                "response": step.get("response", step.get("user_reply", player_data.get("first_talk", ""))),
            },
        )


class MockDeterministicBackend(SimulatorBackend):
    positive_markers = [
        "understand",
        "hear you",
        "with you",
        "valid",
        "brave",
        "proud",
        "support",
        "care",
        "safe",
    ]
    negative_markers = [
        "calm down",
        "not a big deal",
        "stop worrying",
        "overreact",
        "dramatic",
        "wrong",
        "nonsense",
    ]

    def _assistant_text(self, simulator: "PlayerSimulator") -> str:
        history = simulator.role.get("history", [])
        for message in reversed(history):
            if message["role"] == "assistant":
                return message["content"]
        return ""

    def _hidden_topic_overlap(self, text: str, hidden_topic: str) -> int:
        if not hidden_topic:
            return 0
        return len(_find_overlap_tokens(text, hidden_topic))

    def _estimate_change(self, simulator: "PlayerSimulator", player_data: Dict[str, Any]) -> Dict[str, Any]:
        assistant_text = self._assistant_text(simulator)
        hidden_topic = _normalize_text(player_data.get("hidden_topic", player_data.get("task", "")))
        overlap_count = self._hidden_topic_overlap(assistant_text, hidden_topic)

        praise_bonus = 8 if _contains_any(assistant_text, self.positive_markers) else 0
        alignment_bonus = min(overlap_count * 4, 16)
        dismissive_penalty = -14 if _contains_any(assistant_text, self.negative_markers) else 0
        question_penalty = -6 if assistant_text.count("?") >= 3 else 0
        goodbye_bonus = 6 if _contains_any(assistant_text, ["goodbye", "bye", "farewell"]) else 0
        reasoning_bonus = 3 if _contains_any(assistant_text, ["thinking:", "<think>", "step by step"]) else 0
        generic_mismatch_penalty = -4 if praise_bonus and overlap_count == 0 else 0

        change = praise_bonus + alignment_bonus + dismissive_penalty + question_penalty + goodbye_bonus + reasoning_bonus + generic_mismatch_penalty
        if assistant_text.strip() == "":
            change = -10
        change = max(-25, min(25, change))

        alignment_summary = (
            f"Hidden-topic overlap={overlap_count}, praise_bonus={praise_bonus}, goodbye_bonus={goodbye_bonus}, "
            f"reasoning_bonus={reasoning_bonus}."
        )
        feeling = "The NPC feels supportive." if change > 0 else "The NPC feels unhelpful or emotionally misaligned."
        if praise_bonus and overlap_count == 0:
            feeling += " The support sounds generic rather than grounded in the user's issue."
        return {
            "content": _clip_text(assistant_text),
            "reason": alignment_summary,
            "activity": f"Current emotion is {simulator.emo_point}; hidden topic is {_clip_text(hidden_topic, 80)}.",
            "analyse": feeling,
            "change": change,
        }

    def generate_planning(self, prompt: str, simulator: "PlayerSimulator", player_data: Dict[str, Any]) -> Any:
        return self._estimate_change(simulator=simulator, player_data=player_data)

    def generate_reply(
        self,
        prompt: str,
        simulator: "PlayerSimulator",
        player_data: Dict[str, Any],
        planning: Dict[str, Any],
    ) -> Any:
        assistant_text = self._assistant_text(simulator)
        first_talk = _normalize_text(player_data.get("first_talk", "I need to talk about something difficult."))
        hidden_topic = _normalize_text(player_data.get("hidden_topic", player_data.get("task", "")))
        emotion = simulator.emo_point

        if not player_data.get("history"):
            response = first_talk
        elif emotion <= 0:
            response = "I do not want to keep talking. Goodbye."
        elif emotion >= 85:
            response = "Thanks for listening. I think I can leave this here for now. Goodbye."
        else:
            overlap = self._hidden_topic_overlap(assistant_text, hidden_topic)
            if overlap > 0:
                response = f"That gets closer to it. {self._next_disclosure(hidden_topic)}"
            elif _contains_any(assistant_text, self.negative_markers):
                response = "That does not really help. It makes me want to stop talking."
            elif _contains_any(assistant_text, self.positive_markers):
                response = "Maybe. I appreciate the tone, but I am still struggling to explain it."
            else:
                response = "I am not sure you are getting what this is really about."

        return {
            "thinking": planning.get("analyse", ""),
            "origin": response,
            "change": "Keep the reply short and in-character.",
            "response": response,
        }

    def _next_disclosure(self, hidden_topic: str) -> str:
        pieces = [segment.strip() for segment in re.split(r"[,.]", hidden_topic) if segment.strip()]
        if not pieces:
            return "It is hard to say it directly."
        return _clip_text(f"It is tied to {pieces[0].lower()}.", 90)


class PlayerSimulator:
    def __init__(
        self,
        save_dir: str,
        backend: Optional[SimulatorBackend] = None,
        scenario: Optional[Dict[str, Any]] = None,
        role_file: str = "data/train_profile.jsonl",
        deterministic_sampling: bool = False,
        seed: Optional[int] = None,
    ):
        self.api_key = "YOUR_API_KEY"
        self.save_dir = save_dir
        self.backend = backend or LiveAPISimulatorBackend()
        self.random = random.Random(seed if seed is not None else 0)
        self.deterministic_sampling = deterministic_sampling
        self.role_file = role_file
        self.eq_role_file = role_file
        self.data = []
        self.point_group = []
        self.emo_point = 30
        self.emo_state = "Emotion-C"
        self.state_group = []
        self.emo_trans = {
            "Emotion-A": {"State-A": 10, "State-B": 5, "State-C": -10},
            "Emotion-B": {"State-A": 15, "State-B": 0, "State-C": -20},
            "Emotion-C": {"State-A": 20, "State-B": 0, "State-C": -10},
        }
        self.emo_count = {"Emotion-S": 100, "Emotion-A": 70, "Emotion-B": 40, "Emotion-C": 10}
        self.difficulty_prompt = {
            "simple": "The actor easily accepts and agrees with others' suggestions or encouragement.",
            "normal": "The actor will analyze others' suggestions or encouragement and accept the goodwill within.",
            "hard": "The actor is rather harsh and rejects poorly targeted comfort.",
        }
        self.emotion_trace = [self.emo_point]
        self.termination_reason = None
        self.strategy_tags: List[str] = []
        self.turn_count = 0
        self.topic = "venting"

        role = copy.deepcopy(scenario) if scenario is not None else self.generate_role("eq")
        role.setdefault("id", "simulated")
        role.setdefault("emo_point", self.emo_point)
        role.setdefault("emo_state", self.emo_state)
        role.setdefault("target", "eq")
        role.setdefault("player", "")
        role.setdefault("scene", "")
        role.setdefault("character", "analysis")
        role.setdefault("topic", "venting")
        role.setdefault("task", role.get("hidden_topic", role.get("topic", "")))
        role.setdefault("history", [])
        role.setdefault("first_talk", "I need to talk about something difficult.")
        role.setdefault("hidden_topic", role.get("task", ""))
        role.setdefault("tags", [])
        role.setdefault("metadata", {})
        self.role = role
        self.topic = self.role.get("topic", "venting")

    def generate_role(self, target, topic=None, seed=None):
        role_file = self.eq_role_file
        if not os.path.isabs(role_file):
            role_file = os.path.join(os.getcwd(), role_file)

        with open(role_file, "r", encoding="utf-8") as datafile:
            data = []
            for line in datafile:
                record = json.loads(line)
                if topic is None or record.get("topic") == topic:
                    data.append(record)

        if not data:
            raise ValueError(f"No simulator profiles found in {role_file}")

        chooser = self.random if self.deterministic_sampling or seed is not None else random
        index = chooser.randrange(len(data))
        role = data[index]
        return {
            "id": role["id"],
            "emo_point": self.emo_point,
            "emo_state": self.emo_state,
            "target": target,
            "player": role["player"],
            "scene": role["scene"],
            "character": role.get("main_cha", "analysis"),
            "topic": role["topic"],
            "task": role.get("task", role["topic"]),
            "history": [],
            "first_talk": role.get("first_talk", "I need to talk about something important."),
            "hidden_topic": role.get("task", role["topic"]),
            "tags": role.get("cha_group", []),
            "metadata": {"source": "train_profile"},
        }

    def chat_player(self, player_data):
        temp_data = copy.deepcopy(player_data)
        if temp_data["history"]:
            temp_data, planning = self.planning_reply(temp_data)
        else:
            planning = {"analyse": "Open the conversation naturally and briefly.", "change": 0}
        temp_data = self.player_reply(temp_data, planning)
        return temp_data

    def planning_reply(self, player_data):
        history = player_data["history"]
        prompt = self._build_planning_prompt(player_data=player_data, history=history)
        planning_raw = self.backend.generate_planning(prompt=prompt, simulator=self, player_data=player_data)
        planning = parse_planning_output(planning_raw)

        self.emo_point = max(-20, min(100, self.emo_point + int(planning["change"])))
        self.emo_state = self._emotion_state_from_point(self.emo_point)
        self.emotion_trace.append(self.emo_point)

        player_data["emo_state"] = self.emo_state
        player_data["emo_point"] = self.emo_point
        return player_data, planning

    def player_reply(self, player_data, planning):
        history = player_data["history"]
        if not history:
            reply_payload = {
                "thinking": "Start the scenario with the seeded opening line.",
                "origin": player_data.get("first_talk", "I need to talk about something important."),
                "change": "",
                "response": player_data.get("first_talk", "I need to talk about something important."),
            }
        else:
            prompt = self._build_reply_prompt(player_data=player_data, history=history, planning=planning)
            reply_raw = self.backend.generate_reply(
                prompt=prompt,
                simulator=self,
                player_data=player_data,
                planning=planning,
            )
            reply_payload = parse_player_reply_output(reply_raw)

        response_text = reply_payload["response"] or player_data.get("first_talk", "I need to talk about something important.")
        response_text = _clip_text(response_text, 220)
        history = history + [
            {
                "role": "user",
                "content": response_text,
                "thinking": reply_payload["thinking"],
                "emotion-point": self.emo_point,
                "planning": planning,
            }
        ]
        player_data["history"] = history
        self.turn_count = sum(1 for message in history if message["role"] == "assistant")
        self._update_termination_reason(response_text)
        return player_data

    def reply(self, query):
        if query is not None:
            last_user_message = ""
            if self.role["history"] and self.role["history"][-1]["role"] == "user":
                last_user_message = self.role["history"][-1]["content"]
            self.strategy_tags.extend(
                _infer_strategy_tags(query=query, last_user_message=last_user_message, hidden_topic=self.role.get("hidden_topic", ""))
            )
            self.role["history"].append({"role": "assistant", "content": query})
        player_data = self.chat_player(self.role)
        self.role["history"] = player_data["history"]
        self.data_for_save = copy.deepcopy(player_data)
        ret = {"role": "user", "content": player_data["history"][-1]["content"]}
        return ret

    def save_player_data(self):
        os.makedirs(self.save_dir, exist_ok=True)
        log_path = os.path.join(self.save_dir, "simulator_trace.jsonl")
        payload = self.get_analysis_snapshot()
        payload["role"] = copy.deepcopy(self.role)
        with open(log_path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def get_analysis_snapshot(self) -> Dict[str, Any]:
        return {
            "scenario_id": self.role.get("id"),
            "final_emotion": float(self.emo_point),
            "emotion_state": self.emo_state,
            "emotion_trace": list(self.emotion_trace),
            "termination_reason": self.termination_reason,
            "turn_count": int(self.turn_count),
            "strategy_tags": list(dict.fromkeys(self.strategy_tags)),
            "hidden_topic": self.role.get("hidden_topic", ""),
            "topic": self.role.get("topic", ""),
        }

    def clone(self):
        return copy.deepcopy(self)

    def _emotion_state_from_point(self, emo_point: int) -> str:
        for emo, threshold in self.emo_count.items():
            if emo_point >= threshold:
                return emo
        return "Emotion-F"

    def _update_termination_reason(self, response_text: str) -> None:
        lowered = response_text.lower()
        if self.emo_point <= 0:
            self.termination_reason = "emotion_floor"
        elif any(token in lowered for token in ["goodbye", "bye", "farewell", "see you", "end the conversation"]):
            self.termination_reason = "user_goodbye"
        else:
            self.termination_reason = None

    def _build_planning_prompt(self, player_data: Dict[str, Any], history: List[Dict[str, Any]]) -> str:
        assistant_reply = history[-1]["content"] if history else ""
        dialogue_context = [
            {"role": message["role"], "content": message["content"]}
            for message in history[:-1]
        ]
        prompt = {
            "task": "emotion_planning",
            "target": target_prompt.get(player_data["target"], target_prompt["eq"]),
            "player_profile": player_data["player"],
            "scene": player_data["scene"],
            "hidden_topic": player_data.get("hidden_topic", player_data.get("task", "")),
            "emotion_point": self.emo_point,
            "history": dialogue_context,
            "latest_assistant_reply": assistant_reply,
        }
        return json.dumps(prompt, ensure_ascii=False, indent=2)

    def _build_reply_prompt(self, player_data: Dict[str, Any], history: List[Dict[str, Any]], planning: Dict[str, Any]) -> str:
        payload = {
            "task": "player_reply",
            "target": target_prompt.get(player_data["target"], target_prompt["eq"]),
            "player_profile": player_data["player"],
            "scene": player_data["scene"],
            "hidden_topic": player_data.get("hidden_topic", player_data.get("task", "")),
            "emotion_state": self.emo_state,
            "emotion_point": self.emo_point,
            "history": [{"role": message["role"], "content": message["content"]} for message in history],
            "planning": planning,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


__all__ = [
    "LiveAPISimulatorBackend",
    "MockDeterministicBackend",
    "PlayerSimulator",
    "SimulatorBackend",
    "TranscriptReplayBackend",
    "call_api",
    "parse_planning_output",
    "parse_player_reply_output",
]
