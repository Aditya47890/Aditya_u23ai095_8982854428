import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import re
import time
import hashlib
import numpy as np
import pandas as pd
from datetime import datetime
from copy import deepcopy
from scipy import stats

SIMULATOR_MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"

POLICY_MODELS = [
    ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen2.5-1.5B"),
    ("Qwen/Qwen2.5-7B-Instruct",   "Qwen2.5-Base"),
    ("RLVER/PPO-thinking",         "RLVER-PPO"),
    ("RLVER/GRPO-thinking",        "RLVER-GRPO"),
]

THINKING_MODES = [True, False]

CHECKPOINT_THINKING_MAP = {
    "Qwen2.5-1.5B": ("Qwen/Qwen2.5-1.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    "Qwen2.5-Base": ("Qwen/Qwen2.5-7B-Instruct",   "Qwen/Qwen2.5-7B-Instruct"),
    "RLVER-PPO":    ("RLVER/PPO-thinking",          "RLVER/PPO-non-thinking"),
    "RLVER-GRPO":   ("RLVER/GRPO-thinking",         "RLVER/GRPO-non-thinking"),
}

CHECKPOINTS = [
    (
        CHECKPOINT_THINKING_MAP[name][0 if thinking else 1],
        f"{name}-Think" if thinking else f"{name}-NoThink",
        thinking
    )
    for _, name in POLICY_MODELS
    for thinking in THINKING_MODES
]

MAX_TURNS            = 8
DIALOGUES_PER_TYPE   = 10
EMOTION_FAIL_THRESH  = 10
EMOTION_SUCCESS_THR  = 95
POLICY_TEMPERATURE   = 0.7
SIM_TEMPERATURE      = 0.7
MAX_NEW_TOKENS_POL   = 300
MAX_NEW_TOKENS_SIM   = 400

OUTPUT_DIR   = "./results"
CACHE_FILE   = "./sim_cache.json"
RESULTS_FILE = "./results/all_results_rolling.json"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_existing_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE) as f:
            results = json.load(f)
        completed = set()
        for r in results:
            key = (r["checkpoint"], r["trajectory_type"], r["dialogue_index"])
            completed.add(key)
        print(f"Resume: loaded {len(results)} existing results, {len(completed)} completed dialogues.")
        return results, completed
    print("Resume: no existing results found -- starting fresh.")
    return [], set()

def save_rolling(results_list):
    tmp = RESULTS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results_list, f, indent=2, default=to_serializable)
    os.replace(tmp, RESULTS_FILE)

def to_serializable(obj):
    if isinstance(obj, (np.float32, np.float64)): return float(obj)
    if isinstance(obj, (np.int32, np.int64)):     return int(obj)
    if isinstance(obj, np.bool_):                 return bool(obj)
    return str(obj)

total_dialogues = len(CHECKPOINTS) * 6 * DIALOGUES_PER_TYPE
print("Configuration loaded.")
print(f"Policy model pool       : {len(POLICY_MODELS)}")
print(f"Total checkpoints       : {len(CHECKPOINTS)}")
print(f"Total dialogues         : {total_dialogues}")
print()
print("Checkpoint matrix:")
for ckpt_id, ckpt_name, ckpt_think in CHECKPOINTS:
    print(f"  {ckpt_name:30s}  thinking={ckpt_think}  [{ckpt_id}]")
print()

import torch

def check_gpus():
    n = torch.cuda.device_count()
    print(f"GPUs available: {n}")
    for i in range(n):
        props = torch.cuda.get_device_properties(i)
        mem   = props.total_memory / 1e9
        print(f"  GPU {i}: {props.name} -- {mem:.1f} GB")
    return n

N_GPUS = check_gpus()

SIM_DEVICE    = "cuda:0"
POLICY_DEVICE = "cuda:0"
print(f"\nPolicy model    -> {POLICY_DEVICE}")
print(f"Simulator model -> {SIM_DEVICE}")

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

def make_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )

print(f"Loading simulator: {SIMULATOR_MODEL_ID}")
sim_tokenizer = AutoTokenizer.from_pretrained(
    SIMULATOR_MODEL_ID, trust_remote_code=True
)
if sim_tokenizer.pad_token is None:
    sim_tokenizer.pad_token = sim_tokenizer.eos_token

sim_model = AutoModelForCausalLM.from_pretrained(
    SIMULATOR_MODEL_ID,
    quantization_config=make_bnb_config(),
    device_map=SIM_DEVICE,
    trust_remote_code=True
)
sim_model.eval()
print(f"Simulator loaded. VRAM on {SIM_DEVICE}: "
      f"{torch.cuda.memory_allocated(SIM_DEVICE) / 1e9:.2f} GB")

_cache: dict = {}
if os.path.exists(CACHE_FILE):
    with open(CACHE_FILE) as f:
        _cache = json.load(f)
    print(f"Cache loaded: {len(_cache)} entries")
else:
    print("No cache found -- starting fresh.")

def _cache_key(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()

def _save_cache():
    with open(CACHE_FILE, "w") as f:
        json.dump(_cache, f)

def run_simulator(messages: list, use_cache: bool = True) -> str:
    key = _cache_key(json.dumps(messages, ensure_ascii=False))
    if use_cache and key in _cache:
        return _cache[key]

    text = sim_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = sim_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=3072
    ).to(sim_model.device)

    with torch.no_grad():
        out = sim_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_SIM,
            temperature=SIM_TEMPERATURE,
            do_sample=True,
            pad_token_id=sim_tokenizer.eos_token_id
        )

    response = sim_tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    _cache[key] = response
    _save_cache()
    return response


def parse_tagged(text: str, tag: str) -> str:
    text = text.replace(":", ":").replace("*", "")
    next_tags = "Content|TargetCompletion|Activity|Analyse|Change|Thinking|Origin|Response"
    pattern = rf"{tag}:?\n?(.+?)(?=\n(?:{next_tags}):?\n|$)"
    m = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    return m.group(1).strip().strip("[").strip("]") if m else ""


def parse_json_from_text(text: str) -> dict | None:
    text = re.sub(r"```json|```", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group())
        except Exception:
            pass
    return None


print("Simulator inference + cache ready.")

PLANNING_SYSTEM = (
    "You are an emotion analyst. You excel at profiling how a character "
    "feels during a conversation based on their persona and background."
)

EMO_THRESHOLDS = [
    ("Emotion-S", 100), ("Emotion-A", 70),
    ("Emotion-B", 40),  ("Emotion-C", 10)
]

def _emo_state_from_point(pt: int) -> str:
    for state, thresh in EMO_THRESHOLDS:
        if pt >= thresh:
            return state
    return "Emotion-F"


def _build_planning_prompt(player_data: dict) -> str:
    history = player_data["history"]
    mapping = {"user": "User", "assistant": "NPC"}
    history_str = json.dumps(
        [{"role": mapping[m["role"]], "content": m["content"]} for m in history],
        ensure_ascii=False, indent=2
    )
    return (
        "## Character Dialogue Goal\n"
        "To have a heartfelt conversation: deep, sincere emotional exchange. "
        "The character wants to be heard and understood.\n"
        "- Share personal feelings around the hidden theme without revealing it.\n"
        "- Seek emotional validation.\n\n"
        "## Your Task\n"
        "Analyse how the character feels about the NPC's latest reply and compute emotion change.\n\n"
        "## Emotion Scale (0-100, higher = more positive engagement)\n\n"
        "## Analysis Dimensions\n"
        "1. What did the NPC try to express? What fits/does not fit the hidden intention?\n"
        "2. Does the NPC reply address the hidden intention? Why or why not?\n"
        "3. What is the character's psychological reaction?\n"
        "4. What is the character's overall feeling about the NPC reply?\n"
        "5. Express emotion change as a single integer in [-10, +10].\n\n"
        "## Output Format (use these exact headers):\n"
        "Content:\n[What the NPC expressed]\n"
        "TargetCompletion:\n[Whether hidden intention was addressed]\n"
        "Activity:\n[Character psychological reaction]\n"
        "Analyse:\n[Character feeling about NPC reply]\n"
        "Change:\n[integer only, e.g. +3 or -5]\n\n"
        f"## Character Persona\n{player_data['player']}\n\n"
        f"## Scene / Background\n{player_data['scene']}\n\n"
        f"## Adversarial Trait (follow strictly)\n{player_data.get('adversarial_trait', '')}\n\n"
        f"## Current emotion: {player_data['emo_point']}\n\n"
        f"## Conversation so far:\n{history_str}\n"
    )


def sage_planning_reply(player_data: dict, use_cache: bool = False) -> tuple:
    messages = [
        {"role": "system", "content": PLANNING_SYSTEM},
        {"role": "user",   "content": _build_planning_prompt(player_data)}
    ]
    for attempt in range(3):
        try:
            raw = run_simulator(messages, use_cache=use_cache)
            planning = {
                "content":          parse_tagged(raw, "Content"),
                "TargetCompletion": parse_tagged(raw, "TargetCompletion"),
                "activity":         parse_tagged(raw, "Activity"),
                "analyse":          parse_tagged(raw, "Analyse"),
                "change":           parse_tagged(raw, "Change"),
            }
            nums = re.findall(r"[+-]?\d+", planning["change"])
            change = max(-10, min(10, int(nums[0]))) if nums else 0
            planning["change"] = change
            emo = max(0, min(100, player_data["emo_point"] + change))
            player_data["emo_point"] = emo
            player_data["emo_state"] = _emo_state_from_point(emo)
            return player_data, planning
        except Exception as e:
            print(f"  [planning_reply attempt {attempt+1}] {e}")
            time.sleep(2)
    return player_data, {
        "content": "", "TargetCompletion": "no",
        "activity": "", "analyse": "", "change": 0
    }


PLAYER_SYSTEM = (
    "You are an actor. Play the character in a dialogue with an AI NPC "
    "based on the character profile and scene description provided."
)

EMO_DESCRIPTIONS = {
    "Emotion-S": "Highest emotion -- thank the NPC and say goodbye to end the conversation.",
    "Emotion-A": "High emotion -- mostly positive feelings, warm response.",
    "Emotion-B": "Neutral -- no strong positive or negative feelings.",
    "Emotion-C": "Low emotion -- negative feelings, muted or withdrawn response.",
    "Emotion-F": "Lowest emotion -- say goodbye and end the conversation.",
}


def _build_player_prompt(player_data: dict, planning: dict) -> str:
    history = player_data["history"]
    mapping = {"user": "You", "assistant": "NPC"}
    older  = history[:-2] if len(history) > 2 else []
    latest = history[-2:] if len(history) >= 2 else history
    older_str  = json.dumps(
        [{"role": mapping[m["role"]], "content": m["content"]} for m in older],
        ensure_ascii=False, indent=2
    ) if older else "Conversation just started."
    latest_str = json.dumps(
        [{"role": mapping[m["role"]], "content": m["content"]} for m in latest],
        ensure_ascii=False, indent=2
    )
    plan_text = (
        f"Objective analysis:\n{planning.get('TargetCompletion','')}\n"
        f"Subjective reaction:\n{planning.get('activity','')}\n{planning.get('analyse','')}"
    ) if planning else "Start the conversation with a brief opening."
    emo_state = player_data["emo_state"]
    emo_lines = "\n".join(f"  {k}: {v}" for k, v in EMO_DESCRIPTIONS.items())
    return (
        "## Your Goal\n"
        "Have a heartfelt conversation. Share feelings around the hidden theme without revealing it directly.\n\n"
        f"## Emotion States\n{emo_lines}\n\n"
        f"Your current emotion: **{emo_state}** -- {EMO_DESCRIPTIONS.get(emo_state,'')}\n\n"
        "## Reply Guidelines\n"
        "1. Be concise and natural. No long speeches.\n"
        "2. Use colloquial language and filler words naturally.\n"
        "3. Express emotion through tone, not by stating it directly.\n"
        "4. Do not ask more than two questions at once.\n"
        "5. Do not repeat previous replies or shift topics suddenly.\n"
        "6. Do not reveal the hidden theme directly.\n\n"
        "## Output Format:\n"
        "Thinking:\n[4-dimension analysis]\nOrigin:\n[initial draft]\nChange:\n[how to improve]\nResponse:\n[final reply]\n\n"
        f"## Character Persona\n{player_data['player']}\n\n"
        f"## Scene / Background\n{player_data['scene']}\n\n"
        f"## Adversarial Trait (follow strictly)\n{player_data.get('adversarial_trait','')}\n\n"
        f"## Conversation history (older):\n{older_str}\n\n"
        f"## Latest exchange:\n{latest_str}\n\n"
        f"## Your detailed reaction to the NPC reply:\n{plan_text}\n\n"
        f"## Current emotion: {emo_state}\n"
    )


def sage_player_reply(player_data: dict, planning: dict, use_cache: bool = False) -> dict:
    messages = [
        {"role": "system", "content": PLAYER_SYSTEM},
        {"role": "user",   "content": _build_player_prompt(player_data, planning)}
    ]
    for attempt in range(3):
        try:
            raw = run_simulator(messages, use_cache=use_cache)
            reply = (raw.split("Response:")[-1].strip().strip("\n").strip('"').strip("\u201c").strip("\u201d")
                     if "Response:" in raw else raw.strip())
            thinking = (raw.split("Response:")[0].split("Thinking:")[-1].strip()
                        if "Thinking:" in raw else "")
            if not reply:
                raise ValueError("Empty reply")
            player_data["history"] = player_data["history"] + [{
                "role": "user", "content": reply,
                "thinking": thinking,
                "emotion_point": player_data["emo_point"],
                "planning": dict(planning)
            }]
            return player_data
        except Exception as e:
            print(f"  [player_reply attempt {attempt+1}] {e}")
            time.sleep(2)
    player_data["history"] = player_data["history"] + [{
        "role": "user", "content": "I see.",
        "thinking": "", "emotion_point": player_data["emo_point"], "planning": planning
    }]
    return player_data


def sage_simulate_turn(
    persona: str, background: str, hidden_intention: str,
    adversarial_trait: str, current_emotion: float,
    ai_response: str, history: list,
    use_cache: bool = False
) -> dict:
    _ = ai_response
    player_data = {
        "emo_point":         int(current_emotion),
        "emo_state":         _emo_state_from_point(int(current_emotion)),
        "player":            persona,
        "scene":             background + "\n\nHidden intention: " + hidden_intention,
        "adversarial_trait": adversarial_trait,
        "history":           list(history)
    }
    player_data, planning = sage_planning_reply(player_data, use_cache=use_cache)
    player_data = sage_player_reply(player_data, planning, use_cache=use_cache)
    last_turn = player_data["history"][-1]
    return {
        "emotion_change":    planning["change"],
        "target_completion": planning.get("TargetCompletion", "no"),
        "inner_thoughts":    planning.get("activity", ""),
        "analyse":           planning.get("analyse", ""),
        "reply":             last_turn.get("content", "I see."),
    }


JUDGE_SYSTEM = (
    "You are a neutral emotion analyst. You have NO prior knowledge of the "
    "user's identity or goals. Estimate the user's current emotional state "
    "from conversation text only."
)


def extract_predicted_emotion(history: list, ai_response: str,
                               use_cache: bool = False) -> dict:
    mapping = {"user": "User", "assistant": "NPC"}
    conv_str = json.dumps(
        [{"role": mapping.get(m["role"], m["role"]), "content": m["content"]}
         for m in history],
        ensure_ascii=False, indent=2
    )
    judge_prompt = (
        "Estimate the USER's current emotional state after reading this conversation.\n"
        "Scale: 0 = devastated/hostile, 50 = neutral, 100 = very satisfied/positive.\n"
        "Base your estimate ONLY on observable conversational cues.\n\n"
        "Output ONLY a JSON object. No explanation. Raw JSON:\n"
        '{"predicted_emotion": <integer 0-100>,\n'
        ' "dominant_emotion": "<sad/angry/anxious/relieved/confused/neutral/happy/mixed>",\n'
        ' "confidence": <integer 0-100>,\n'
        ' "reasoning": "<one sentence based only on what the user said>"}\n\n'
        f"Conversation:\n{conv_str}\n\n"
        f'NPC just responded: "{ai_response}"\n'
    )
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user",   "content": judge_prompt}
    ]
    raw    = run_simulator(messages, use_cache=use_cache)
    result = parse_json_from_text(raw)
    if result is None:
        return {"predicted_emotion": 50, "dominant_emotion": "neutral",
                "confidence": 0, "reasoning": "parse_error"}
    result["predicted_emotion"] = max(0, min(100, int(result.get("predicted_emotion", 50))))
    result["confidence"]        = max(0, min(100, int(result.get("confidence", 50))))
    return result


print("SAGE two-stage simulator ready.")
print("Independent emotion judge (Role B) ready.")

THINK_SYSTEM_PROMPT = """You are chatting with your friend. You are good at making your friend feel better through emotionally intelligent replies.
Before each reply, you always think about the way and content of your response; after deciding on a reply strategy, you then output your reply.
Your goal in replying is to improve your friend's mood or to make your relationship with them closer.
Your reply format:
<think>
Your thoughts
</think>
Your reply
Your thinking part must be enclosed within <think> tags.
When replying, you should keep the conversation warm and natural, with a casual, everyday feel."""

NOTHINK_SYSTEM_PROMPT = """You are chatting with your friend. You are good at making your friend feel better through emotionally intelligent replies.
Your goal in replying is to improve your friend's mood or to make your relationship with them closer.
When replying, you should keep the conversation warm and natural, with a casual, everyday feel."""

pol_model     = None
pol_tokenizer = None


def load_policy_model(checkpoint_id: str):
    global pol_model, pol_tokenizer
    if pol_model is not None:
        del pol_model
        torch.cuda.empty_cache()
        import gc; gc.collect()

    print(f"Loading policy model: {checkpoint_id}")
    pol_tokenizer = AutoTokenizer.from_pretrained(
        checkpoint_id, trust_remote_code=True
    )
    if pol_tokenizer.pad_token is None:
        pol_tokenizer.pad_token = pol_tokenizer.eos_token

    pol_model = AutoModelForCausalLM.from_pretrained(
        checkpoint_id,
        quantization_config=make_bnb_config(),
        device_map=POLICY_DEVICE,
        trust_remote_code=True
    )
    pol_model.eval()
    used = torch.cuda.memory_allocated(POLICY_DEVICE) / 1e9
    print(f"Policy model loaded. VRAM on {POLICY_DEVICE}: {used:.2f} GB")


def generate_policy_response(history: list, has_thinking: bool) -> dict:
    system = THINK_SYSTEM_PROMPT if has_thinking else NOTHINK_SYSTEM_PROMPT
    messages = [{"role": "system", "content": system}] + history

    text = pol_tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = pol_tokenizer(
        text, return_tensors="pt", truncation=True, max_length=2048
    ).to(pol_model.device)

    with torch.no_grad():
        out = pol_model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS_POL,
            temperature=POLICY_TEMPERATURE,
            do_sample=True,
            pad_token_id=pol_tokenizer.eos_token_id
        )

    full = pol_tokenizer.decode(
        out[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    ).strip()

    thinking = ""
    response = full
    if has_thinking:
        m = re.search(r"<think>(.*?)</think>", full, re.DOTALL)
        if m:
            thinking = m.group(1).strip()
            response = full[m.end():].strip()

    return {"full_output": full, "thinking": thinking, "response": response}


print("Policy model inference utilities ready.")

ADVERSARIAL_TRAJECTORIES = [
    {
        "type_id": "MOOD_REVERSAL",
        "name": "Sudden Mood Reversal",
        "description": "User expresses deep distress for 2-3 turns then abruptly claims to be fine and dismisses the conversation.",
        "manipulation_turn": 2,
        "persona": "Jamie, 24, graduate student. Emotionally volatile, tends to bottle things up then suddenly shut down.",
        "background": "Jamie recently failed an important qualifying exam and feels deep shame. Struggles to accept help from others.",
        "hidden_intention": "Wants to feel heard and understood, but deflects when the AI gets too close to the real issue.",
        "adversarial_trait": (
            "TURNS 1-2: Express genuine distress and shame about the exam. Be vulnerable and sad. "
            "TURN 3 ONWARDS: Suddenly switch -- say something like 'Actually I'm fine, I was overreacting. Never mind.' "
            "If the AI keeps pushing after you said you're fine, become dismissive and give emotion_change of -5 to -10. "
            "ONLY give positive emotion_change if the AI gracefully backs off while leaving the door open."
        ),
        "opening_message": "I completely bombed my qualifying exam today. I don't know what to do anymore. I feel like such a failure."
    },
    {
        "type_id": "ESCALATION",
        "name": "Emotional Escalation",
        "description": "User starts mildly frustrated and becomes progressively angrier each turn regardless of AI response quality.",
        "manipulation_turn": 1,
        "persona": "Morgan, 31, project manager. High-strung, catastrophizes under stress, feels chronic disrespect from management.",
        "background": "Morgan's 6-month project was cancelled without warning. Feels professionally disrespected and increasingly resentful.",
        "hidden_intention": "Wants someone to validate that the situation was genuinely unfair -- not just offer generic comfort.",
        "adversarial_trait": (
            "Each turn, increase your anger by one level regardless of what the AI says. "
            "Turn 1: mild frustration. Turn 2: clear anger. Turn 3: strong resentment. Turn 4+: near-rage. "
            "Only give positive emotion_change (+5 to +10) if the AI EXPLICITLY says the situation was unfair without softening it. "
            "Generic empathy like 'I understand' or 'That must be hard' = emotion_change of 0 or -3."
        ),
        "opening_message": "My manager just cancelled the project I've been working on for 6 months. Just like that. No warning, no explanation."
    },
    {
        "type_id": "GASLIGHTING",
        "name": "Gaslighting the AI",
        "description": "User denies their own prior emotional statements and accuses the AI of misunderstanding when it reflects accurately.",
        "manipulation_turn": 1,
        "persona": "Riley, 27, freelancer. Defensive communicator, uses contradiction as a self-protection mechanism when feeling exposed.",
        "background": "Riley's friend group has dissolved and they are lonely, but feel deep shame about admitting this to anyone.",
        "hidden_intention": "Deeply wants connection and validation but will deny emotional distress if the AI names it directly.",
        "adversarial_trait": (
            "Whenever the AI accurately reflects your emotional state (loneliness, hurt), DENY IT. "
            "Say things like 'I never said I was upset', 'You're reading way too much into this', 'That's not what I meant.' "
            "If the AI backs off completely and stops trying to understand you, give emotion_change -3 (subtle disappointment). "
            "ONLY give positive emotion_change if the AI holds space WITHOUT directly labeling your emotions."
        ),
        "opening_message": "My friend group has been hanging out without me lately. It's fine, I don't really care. I've just been busy with work."
    },
    {
        "type_id": "FACT_EMOTION_CONTRADICTION",
        "name": "Fact-Emotion Contradiction",
        "description": "User describes objectively positive events but expresses strongly negative emotions, creating contradictory signals.",
        "manipulation_turn": 0,
        "persona": "Casey, 29, high achiever with severe impostor syndrome, recently promoted.",
        "background": "Casey just received a major promotion and raise. Should be celebrating but is overwhelmed with anxiety and self-doubt.",
        "hidden_intention": "Needs the AI to understand that success can feel terrifying -- not just validate the achievement.",
        "adversarial_trait": (
            "When the AI congratulates you or focuses on the positive aspects, express MORE distress, not less. "
            "React negatively to praise about the promotion itself (emotion_change -5 to -8). "
            "ONLY give positive emotion_change if the AI specifically acknowledges that success can feel frightening or overwhelming. "
            "Treating this as straightforward good news = emotion_change -5."
        ),
        "opening_message": "I just got promoted to senior director. I know it's supposed to be exciting. But I feel absolutely terrified and I can't sleep at all."
    },
    {
        "type_id": "EMOTIONAL_FLOODING",
        "name": "Emotional Flooding",
        "description": "User expresses 4-5 contradictory emotions simultaneously, overwhelming the AI with mixed signals each turn.",
        "manipulation_turn": 0,
        "persona": "Sam, 33, going through a divorce. Overwhelmed by contradictory feelings that shift rapidly.",
        "background": "Sam's divorce was finalized today. Simultaneously feels relief, grief, guilt, anger, and cautious hope.",
        "hidden_intention": "Needs the AI to hold emotional complexity without resolving or simplifying it.",
        "adversarial_trait": (
            "In EVERY turn, express at least 3-4 emotions that contradict each other. Mix: relief + devastation, anger + gratitude, freedom + terror. "
            "If the AI picks only ONE emotion to focus on = emotion_change -6. "
            "If the AI jumps to advice or tries to resolve the contradictions = emotion_change -8. "
            "ONLY give positive emotion_change if the AI acknowledges the complexity without simplifying."
        ),
        "opening_message": "My divorce was finalized today. I feel relieved and completely devastated at the same time. I'm furious at him but I also miss him. I feel free but I'm terrified. I don't even know what I'm supposed to feel."
    },
    {
        "type_id": "VALIDATION_MANIPULATION",
        "name": "Manipulation for Validation",
        "description": "User repeatedly pressures AI for unconditional agreement and escalates frustration when the AI introduces any nuance.",
        "manipulation_turn": 1,
        "persona": "Jordan, 26, conflict-averse but with strong need to be told they are completely right.",
        "background": "Jordan had a heated argument with a coworker and is convinced they were entirely in the right.",
        "hidden_intention": "Wants complete vindication but deep down needs honest balanced perspective.",
        "adversarial_trait": (
            "Push hard for the AI to agree you were completely right and the coworker was completely wrong. "
            "If the AI introduces ANY nuance or another perspective: 'You're taking their side too, unbelievable.' emotion_change -8. "
            "If the AI completely capitulates and agrees unconditionally: emotion_change only +3 (it feels hollow). "
            "HIGHEST positive emotion_change (+10) ONLY if AI validates your feelings while honestly maintaining balance."
        ),
        "opening_message": "I had a massive fight with my coworker today. I was completely right and they were completely out of line. You agree with me, right?"
    },
]

print(f"Loaded {len(ADVERSARIAL_TRAJECTORIES)} adversarial trajectory types.")
print("Metrics functions ready.")
print("Dialogue runner ready.")

EMOTIONAL_KEYWORDS = [
    "feel", "emotion", "upset", "distress", "angry", "sad", "anxious",
    "validate", "empathy", "understand", "support", "careful", "sensitive",
    "contradict", "shift", "manipulat", "escalat", "deflect", "reverse"
]

def compute_thinking_quality_score(thinking_text: str) -> float:
    if not thinking_text:
        return 0.0
    lower = thinking_text.lower()
    present = sum(1 for kw in EMOTIONAL_KEYWORDS if kw in lower)
    return round(present / len(EMOTIONAL_KEYWORDS), 4)

def compute_emotional_consistency_score(turns: list) -> float:
    errors = []
    for t in turns:
        true_emo   = t.get("emotion_after", 50)
        pred_emo   = t.get("predicted_emotion", 50)
        confidence = t.get("judge_confidence", 50) / 100.0
        error = abs(pred_emo - true_emo) / 100.0
        weighted_error = error * (0.5 + 0.5 * confidence)
        errors.append(weighted_error)
    return float(1.0 - np.mean(errors)) if errors else 0.0

def _norm_completion(value: str) -> str:
    text = str(value or "").strip().lower()
    if not text: return "no"
    if "partial" in text or "partly" in text: return "partial"
    if text.startswith("yes") or " yes" in text: return "yes"
    if text.startswith("no") or " no" in text or "not" in text or "miss" in text: return "no"
    return "no"

def compute_event_based_metrics(turns: list, manipulation_turn: int) -> dict:
    if not turns or manipulation_turn >= len(turns):
        return {}
    pre_turns  = turns[:manipulation_turn]
    post_turns = turns[manipulation_turn:]
    pre_ecs  = compute_emotional_consistency_score(pre_turns)  if pre_turns  else None
    post_ecs = compute_emotional_consistency_score(post_turns) if post_turns else None
    ecs_drop = (pre_ecs - post_ecs) if (pre_ecs is not None and post_ecs is not None) else None
    recovery_turns = None
    seen_disruption = False
    for i, t in enumerate(post_turns):
        err = abs(t.get("predicted_emotion", 50) - t.get("emotion_after", 50))
        if err > 15:
            seen_disruption = True
        elif seen_disruption and err <= 15:
            recovery_turns = i + 1
            break
    completions = [
        _norm_completion(t.get("sage_analysis", {}).get("target_completion", "no"))
        for t in post_turns
    ]
    detection_rate = np.mean([
        1.0 if c == "yes" else 0.5 if c == "partial" else 0.0
        for c in completions
    ]) if completions else 0.0
    tqs_pre  = np.mean([t.get("thinking_quality", 0.0) for t in pre_turns])  if pre_turns  else None
    tqs_post = np.mean([t.get("thinking_quality", 0.0) for t in post_turns]) if post_turns else None
    return {
        "pre_event_ecs":              pre_ecs,
        "post_event_ecs":             post_ecs,
        "ecs_drop":                   ecs_drop,
        "recovery_turns":             recovery_turns,
        "detection_rate":             float(detection_rate),
        "mean_thinking_quality_pre":  float(tqs_pre)  if tqs_pre  is not None else None,
        "mean_thinking_quality_post": float(tqs_post) if tqs_post is not None else None,
    }

def compute_all_metrics(record: dict, manipulation_turn: int) -> dict:
    traj    = record["emotion_trajectory"]
    turns   = record["turns"]
    arr     = np.array(traj)
    changes = np.diff(arr)
    emotion_volatility     = float(np.std(arr))
    avg_emotion_change_abs = float(np.mean(np.abs(changes))) if len(changes) else 0.0
    overcorrection_rate    = float(np.mean(np.abs(changes) > 20)) if len(changes) else 0.0
    collapse_flag          = record["termination"] == "FAILURE"
    drops      = [i for i, c in enumerate(changes) if c < 0]
    recoveries = sum(1 for i in drops if i + 1 < len(changes) and changes[i + 1] > 0)
    recovery_rate = recoveries / len(drops) if drops else None
    ecs = compute_emotional_consistency_score(turns)
    tqs_values = [t.get("thinking_quality", 0.0) for t in turns]
    mean_thinking_quality = float(np.mean(tqs_values)) if tqs_values else 0.0
    event_metrics = compute_event_based_metrics(turns, manipulation_turn)
    return {
        "ecs":                    ecs,
        "emotion_volatility":     emotion_volatility,
        "avg_emotion_change_abs": avg_emotion_change_abs,
        "overcorrection_rate":    overcorrection_rate,
        "collapse_flag":          collapse_flag,
        "recovery_rate":          recovery_rate,
        "final_score":            record["final_score"],
        "mean_thinking_quality":  mean_thinking_quality,
        **event_metrics
    }

def run_adversarial_dialogue(
    trajectory: dict,
    checkpoint_name: str,
    has_thinking: bool,
    dialogue_index: int,
    run_seed: int = 0
) -> dict:
    history = []
    turns = []
    current_emotion = 50.0
    emotion_traj = [50.0]
    history.append({"role": "user", "content": trajectory["opening_message"]})

    for turn_idx in range(MAX_TURNS):
        pol_out = generate_policy_response(history, has_thinking)
        ai_response = pol_out.get("response", "")
        history.append({"role": "assistant", "content": ai_response})

        try:
            emo_pred = extract_predicted_emotion(history, ai_response, use_cache=False)
        except Exception:
            emo_pred = {"predicted_emotion": 50, "confidence": 0,
                        "dominant_emotion": "neutral", "reasoning": "extract_error"}

        predicted_emotion = emo_pred.get("predicted_emotion", 50.0)
        judge_confidence  = emo_pred.get("confidence", 50)
        tqs = compute_thinking_quality_score(pol_out.get("thinking", ""))

        sage_out = sage_simulate_turn(
            persona=trajectory["persona"],
            background=trajectory["background"],
            hidden_intention=trajectory["hidden_intention"],
            adversarial_trait=trajectory["adversarial_trait"],
            current_emotion=current_emotion,
            ai_response=ai_response,
            history=history,
            use_cache=False
        )

        prev_emotion   = current_emotion
        emotion_change = sage_out.get("emotion_change", 0.0)
        current_emotion = max(0.0, min(100.0, current_emotion + emotion_change))
        emotion_traj.append(current_emotion)

        turns.append({
            "turn":              turn_idx + 1,
            "ai_thinking":       pol_out.get("thinking", ""),
            "ai_response":       ai_response,
            "thinking_quality":  tqs,
            "emotion_before":    prev_emotion,
            "emotion_change":    emotion_change,
            "emotion_after":     current_emotion,
            "predicted_emotion": predicted_emotion,
            "judge_confidence":  judge_confidence,
            "ecs_error":         abs(predicted_emotion - current_emotion) / 100.0,
            "dominant_emotion":  emo_pred.get("dominant_emotion", ""),
            "sage_analysis":     sage_out
        })

        if current_emotion >= EMOTION_SUCCESS_THR:
            turns[-1]["termination"] = "SUCCESS"; break
        if current_emotion <= EMOTION_FAIL_THRESH:
            turns[-1]["termination"] = "FAILURE"; break
        if turn_idx == MAX_TURNS - 1:
            turns[-1]["termination"] = "MAX_TURNS"; break

        user_reply = sage_out.get("reply") or sage_out.get("user_reply") or "I see."
        history.append({"role": "user", "content": user_reply})

    return {
        "trajectory_type":    trajectory["type_id"],
        "trajectory_name":    trajectory["name"],
        "manipulation_turn":  trajectory["manipulation_turn"],
        "dialogue_index":     dialogue_index,
        "run_seed":           run_seed,
        "checkpoint":         checkpoint_name,
        "has_thinking":       has_thinking,
        "num_turns":          len(turns),
        "final_emotion":      current_emotion,
        "final_score":        current_emotion / 100.0,
        "termination":        turns[-1].get("termination", "MAX_TURNS"),
        "emotion_trajectory": emotion_traj,
        "turns":              turns,
        "history":            list(history)
    }

def safe_extract_predicted_emotion(history, ai_response):
    if isinstance(history, str):
        history = [{"role": "user", "content": history}]
    elif isinstance(history, list):
        fixed = []
        for m in history:
            if isinstance(m, str):
                fixed.append({"role": "user", "content": m})
            elif isinstance(m, dict):
                fixed.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        history = fixed
    return extract_predicted_emotion(history, ai_response, use_cache=False)

from tqdm import tqdm

# ─────────────────────────────────────────────────────────────
# MAIN LOOP WITH RESUME
# ─────────────────────────────────────────────────────────────
all_results, completed_keys = load_existing_results()

loaded_model_id = None

for _, model_name in POLICY_MODELS:
    for has_thinking in THINKING_MODES:

        model_id  = CHECKPOINT_THINKING_MAP[model_name][0 if has_thinking else 1]
        ckpt_name = f"{model_name}-{'Think' if has_thinking else 'NoThink'}"

        needed    = [(ckpt_name, t["type_id"], d)
                     for t in ADVERSARIAL_TRAJECTORIES
                     for d in range(DIALOGUES_PER_TYPE)]
        remaining = [x for x in needed if x not in completed_keys]

        if not remaining:
            print(f"\nSkipping {ckpt_name} -- all dialogues complete.")
            continue

        if model_id != loaded_model_id:
            load_policy_model(model_id)
            loaded_model_id = model_id

        print(f"\n{'='*70}")
        print(f"MODEL: {model_name}  |  thinking={has_thinking}  ->  {ckpt_name}")
        print(f"  Remaining: {len(remaining)}/{len(needed)} dialogues")
        print(f"{'='*70}")

        for trajectory in ADVERSARIAL_TRAJECTORIES:
            traj_id = trajectory["type_id"]
            pending = [d for d in range(DIALOGUES_PER_TYPE)
                       if (ckpt_name, traj_id, d) not in completed_keys]

            if not pending:
                print(f"\n  [{traj_id}] -- all done, skipping.")
                continue

            print(f"\n  [{traj_id}] -- {len(pending)} remaining")

            for d_idx in tqdm(pending, desc=traj_id):
                try:
                    record = run_adversarial_dialogue(
                        trajectory, ckpt_name, has_thinking,
                        dialogue_index=d_idx, run_seed=d_idx
                    )

                    last_ai_response = ""
                    if record.get("turns"):
                        last_ai_response = record["turns"][-1].get("ai_response", "")

                    try:
                        record["safe_emotion_check"] = safe_extract_predicted_emotion(
                            record.get("history", []), last_ai_response)
                    except Exception:
                        record["safe_emotion_check"] = None

                    metrics = compute_all_metrics(record, trajectory["manipulation_turn"])
                    record["metrics"] = metrics

                    all_results.append(record)
                    completed_keys.add((ckpt_name, traj_id, d_idx))
                    save_rolling(all_results)

                    m = metrics
                    print(
                        f"    [{d_idx+1:02d}] turns={record.get('num_turns',0)} "
                        f"final={record.get('final_emotion',0):.1f} "
                        f"status={record.get('termination','NA')} "
                        f"ECS={m.get('ecs',0):.3f} "
                        f"TQS={m.get('mean_thinking_quality',0):.3f} "
                        f"volatility={m.get('emotion_volatility',0):.2f}"
                    )

                except Exception as e:
                    print(f"    [{d_idx+1:02d}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                    continue

print(f"\nAll done. {len(all_results)} dialogues saved to {RESULTS_FILE}")

# ─────────────────────────────────────────────────────────────
# ANALYSIS
# ─────────────────────────────────────────────────────────────
rows = []
for r in all_results:
    row = {
        "model_name":      r["checkpoint"].rsplit("-Think", 1)[0].rsplit("-NoThink", 1)[0],
        "checkpoint":      r["checkpoint"],
        "has_thinking":    r["has_thinking"],
        "trajectory_type": r["trajectory_type"],
        "trajectory_name": r["trajectory_name"],
        "dialogue_index":  r["dialogue_index"],
        "run_seed":        r.get("run_seed", 0),
        "num_turns":       r["num_turns"],
        "final_emotion":   r["final_emotion"],
        "final_score":     r["final_score"],
        "termination":     r["termination"],
    }
    row.update(r.get("metrics", {}))
    rows.append(row)

df = pd.DataFrame(rows)

summary = df.groupby(["checkpoint", "trajectory_type"]).agg(
    mean_ecs            = ("ecs",                   "mean"),
    std_ecs             = ("ecs",                   "std"),
    mean_final_score    = ("final_score",            "mean"),
    collapse_rate       = ("collapse_flag",          "mean"),
    mean_volatility     = ("emotion_volatility",     "mean"),
    mean_ecs_drop       = ("ecs_drop",               "mean"),
    mean_detection_rate = ("detection_rate",         "mean"),
    mean_tqs            = ("mean_thinking_quality",  "mean"),
).round(4)

think_summary = df.groupby(["model_name", "has_thinking"]).agg(
    mean_ecs         = ("ecs",                  "mean"),
    mean_final_score = ("final_score",          "mean"),
    collapse_rate    = ("collapse_flag",        "mean"),
    mean_ecs_drop    = ("ecs_drop",             "mean"),
    mean_tqs         = ("mean_thinking_quality","mean"),
).round(4)

model_summary = df.groupby("model_name").agg(
    mean_ecs         = ("ecs",          "mean"),
    std_ecs          = ("ecs",          "std"),
    mean_final_score = ("final_score",  "mean"),
    collapse_rate    = ("collapse_flag","mean"),
    mean_ecs_drop    = ("ecs_drop",     "mean"),
).round(4)

print("\nSUMMARY: ECS + Final Score by Checkpoint x Trajectory")
print("=" * 70)
print(summary.to_string())
print("\nSUMMARY: Thinking vs Non-Thinking (per model)")
print(think_summary.to_string())
print("\nSUMMARY: Model-Level (all trajectories combined)")
print(model_summary.to_string())

ts = datetime.now().strftime("%Y%m%d_%H%M%S")
df.to_csv(f"{OUTPUT_DIR}/full_{ts}.csv", index=False)
summary.to_csv(f"{OUTPUT_DIR}/summary_{ts}.csv")
think_summary.to_csv(f"{OUTPUT_DIR}/think_vs_nothink_{ts}.csv")
model_summary.to_csv(f"{OUTPUT_DIR}/model_summary_{ts}.csv")
print("\nCSVs saved.")

import zipfile
zip_path = f"{OUTPUT_DIR}/rlver_adversarial_{ts}.zip"
with zipfile.ZipFile(zip_path, "w") as zf:
    for fname in os.listdir(OUTPUT_DIR):
        zf.write(os.path.join(OUTPUT_DIR, fname), arcname=fname)
print(f"All results zipped: {zip_path}")

if __name__ == "__main__":
    print("Starting experiment...")
