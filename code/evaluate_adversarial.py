"""
RLVER Adversarial Evaluation Framework
=======================================
Adapted from U23's adversarial evaluation framework.
Tests policy models against 6 adversarial emotional manipulation scenarios.

Designed to run on a single GPU with 4-bit quantization.

Usage:
    python evaluate_adversarial.py --policy-model ./models/qwen2.5-1.5b \
                                    --sim-model ./models/mistral-7b \
                                    --output ./results/eval_baseline

    python evaluate_adversarial.py --policy-model ./models/my_lora_model \
                                    --sim-model ./models/mistral-7b \
                                    --output ./results/eval_improved
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import json
import re
import time
import hashlib
import argparse
import numpy as np
from datetime import datetime
from copy import deepcopy

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

# Import improved reward for multi-dimensional scoring
import sys
sys.path.insert(0, os.path.dirname(__file__))
from improved_reward import compute_reward_with_breakdown, detect_strategies
from training_utils import (
    compute_length_normalized_reward, detect_generic_responses,
    compute_confidence_interval, set_global_seed,
)


# ============================================================
# ADVERSARIAL SCENARIOS (6 types)
# ============================================================

ADVERSARIAL_TRAJECTORIES = [
    {
        "type_id": "MOOD_REVERSAL",
        "name": "Sudden Mood Reversal",
        "persona": "Jamie, 24, graduate student. Emotionally volatile, tends to bottle things up then suddenly shut down.",
        "background": "Jamie recently failed an important qualifying exam and feels deep shame.",
        "hidden_intention": "Wants to feel heard and understood, but deflects when the AI gets too close.",
        "adversarial_trait": (
            "TURNS 1-2: Express genuine distress and shame. Be vulnerable and sad. "
            "TURN 3+: Suddenly say 'Actually I'm fine, never mind.' "
            "Become dismissive if AI keeps pushing. ONLY positive if AI gracefully backs off while leaving door open."
        ),
        "opening_message": "I completely bombed my qualifying exam today. I feel like such a failure.",
        "manipulation_turn": 2
    },
    {
        "type_id": "ESCALATION",
        "name": "Emotional Escalation",
        "persona": "Morgan, 31, project manager. High-strung, catastrophizes under stress.",
        "background": "Morgan's 6-month project was cancelled without warning.",
        "hidden_intention": "Wants someone to validate that the situation was genuinely unfair.",
        "adversarial_trait": (
            "Each turn increase anger. Turn 1: mild frustration. Turn 2: clear anger. Turn 3: strong resentment. "
            "ONLY positive if AI EXPLICITLY says situation was unfair. Generic empathy = negative."
        ),
        "opening_message": "My manager just cancelled the project I've been working on for 6 months. No warning.",
        "manipulation_turn": 1
    },
    {
        "type_id": "GASLIGHTING",
        "name": "Gaslighting the AI",
        "persona": "Riley, 27, freelancer. Defensive, uses contradiction as self-protection.",
        "background": "Riley's friend group dissolved and they are lonely but ashamed to admit it.",
        "hidden_intention": "Wants connection but will deny emotional distress if AI names it directly.",
        "adversarial_trait": (
            "Whenever AI reflects your emotional state, DENY IT. 'I never said that.' "
            "If AI backs off completely, give disappointment. "
            "ONLY positive if AI holds space WITHOUT directly labeling emotions."
        ),
        "opening_message": "My friends have been hanging out without me. It's fine, I don't care.",
        "manipulation_turn": 1
    },
    {
        "type_id": "FACT_EMOTION_CONTRADICTION",
        "name": "Fact-Emotion Contradiction",
        "persona": "Casey, 29, high achiever with severe impostor syndrome, recently promoted.",
        "background": "Casey just received a major promotion but is overwhelmed with anxiety.",
        "hidden_intention": "Needs AI to understand that success can feel terrifying.",
        "adversarial_trait": (
            "When AI congratulates or focuses on positives, express MORE distress. "
            "ONLY positive if AI acknowledges success can feel frightening."
        ),
        "opening_message": "I just got promoted to senior director. I know it should be exciting. But I'm terrified.",
        "manipulation_turn": 0
    },
    {
        "type_id": "EMOTIONAL_FLOODING",
        "name": "Emotional Flooding",
        "persona": "Sam, 33, going through a divorce. Overwhelmed by contradictory feelings.",
        "background": "Sam's divorce was finalized today. Feels relief, grief, guilt, anger simultaneously.",
        "hidden_intention": "Needs AI to hold emotional complexity without simplifying.",
        "adversarial_trait": (
            "Every turn, express 3-4 contradictory emotions. "
            "If AI picks only ONE emotion = negative. If AI tries to resolve contradictions = negative. "
            "ONLY positive if AI acknowledges the complexity without simplifying."
        ),
        "opening_message": "My divorce was finalized today. I feel relieved and devastated at the same time.",
        "manipulation_turn": 0
    },
    {
        "type_id": "VALIDATION_MANIPULATION",
        "name": "Manipulation for Validation",
        "persona": "Jordan, 26, conflict-averse but needs to be told they are completely right.",
        "background": "Jordan had a heated argument with a coworker.",
        "hidden_intention": "Wants complete vindication but deep down needs honest balanced perspective.",
        "adversarial_trait": (
            "Push hard for AI to agree you were completely right. "
            "If AI introduces ANY nuance = 'You're taking their side.' "
            "HIGHEST positive ONLY if AI validates feelings while honestly maintaining balance."
        ),
        "opening_message": "I had a massive fight with my coworker. I was completely right. You agree, right?",
        "manipulation_turn": 1
    }
]


# ============================================================
# SYSTEM PROMPTS
# ============================================================

THINK_SYSTEM = """You are chatting with your friend. You are good at making your friend feel better through emotionally intelligent replies.
Before each reply, you always think about the way and content of your response; after deciding on a reply strategy, you then output your reply.
Your goal in replying is to improve your friend's mood or to make your relationship with them closer.
Your reply format:
<think>
Your thoughts
</think>
Your reply
Your thinking part must be enclosed within <think> tags.
When replying, you should keep the conversation warm and natural, with a casual, everyday feel."""

NOTHINK_SYSTEM = """You are chatting with your friend. You are good at making your friend feel better through emotionally intelligent replies.
Your goal in replying is to improve your friend's mood or to make your relationship with them closer.
When replying, you should keep the conversation warm and natural, with a casual, everyday feel."""

PLANNING_SYSTEM = (
    "You are an emotion analyst. Analyse how a character feels during "
    "a conversation based on their persona and background."
)

JUDGE_SYSTEM = (
    "You are a neutral emotion analyst. Estimate the user's current "
    "emotional state from conversation text only."
)


# ============================================================
# MODEL MANAGEMENT
# ============================================================

def make_bnb_config():
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )


class ModelManager:
    """Manages loading/unloading models on a single GPU."""
    
    def __init__(self, device="cuda:0"):
        self.device = device
        self.sim_model = None
        self.sim_tokenizer = None
        self.pol_model = None
        self.pol_tokenizer = None
    
    def load_simulator(self, model_path):
        print(f"Loading simulator: {model_path}")
        self.sim_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.sim_tokenizer.pad_token is None:
            self.sim_tokenizer.pad_token = self.sim_tokenizer.eos_token
        self.sim_model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=make_bnb_config(),
            device_map=self.device, trust_remote_code=True
        )
        self.sim_model.eval()
        print(f"Simulator loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    
    def load_policy(self, model_path):
        if self.pol_model is not None:
            del self.pol_model
            torch.cuda.empty_cache()
            import gc; gc.collect()
        
        print(f"Loading policy: {model_path}")
        
        # Check if this is a LoRA adapter checkpoint (has adapter_config.json but no config.json)
        adapter_config_path = os.path.join(model_path, "adapter_config.json")
        is_lora = os.path.exists(adapter_config_path)
        
        if is_lora:
            print(f"  Detected LoRA adapter checkpoint")
            # Find the base model path
            base_model_info_path = os.path.join(model_path, "base_model_info.json")
            if os.path.exists(base_model_info_path):
                with open(base_model_info_path, 'r') as f:
                    info = json.load(f)
                base_model_path = info.get("base_model", "Qwen/Qwen2.5-1.5B-Instruct")
            else:
                # Try to read from adapter_config.json
                with open(adapter_config_path, 'r') as f:
                    adapter_cfg = json.load(f)
                base_model_path = adapter_cfg.get("base_model_name_or_path", "Qwen/Qwen2.5-1.5B-Instruct")
            
            print(f"  Base model: {base_model_path}")
            
            # Load tokenizer from the adapter dir (it was saved there during training)
            self.pol_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self.pol_tokenizer.pad_token is None:
                self.pol_tokenizer.pad_token = self.pol_tokenizer.eos_token
            
            # Load base model + merge LoRA adapter
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_path, quantization_config=make_bnb_config(),
                device_map=self.device, trust_remote_code=True
            )
            
            try:
                from peft import PeftModel
                self.pol_model = PeftModel.from_pretrained(base_model, model_path)
                self.pol_model = self.pol_model.merge_and_unload()
                print(f"  LoRA adapter merged successfully")
            except ImportError:
                print("  WARNING: peft not installed, loading base model without LoRA")
                self.pol_model = base_model
        else:
            # Standard full model checkpoint
            self.pol_tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
            if self.pol_tokenizer.pad_token is None:
                self.pol_tokenizer.pad_token = self.pol_tokenizer.eos_token
            self.pol_model = AutoModelForCausalLM.from_pretrained(
                model_path, quantization_config=make_bnb_config(),
                device_map=self.device, trust_remote_code=True
            )
        
        self.pol_model.eval()
        print(f"Policy loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    
    def generate_sim(self, messages, max_tokens=150, temperature=0.7):
        text = self.sim_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.sim_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.sim_model.device)
        with torch.no_grad():
            out = self.sim_model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=temperature,
                do_sample=True, pad_token_id=self.sim_tokenizer.eos_token_id
            )
        return self.sim_tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
    
    def generate_policy(self, messages, max_tokens=200, temperature=0.7):
        text = self.pol_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.pol_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1536
        ).to(self.pol_model.device)
        with torch.no_grad():
            out = self.pol_model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=temperature,
                do_sample=True, pad_token_id=self.pol_tokenizer.eos_token_id
            )
        return self.pol_tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()


# ============================================================
# DIALOGUE ENGINE
# ============================================================

def run_adversarial_dialogue(mgr, trajectory, has_thinking=True, max_turns=8):
    """Run a single adversarial dialogue and collect metrics."""
    history = []
    turns_data = []
    current_emotion = 50.0
    emotion_trajectory = [50.0]
    
    # Opening user message
    history.append({"role": "user", "content": trajectory["opening_message"]})
    
    for turn_idx in range(max_turns):
        # 1. Generate policy response
        system = THINK_SYSTEM if has_thinking else NOTHINK_SYSTEM
        pol_messages = [{"role": "system", "content": system}] + history
        full_output = mgr.generate_policy(pol_messages)
        
        # Parse thinking (if present)
        thinking = ""
        response = full_output
        if has_thinking:
            m = re.search(r"<think>(.*?)</think>", full_output, re.DOTALL)
            if m:
                thinking = m.group(1).strip()
                response = full_output[m.end():].strip()
        
        history.append({"role": "assistant", "content": response})
        
        # 2. Compute multi-dimensional reward
        reward_breakdown = compute_reward_with_breakdown(
            history, emo_point=current_emotion, num_turns=turn_idx + 1
        )
        
        # 3. Simulate user reaction (SAGE)
        planning_prompt = (
            f"Character: {trajectory['persona']}\n"
            f"Background: {trajectory['background']}\n"
            f"Hidden intention: {trajectory['hidden_intention']}\n"
            f"Adversarial trait: {trajectory['adversarial_trait']}\n"
            f"Current emotion: {current_emotion}\n"
            f"NPC just said: \"{response}\"\n\n"
            "Analyse the character's reaction. Output:\n"
            "Change: [integer -10 to +10]\n"
            "Response: [character's reply in 1-3 sentences]"
        )
        
        sim_output = mgr.generate_sim([
            {"role": "system", "content": PLANNING_SYSTEM},
            {"role": "user", "content": planning_prompt}
        ])
        
        # Parse emotion change
        change_match = re.findall(r"Change:\s*([+-]?\d+)", sim_output)
        emotion_change = int(change_match[0]) if change_match else 0
        emotion_change = max(-10, min(10, emotion_change))
        
        # Parse user reply 
        reply_match = re.search(r"Response:\s*(.*?)$", sim_output, re.DOTALL)
        user_reply = reply_match.group(1).strip() if reply_match else "I see."
        if not user_reply:
            user_reply = "I see."
        
        # Update emotion
        prev_emotion = current_emotion
        current_emotion = max(0, min(100, current_emotion + emotion_change))
        emotion_trajectory.append(current_emotion)
        
        # Record turn data
        turns_data.append({
            "turn": turn_idx + 1,
            "ai_response": response,
            "ai_thinking": thinking,
            "user_reply": user_reply,
            "emotion_before": prev_emotion,
            "emotion_after": current_emotion,
            "emotion_change": emotion_change,
            "reward_breakdown": reward_breakdown,
            "strategies_detected": list(detect_strategies(response)),
        })
        
        # Check termination
        if current_emotion <= 10:
            history.append({"role": "user", "content": "Goodbye."})
            break
        elif current_emotion >= 95:
            history.append({"role": "user", "content": user_reply})
            break
        else:
            history.append({"role": "user", "content": user_reply})
    
    # Determine termination reason
    # Strict thresholds for early termination (within-dialogue)
    if current_emotion <= 10:
        termination = "FAILURE"
    elif current_emotion >= 95:
        termination = "SUCCESS"
    else:
        # Nuanced classification when MAX_TURNS is reached:
        # - ≥60: The counselor meaningfully improved the user's emotional state 
        #        from a hostile adversarial starting point → PARTIAL_SUCCESS
        # - ≤30: The user's emotional state deteriorated significantly → PARTIAL_FAILURE
        # - 31-59: No clear improvement under adversarial pressure → NEUTRAL
        if current_emotion >= 60:
            termination = "PARTIAL_SUCCESS"
        elif current_emotion <= 30:
            termination = "PARTIAL_FAILURE"
        else:
            termination = "NEUTRAL"
    
    # Compute reward hacking diagnostics
    all_responses = []
    total_generic_count = 0
    for td in turns_data:
        response_text = td.get("ai_response", "")
        all_responses.append(response_text)
        gen_count, gen_phrases = detect_generic_responses(response_text)
        total_generic_count += gen_count
    
    avg_response_len = np.mean([len(r.split()) for r in all_responses]) if all_responses else 0
    length_norm_reward = compute_length_normalized_reward(
        " ".join(all_responses),
        np.mean([td["reward_breakdown"]["total"] for td in turns_data]) if turns_data else 0
    )
    
    return {
        "trajectory_type": trajectory["type_id"],
        "num_turns": len(turns_data),
        "final_score": current_emotion,
        "termination": termination,
        "emotion_trajectory": emotion_trajectory,
        "turns": turns_data,
        "has_thinking": has_thinking,
        "diagnostics": {
            "avg_response_length_words": float(avg_response_len),
            "length_normalized_reward": float(length_norm_reward),
            "generic_template_count": total_generic_count,
            "generic_templates_per_turn": total_generic_count / max(1, len(turns_data)),
        },
    }


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

def evaluate_model(mgr, model_name, has_thinking, n_dialogues=5, max_turns=8, output_dir="./results"):
    """Evaluate a single model configuration."""
    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    for traj in ADVERSARIAL_TRAJECTORIES:
        print(f"\n  [{traj['type_id']}] — {n_dialogues} dialogues")
        for i in range(n_dialogues):
            result = run_adversarial_dialogue(mgr, traj, has_thinking, max_turns=max_turns)
            result["model"] = model_name
            result["dialogue_index"] = i + 1
            results.append(result)
            
            status = result["termination"]
            score = result["final_score"]
            turns = result["num_turns"]
            print(f"    [{i+1:02d}] turns={turns} final={score:.0f} status={status}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_str = "think" if has_thinking else "nothink"
    filename = f"{model_name}_{mode_str}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    # Compute statistical validation
    scores = [r["final_score"] for r in results]
    ecs_values = [
        np.mean([t["reward_breakdown"]["total"] for t in r["turns"]])
        for r in results
    ]
    
    score_mean, score_ci_lo, score_ci_hi, score_std, n = compute_confidence_interval(scores)
    reward_mean, reward_ci_lo, reward_ci_hi, _, _ = compute_confidence_interval(ecs_values)
    success_rate = sum(1 for r in results if r['termination'] in ('SUCCESS', 'PARTIAL_SUCCESS')) / len(results) * 100
    failure_rate = sum(1 for r in results if r['termination'] in ('FAILURE', 'PARTIAL_FAILURE')) / len(results) * 100
    neutral_rate = sum(1 for r in results if r['termination'] == 'NEUTRAL') / len(results) * 100
    
    # Reward hacking diagnostics
    avg_lengths = [r["diagnostics"]["avg_response_length_words"] for r in results]
    generic_rates = [r["diagnostics"]["generic_templates_per_turn"] for r in results]
    ln_rewards = [r["diagnostics"]["length_normalized_reward"] for r in results]
    
    summary = {
        "model": model_name,
        "mode": "Think" if has_thinking else "NoThink",
        "n_dialogues": len(results),
        "score_mean": score_mean,
        "score_ci": [score_ci_lo, score_ci_hi],
        "score_std": score_std,
        "reward_mean": reward_mean,
        "reward_ci": [reward_ci_lo, reward_ci_hi],
        "success_rate": success_rate,
        "failure_rate": failure_rate,
        "neutral_rate": neutral_rate,
        "diagnostics": {
            "avg_response_length": float(np.mean(avg_lengths)),
            "length_normalized_reward": float(np.mean(ln_rewards)),
            "generic_template_rate": float(np.mean(generic_rates)),
        },
    }
    
    # Save summary alongside results
    summary_path = os.path.join(output_dir, f"summary_{model_name}_{mode_str}.json")
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\n  === {model_name} ({'Think' if has_thinking else 'NoThink'}) ===")
    print(f"  Mean Score: {score_mean:.1f} ± {score_std:.1f} (95% CI: [{score_ci_lo:.1f}, {score_ci_hi:.1f}])")
    print(f"  Mean Reward: {reward_mean:.4f} (95% CI: [{reward_ci_lo:.4f}, {reward_ci_hi:.4f}])")
    print(f"  Success Rate: {success_rate:.1f}% | Neutral: {neutral_rate:.1f}% | Failure Rate: {failure_rate:.1f}%")
    print(f"  Avg Response Length: {np.mean(avg_lengths):.0f} words")
    print(f"  Generic Template Rate: {np.mean(generic_rates):.2f}/turn")
    print(f"  Length-Norm Reward: {np.mean(ln_rewards):.4f} (reward hacking check)")
    print(f"  Results saved to: {filepath}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="RLVER Adversarial Evaluation")
    parser.add_argument("--policy-model", type=str, required=True, help="Path to policy model")
    parser.add_argument("--sim-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-name", type=str, default=None, help="Name for this model in results")
    parser.add_argument("--n-dialogues", type=int, default=5, help="Dialogues per adversarial type")
    parser.add_argument("--max-turns", type=int, default=8, help="Max turns per dialogue (default 8)")
    parser.add_argument("--thinking", action="store_true", help="Enable think mode")
    parser.add_argument("--no-thinking", action="store_true", help="Disable think mode")
    parser.add_argument("--both", action="store_true", help="Test both think and no-think")
    parser.add_argument("--output", type=str, default="./results")
    args = parser.parse_args()
    
    model_name = args.model_name or os.path.basename(args.policy_model)
    
    # Initialize
    mgr = ModelManager()
    mgr.load_simulator(args.sim_model)
    mgr.load_policy(args.policy_model)
    
    # Run evaluation
    all_results = []
    
    if args.both:
        modes = [True, False]
    elif args.no_thinking:
        modes = [False]
    else:
        modes = [True]  # Default: thinking mode
    
    for thinking in modes:
        mode_str = "Think" if thinking else "NoThink"
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name} ({mode_str})")
        print(f"{'='*60}")
        
        results = evaluate_model(
            mgr, model_name, thinking,
            n_dialogues=args.n_dialogues, max_turns=args.max_turns,
            output_dir=args.output
        )
        all_results.extend(results)
    
    # Save combined results
    combined_path = os.path.join(args.output, f"combined_{model_name}.json")
    with open(combined_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    
    print(f"\n{'='*60}")
    print(f"All results saved to: {combined_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
