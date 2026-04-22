"""
RLVER Improved Training via LoRA + Multi-Dimensional Reward
============================================================
Single-GPU RL training script using:
1. LoRA (Low-Rank Adaptation) for memory-efficient fine-tuning
2. Multi-dimensional reward function (from improved_reward.py)
3. Curriculum learning (easy -> hard profiles)
4. Local Qwen2.5-7B simulator instead of GPT-4o API
5. Proper REINFORCE with differentiable log-prob recomputation

Designed for: 1x NVIDIA RTX PRO 4500 Blackwell (32GB VRAM)

Usage:
    python train_improved.py \
        --base-model ./models/qwen2.5-1.5b \
        --sim-model ./models/qwen2.5-7b-sim \
        --train-data ../data/train_profile.jsonl \
        --output-dir ./checkpoints/improved_v1 \
        --epochs 3 --lr 2e-5 --curriculum
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import gc
import json
import re
import time
import random
import argparse
import numpy as np
from datetime import datetime
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("WARNING: peft not installed. Run: pip install peft")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from improved_reward import compute_multi_dimensional_reward, detect_strategies
from training_utils import (
    set_global_seed, save_training_state, load_training_state,
    StabilityLogger,
)


# ============================================================
# SYSTEM PROMPTS (bilingual for Chinese dataset)
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
When replying, you should keep the conversation warm and natural, with a casual, everyday feel.
IMPORTANT: Reply in the SAME LANGUAGE as the user. If they speak Chinese, reply in Chinese. If they speak English, reply in English."""

NOTHINK_SYSTEM = """You are chatting with your friend. You are good at making your friend feel better through emotionally intelligent replies.
Your goal in replying is to improve your friend's mood or to make your relationship with them closer.
When replying, you should keep the conversation warm and natural, with a casual, everyday feel.
IMPORTANT: Reply in the SAME LANGUAGE as the user. If they speak Chinese, reply in Chinese. If they speak English, reply in English."""


# ============================================================
# DATA LOADING
# ============================================================

class EmotionProfileDataset(Dataset):
    """Load empathy profiles from JSONL and organize by difficulty."""
    
    def __init__(self, jsonl_path, curriculum=False):
        self.profiles = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    self.profiles.append(json.loads(line))
        
        self.curriculum = curriculum
        if curriculum:
            self.profiles.sort(key=lambda p: len(p.get('player', '')))
        
        print(f"Loaded {len(self.profiles)} profiles")
    
    def __len__(self):
        return len(self.profiles)
    
    def __getitem__(self, idx):
        return self.profiles[idx]
    
    def get_curriculum_batch(self, epoch, total_epochs, batch_size):
        """Return profiles appropriate for current training stage."""
        if not self.curriculum:
            indices = random.sample(range(len(self)), min(batch_size, len(self)))
            return [self.profiles[i] for i in indices]
        
        progress = epoch / max(total_epochs, 1)
        n = len(self.profiles)
        
        if progress < 0.3:
            pool = self.profiles[:n // 3]
        elif progress < 0.7:
            pool = self.profiles[:2 * n // 3]
        else:
            pool = self.profiles
        
        indices = random.sample(range(len(pool)), min(batch_size, len(pool)))
        return [pool[i] for i in indices]


# ============================================================
# LOCAL SIMULATOR
# ============================================================

class LocalSimulator:
    """Local Qwen2.5-7B based simulator (replaces GPT-4o API)."""
    
    def __init__(self, model_path, device="cuda:0"):
        self.device = device
        print(f"Loading simulator: {model_path}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config,
            device_map=device, trust_remote_code=True
        )
        self.model.eval()
        print(f"Simulator loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    
    def generate(self, messages, max_tokens=150, temperature=0.7):
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            out = self.model.generate(
                **inputs, max_new_tokens=max_tokens, temperature=temperature,
                do_sample=True, pad_token_id=self.tokenizer.eos_token_id
            )
        return self.tokenizer.decode(
            out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
    
    def simulate_turn(self, profile, history, ai_response, current_emotion):
        """Simulate user's emotional reaction to AI's response."""
        prompt = (
            f"You are acting as a character in a dialogue simulation. Analyze how the character would react.\n\n"
            f"Character: {profile.get('player', '')}\n"
            f"Scene: {profile.get('scene', '')}\n"
            f"Task/Hidden topic: {profile.get('task', '')}\n"
            f"Current emotion level: {current_emotion}/100\n"
            f"AI just said: \"{ai_response}\"\n\n"
            f"Analyze the character's emotional reaction. The character profile may be in Chinese - respond accordingly.\n"
            f"Output EXACTLY in this format:\n"
            f"Change: [integer from -10 to +10]\n"
            f"Response: [character's reply, 1-3 sentences, in the SAME language as the character profile]"
        )
        
        raw = self.generate([
            {"role": "system", "content": "You are an emotion analyst playing a character."},
            {"role": "user", "content": prompt}
        ])
        
        # Parse change
        change_match = re.findall(r"Change:\s*([+-]?\d+)", raw)
        change = int(change_match[0]) if change_match else 0
        change = max(-10, min(10, change))
        
        # Parse response
        resp_match = re.search(r"Response:\s*(.*?)$", raw, re.DOTALL)
        user_reply = resp_match.group(1).strip() if resp_match else "I see."
        if not user_reply:
            user_reply = "I see."
        
        new_emotion = max(0, min(100, current_emotion + change))
        return new_emotion, change, user_reply


# ============================================================
# FIRST MESSAGE GENERATION
# ============================================================

def generate_first_message(simulator, profile):
    """Generate the first user message from a profile using the simulator."""
    prompt = (
        f"You are playing a character in a dialogue simulation.\n"
        f"Character: {profile.get('player', '')}\n"
        f"Scene: {profile.get('scene', '')}\n"
        f"Task/Hidden topic: {profile.get('task', '')}\n\n"
        f"Generate a SHORT opening message (1-2 sentences) that this character "
        f"would naturally say to start a conversation with a friend. "
        f"The message should hint at their emotional state without being too direct.\n"
        f"IMPORTANT: Write ONLY the character's message, nothing else. "
        f"Write in the SAME language as the character profile above."
    )
    
    try:
        msg = simulator.generate([
            {"role": "system", "content": "You are a character actor. Output only the character's dialogue line."},
            {"role": "user", "content": prompt}
        ], max_tokens=100, temperature=0.9)
        msg = msg.strip().strip('"').strip("'")
        if msg.startswith("Character:") or msg.startswith("Opening:"):
            msg = msg.split(":", 1)[1].strip()
        if len(msg) > 5:
            return msg
    except Exception:
        pass
    
    # Fallback
    scene = profile.get('scene', '')
    task = profile.get('task', '')
    if scene and len(scene) > 10:
        first_sentence = scene.split('\u3002')[0].split('.')[0]
        if len(first_sentence) > 5:
            return first_sentence[:200]
    if task and len(task) > 10:
        return task[:200]
    
    return "\u6211\u6700\u8fd1\u5fc3\u60c5\u4e0d\u592a\u597d\uff0c\u60f3\u627e\u4e2a\u4eba\u804a\u804a..."


# ============================================================
# TRAINING EPISODE (sampling only, no gradients needed)
# ============================================================

def run_training_episode(policy_model, policy_tokenizer, simulator, profile,
                          has_thinking=True, max_turns=6, device="cuda:0"):
    """Run one training episode. Returns sampled sequences for later gradient computation."""
    system = THINK_SYSTEM if has_thinking else NOTHINK_SYSTEM
    
    history = []
    generated_sequences = []  # List of (input_ids_tensor, generated_ids_tensor)
    rewards_list = []
    
    first_message = generate_first_message(simulator, profile)
    history.append({"role": "user", "content": first_message})
    current_emotion = 30.0
    
    for turn in range(max_turns):
        messages = [{"role": "system", "content": system}] + history
        text = policy_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = policy_tokenizer(
            text, return_tensors="pt", truncation=True, max_length=2048
        ).to(device)
        
        # SAMPLE tokens without gradients (generation is not differentiable)
        with torch.no_grad():
            outputs = policy_model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.8,
                do_sample=True,
                pad_token_id=policy_tokenizer.eos_token_id,
            )
        
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        
        if len(generated_ids) == 0:
            history.append({"role": "assistant", "content": "I'm here for you."})
            rewards_list.append(0.1)
            break
        
        response = policy_tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        if not response:
            response = "I'm here for you."
        
        # Save (input_ids, generated_ids) for differentiable log-prob computation
        generated_sequences.append((
            inputs["input_ids"].detach().clone(),
            generated_ids.detach().clone()
        ))
        
        clean_response = response
        if has_thinking:
            m = re.search(r"<think>(.*?)</think>", response, re.DOTALL)
            if m:
                clean_response = response[m.end():].strip()
        
        history.append({"role": "assistant", "content": clean_response})
        
        new_emotion, change, user_reply = simulator.simulate_turn(
            profile, history, clean_response, current_emotion
        )
        
        reward = compute_multi_dimensional_reward(
            history, emo_point=new_emotion, num_turns=turn + 1
        )
        rewards_list.append(reward)
        current_emotion = new_emotion
        
        if current_emotion <= 10 or current_emotion >= 95:
            break
        
        history.append({"role": "user", "content": user_reply})
    
    return generated_sequences, rewards_list, history, current_emotion


# ============================================================
# DIFFERENTIABLE LOG-PROB RECOMPUTATION (REINFORCE trick)
# ============================================================

def compute_differentiable_log_probs(model, sequences, device="cuda:0"):
    """Second forward pass WITH gradients to get differentiable log-probs.
    
    REINFORCE requires:
    1. Sample actions (tokens) without gradients  -> done in run_training_episode
    2. Compute log pi(a|s) WITH gradients         -> done HERE
    3. Policy gradient: L = -sum(log pi * R)      -> done in training loop
    """
    log_probs = []
    
    for input_ids, generated_ids in sequences:
        full_seq = torch.cat([input_ids[0], generated_ids], dim=0).unsqueeze(0).to(device)
        
        # Forward pass WITH gradients enabled
        outputs = model(input_ids=full_seq, use_cache=False)
        logits = outputs.logits
        
        prompt_len = input_ids.shape[1]
        gen_len = len(generated_ids)
        
        # logits[t] predicts token[t+1]
        gen_logits = logits[0, prompt_len - 1 : prompt_len + gen_len - 1, :]
        log_probs_per_token = F.log_softmax(gen_logits, dim=-1)
        
        selected_log_probs = log_probs_per_token.gather(
            1, generated_ids.unsqueeze(1).to(device)
        ).squeeze(1)
        
        total_log_prob = selected_log_probs.sum()
        log_probs.append(total_log_prob)
    
    return log_probs


def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns with normalization."""
    if not rewards:
        return torch.tensor([], dtype=torch.float32)
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns, dtype=torch.float32)
    if len(returns) > 1:
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
    return returns


# ============================================================
# MAIN TRAINING LOOP
# ============================================================

def train_with_reinforce(args):
    """Main training loop using REINFORCE algorithm with LoRA.
    
    Optionally adds DPO refinement stage if --dpo flag is set.
    """
    device = "cuda:0"
    set_global_seed(getattr(args, 'seed', 42))
    
    simulator = LocalSimulator(args.sim_model, device=device)
    
    # For DPO: load reference model
    ref_model = None
    if getattr(args, 'dpo', False):
        print("Loading reference model for DPO...")
        bnb_ref = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4"
        )
        ref_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, quantization_config=bnb_ref,
            device_map=device, trust_remote_code=True
        )
        ref_model.eval()
        for p in ref_model.parameters():
            p.requires_grad = False
        print(f"Reference model loaded. VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    
    print(f"\nLoading policy model: {args.base_model}")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4"
    )
    
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config,
        device_map=device, trust_remote_code=True
    )
    
    if HAS_PEFT:
        model = prepare_model_for_kbit_training(model)
        lora_config = LoraConfig(
            r=16, lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01
    )
    
    dataset = EmotionProfileDataset(args.train_data, curriculum=args.curriculum)
    
    # Save base model path for evaluation later
    base_model_info = {"base_model": args.base_model}
    
    train_log = {
        "config": vars(args),
        "algorithm": "REINFORCE+DPO" if getattr(args, 'dpo', False) else "REINFORCE",
        "episodes": [],
        "dpo_steps": [],
        "epoch_stats": [],
        "stability": {},
    }
    
    stability = StabilityLogger(window_size=5)
    all_preference_pairs = []  # For DPO stage
    
    # Try to resume from checkpoint
    start_epoch = 0
    best_avg_reward = -float('inf')
    
    resume_state = load_training_state(args.output_dir, model, optimizer)
    if resume_state is not None:
        start_epoch, _, best_avg_reward, saved_log = resume_state
        if saved_log is not None:
            train_log = saved_log
    
    print(f"\n{'='*60}")
    print(f"Starting LoRA Training ({train_log['algorithm']})")
    print(f"Base model: {args.base_model}")
    print(f"Epochs: {args.epochs}, Episodes/epoch: {args.episodes_per_epoch}")
    print(f"Curriculum: {args.curriculum}, LR: {args.lr}")
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch + 1}")
    print(f"{'='*60}\n")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_rewards = []
        epoch_scores = []
        
        profiles = dataset.get_curriculum_batch(
            epoch, args.epochs, args.episodes_per_epoch
        )
        
        for ep_idx, profile in enumerate(profiles):
            model.train()
            
            try:
                # Step 1: Sample episode (no gradients)
                sequences, rewards, history, final_emo = run_training_episode(
                    model, tokenizer, simulator, profile,
                    has_thinking=args.thinking, max_turns=args.max_turns, device=device
                )
                
                if not sequences or not rewards:
                    print(f"    (skipped - empty episode)")
                    continue
                
                # Step 2: Recompute differentiable log-probs
                log_probs = compute_differentiable_log_probs(model, sequences, device=device)
                
                if not log_probs:
                    continue
                
                # Step 3: REINFORCE loss = -sum(log_prob * return)
                returns = compute_returns(rewards, gamma=0.99)
                if len(returns) == 0:
                    continue
                
                loss = torch.tensor(0.0, device=device)
                n_terms = min(len(log_probs), len(returns))
                for i in range(n_terms):
                    loss = loss - log_probs[i] * returns[i].to(device)
                
                if not loss.requires_grad:
                    continue
                
                # Step 4: Backprop
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                
                avg_reward = float(np.mean(rewards))
                epoch_rewards.append(avg_reward)
                epoch_scores.append(final_emo)
                
                # Collect preference pairs for DPO (this episode vs a second rollout)
                if getattr(args, 'dpo', False) and sequences:
                    all_preference_pairs.append({
                        'sequences': sequences,
                        'reward': avg_reward,
                        'epoch': epoch,
                        'profile_idx': ep_idx,
                    })
                
                stability.log_step(avg_reward, float(loss.item()))
                
                episode_log = {
                    "epoch": epoch + 1, "episode": ep_idx + 1,
                    "final_emotion": final_emo, "avg_reward": avg_reward,
                    "num_turns": len(rewards),
                    "loss": float(loss.item()),
                }
                train_log["episodes"].append(episode_log)
                
                print(f"  [E{epoch+1}/{args.epochs}][{ep_idx+1}/{len(profiles)}] "
                      f"emo={final_emo:.0f} reward={avg_reward:.4f} "
                      f"turns={len(rewards)} loss={episode_log['loss']:.4f}")
            
            except Exception as e:
                print(f"  [E{epoch+1}/{args.epochs}][{ep_idx+1}/{len(profiles)}] ERROR: {e}")
                # CRITICAL: Free GPU memory to prevent cascading OOM
                torch.cuda.empty_cache()
                gc.collect()
                continue
        
        # Epoch summary
        if epoch_rewards:
            avg_r = np.mean(epoch_rewards)
            avg_s = np.mean(epoch_scores)
            train_log["epoch_stats"].append({
                "epoch": epoch + 1,
                "mean_reward": float(avg_r),
                "mean_score": float(avg_s),
                "std_reward": float(np.std(epoch_rewards)),
            })
            
            print(f"\n  === Epoch {epoch+1} Summary ===")
            print(f"  Mean Reward: {avg_r:.4f} | Mean Score: {avg_s:.1f}")
            
            if avg_r > best_avg_reward:
                best_avg_reward = avg_r
                save_path = os.path.join(args.output_dir, "best_model")
                os.makedirs(save_path, exist_ok=True)
                if HAS_PEFT:
                    model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                # Save base model path so eval script knows which base to use
                with open(os.path.join(save_path, "base_model_info.json"), 'w') as f:
                    json.dump(base_model_info, f)
                print(f"  * New best model saved! (reward={avg_r:.4f})")
        
        # Save checkpoint after each epoch for fault tolerance
        save_training_state(
            args.output_dir, epoch + 1, 0, model, optimizer,
            best_avg_reward, train_log, tokenizer
        )
    
    # ============================
    # DPO REFINEMENT STAGE (if enabled)
    # ============================
    if getattr(args, 'dpo', False) and ref_model is not None and len(all_preference_pairs) >= 2:
        print(f"\n{'='*60}")
        print(f"STAGE 2: DPO Refinement")
        print(f"Preference data: {len(all_preference_pairs)} episodes")
        print(f"{'='*60}\n")
        
        from train_grpo_dpo import _compute_seq_logprob
        
        dpo_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr * 0.5, weight_decay=0.01
        )
        
        # Sort episodes by reward, pair best with worst
        sorted_eps = sorted(all_preference_pairs, key=lambda x: x['reward'])
        n_pairs = len(sorted_eps) // 2
        dpo_pairs = []
        for i in range(n_pairs):
            worst = sorted_eps[i]
            best = sorted_eps[-(i+1)]
            if best['reward'] > worst['reward'] and best['sequences'] and worst['sequences']:
                dpo_pairs.append((best['sequences'], worst['sequences']))
        
        dpo_beta = getattr(args, 'dpo_beta', 0.1)
        for dpo_epoch in range(getattr(args, 'dpo_epochs', 2)):
            epoch_losses = []
            for winner_seqs, loser_seqs in dpo_pairs:
                try:
                    model.train()
                    w_input, w_gen = winner_seqs[0]
                    l_input, l_gen = loser_seqs[0]
                    
                    w_lp = _compute_seq_logprob(model, w_input, w_gen, device)
                    l_lp = _compute_seq_logprob(model, l_input, l_gen, device)
                    with torch.no_grad():
                        w_ref = _compute_seq_logprob(ref_model, w_input, w_gen, device)
                        l_ref = _compute_seq_logprob(ref_model, l_input, l_gen, device)
                    
                    dpo_loss = -F.logsigmoid(dpo_beta * ((w_lp - w_ref) - (l_lp - l_ref)))
                    
                    if dpo_loss.requires_grad:
                        dpo_optimizer.zero_grad()
                        dpo_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], 1.0
                        )
                        dpo_optimizer.step()
                        epoch_losses.append(float(dpo_loss.item()))
                        train_log["dpo_steps"].append({
                            "epoch": dpo_epoch + 1,
                            "loss": float(dpo_loss.item()),
                        })
                except Exception as e:
                    print(f"  [DPO] ERROR: {e}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
            
            if epoch_losses:
                avg_dpo = np.mean(epoch_losses)
                print(f"  [DPO Epoch {dpo_epoch+1}] avg_loss={avg_dpo:.4f} pairs={len(epoch_losses)}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "final_model")
    os.makedirs(final_path, exist_ok=True)
    if HAS_PEFT:
        model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    with open(os.path.join(final_path, "base_model_info.json"), 'w') as f:
        json.dump(base_model_info, f)
    
    # Save training log with stability metrics
    train_log["stability"] = stability.get_summary()
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(train_log, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Training complete! ({train_log['algorithm']})")
    print(f"Best model: {os.path.join(args.output_dir, 'best_model')}")
    print(f"Final model: {final_path}")
    print(f"Training log: {log_path}")
    print(f"{'='*60}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RLVER Improved Training")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sim-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./checkpoints/improved")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--episodes-per-epoch", type=int, default=20)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--curriculum", action="store_true", help="Enable curriculum learning")
    parser.add_argument("--thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--dpo", action="store_true", help="Enable DPO refinement after REINFORCE")
    parser.add_argument("--dpo-epochs", type=int, default=2, help="Number of DPO refinement epochs")
    parser.add_argument("--dpo-beta", type=float, default=0.1, help="DPO temperature parameter")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    args = parser.parse_args()
    
    if args.no_thinking:
        args.thinking = False
    
    os.makedirs(args.output_dir, exist_ok=True)
    train_with_reinforce(args)


if __name__ == "__main__":
    main()
