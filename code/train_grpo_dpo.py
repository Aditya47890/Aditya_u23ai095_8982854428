"""
RLVER GRPO + DPO Training Pipeline
====================================
Two-stage training:
  Stage 1 (GRPO): Group Relative Policy Optimization
    - For each profile, generate K responses
    - Score each with multi-dimensional reward
    - Compute group-relative advantages
    - Train using differentiable log-prob × advantage

  Stage 2 (DPO): Direct Preference Optimization
    - Use best/worst pairs from GRPO as preference data
    - Apply DPO loss for stable alignment

Designed for: 1× GPU with 32GB VRAM (LoRA + 4-bit quantization)

Usage:
    python train_grpo_dpo.py \
        --base-model ./models/qwen2.5-1.5b \
        --sim-model ./models/qwen2.5-7b-sim \
        --train-data ../data/train_profile.jsonl \
        --output-dir ./checkpoints/grpo_dpo \
        --epochs 3 --episodes-per-epoch 15 --curriculum
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
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig,
)

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    HAS_PEFT = True
except ImportError:
    HAS_PEFT = False
    print("WARNING: peft not installed. Run: pip install peft")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from improved_reward import compute_multi_dimensional_reward, compute_reward_with_breakdown
from train_improved import (
    THINK_SYSTEM, NOTHINK_SYSTEM,
    EmotionProfileDataset, LocalSimulator,
    generate_first_message,
)
from training_utils import (
    set_global_seed, save_training_state, load_training_state,
    StabilityLogger, DPOQualityFilter,
)


# ============================================================
# GRPO: Generate K responses and rank them
# ============================================================

def generate_k_responses(policy_model, policy_tokenizer, simulator, profile,
                         K=4, has_thinking=True, max_turns=6, device="cuda:0"):
    """Generate K complete dialogue trajectories for one profile.
    
    Returns:
        List of K tuples: (sequences, rewards, history, final_emo, total_reward)
    """
    system = THINK_SYSTEM if has_thinking else NOTHINK_SYSTEM
    trajectories = []
    
    for k in range(K):
        history = []
        generated_sequences = []
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
            
            with torch.no_grad():
                outputs = policy_model.generate(
                    **inputs,
                    max_new_tokens=200,
                    temperature=0.9,  # Slightly higher for diversity in group
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
        
        total_reward = float(np.mean(rewards_list)) if rewards_list else 0.0
        trajectories.append((generated_sequences, rewards_list, history, current_emotion, total_reward))
    
    return trajectories


def compute_grpo_loss(model, trajectories, device="cuda:0"):
    """Compute GRPO loss using group-relative advantages.
    
    GRPO Algorithm:
    1. For K trajectories, compute rewards r_1, ..., r_K
    2. Advantages: A_i = (r_i - mean(r)) / (std(r) + eps)
    3. Loss = -1/K * sum( A_i * log_prob(trajectory_i) )
    """
    rewards = [t[4] for t in trajectories]  # total_reward for each trajectory
    
    mean_r = np.mean(rewards)
    std_r = np.std(rewards) + 1e-8
    advantages = [(r - mean_r) / std_r for r in rewards]
    
    total_loss = torch.tensor(0.0, device=device)
    n_terms = 0
    
    for traj_idx, (sequences, _, _, _, _) in enumerate(trajectories):
        advantage = advantages[traj_idx]
        
        for input_ids, generated_ids in sequences:
            full_seq = torch.cat([input_ids[0], generated_ids], dim=0).unsqueeze(0).to(device)
            
            outputs = model(input_ids=full_seq, use_cache=False)
            logits = outputs.logits
            
            prompt_len = input_ids.shape[1]
            gen_len = len(generated_ids)
            
            gen_logits = logits[0, prompt_len - 1 : prompt_len + gen_len - 1, :]
            log_probs = F.log_softmax(gen_logits, dim=-1)
            selected = log_probs.gather(1, generated_ids.unsqueeze(1).to(device)).squeeze(1)
            
            log_prob_sum = selected.sum()
            total_loss = total_loss - advantage * log_prob_sum
            n_terms += 1
    
    if n_terms > 0:
        total_loss = total_loss / n_terms
    
    return total_loss


# ============================================================
# DPO: Direct Preference Optimization
# ============================================================

def collect_preference_pairs(trajectories):
    """Extract (best, worst) preference pairs from GRPO trajectories.
    
    Returns list of (profile_idx, best_sequences, worst_sequences) tuples.
    """
    rewards = [t[4] for t in trajectories]
    best_idx = int(np.argmax(rewards))
    worst_idx = int(np.argmin(rewards))
    
    if best_idx == worst_idx:
        return None  # No preference signal
    
    return (trajectories[best_idx][0], trajectories[worst_idx][0])  # (winner_seqs, loser_seqs)


def compute_dpo_loss(model, ref_model, preference_pairs, beta=0.1, device="cuda:0"):
    """Compute DPO loss on collected preference pairs.
    
    L_DPO = -log sigmoid(beta * (log pi/pi_ref(y_w|x) - log pi/pi_ref(y_l|x)))
    """
    total_loss = torch.tensor(0.0, device=device)
    n_pairs = 0
    
    for winner_seqs, loser_seqs in preference_pairs:
        # Use the first turn's sequences for simplicity (most impactful)
        if not winner_seqs or not loser_seqs:
            continue
        
        w_input, w_gen = winner_seqs[0]
        l_input, l_gen = loser_seqs[0]
        
        # Compute log probs under policy
        w_logprob = _compute_seq_logprob(model, w_input, w_gen, device)
        l_logprob = _compute_seq_logprob(model, l_input, l_gen, device)
        
        # Compute log probs under reference policy
        with torch.no_grad():
            w_ref_logprob = _compute_seq_logprob(ref_model, w_input, w_gen, device)
            l_ref_logprob = _compute_seq_logprob(ref_model, l_input, l_gen, device)
        
        # DPO loss
        w_ratio = w_logprob - w_ref_logprob
        l_ratio = l_logprob - l_ref_logprob
        
        loss = -F.logsigmoid(beta * (w_ratio - l_ratio))
        total_loss = total_loss + loss
        n_pairs += 1
    
    if n_pairs > 0:
        total_loss = total_loss / n_pairs
    
    return total_loss


def _compute_seq_logprob(model, input_ids, generated_ids, device):
    """Compute log probability of generated sequence under model."""
    full_seq = torch.cat([input_ids[0], generated_ids], dim=0).unsqueeze(0).to(device)
    
    outputs = model(input_ids=full_seq, use_cache=False)
    logits = outputs.logits
    
    prompt_len = input_ids.shape[1]
    gen_len = len(generated_ids)
    
    gen_logits = logits[0, prompt_len - 1 : prompt_len + gen_len - 1, :]
    log_probs = F.log_softmax(gen_logits, dim=-1)
    selected = log_probs.gather(1, generated_ids.unsqueeze(1).to(device)).squeeze(1)
    
    return selected.sum()


# ============================================================
# MAIN TRAINING LOOP: GRPO + DPO
# ============================================================

def train_grpo_dpo(args):
    """Two-stage training: GRPO then DPO."""
    device = "cuda:0"
    set_global_seed(getattr(args, 'seed', 42))
    
    # Load simulator
    simulator = LocalSimulator(args.sim_model, device=device)
    
    # Load policy model
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
    
    # Load reference model for DPO (frozen copy)
    print("Loading reference model for DPO...")
    ref_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb_config,
        device_map=device, trust_remote_code=True
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"Reference model loaded. Total VRAM: {torch.cuda.memory_allocated(0)/1e9:.2f} GB")
    
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=0.01
    )
    
    dataset = EmotionProfileDataset(args.train_data, curriculum=args.curriculum)
    base_model_info = {"base_model": args.base_model}
    
    # Initialize stability logger and DPO quality filter
    stability = StabilityLogger(window_size=5)
    dpo_filter = DPOQualityFilter(min_margin=getattr(args, 'dpo_margin', 0.05))
    
    train_log = {
        "config": vars(args),
        "algorithm": "GRPO+DPO",
        "grpo_episodes": [],
        "dpo_steps": [],
        "epoch_stats": [],
        "stability": {},
        "dpo_filter_stats": {},
    }
    
    # Try to resume from checkpoint
    start_epoch = 0
    best_avg_reward = -float('inf')
    all_preference_pairs = []
    
    resume_state = load_training_state(args.output_dir, model, optimizer)
    if resume_state is not None:
        start_epoch, _, best_avg_reward, saved_log = resume_state
        if saved_log is not None:
            train_log = saved_log
    
    print(f"\n{'='*60}")
    print(f"Starting GRPO + DPO Training")
    print(f"Base model: {args.base_model}")
    print(f"K (group size): {args.K}")
    print(f"DPO beta: {args.dpo_beta}")
    print(f"DPO margin filter: {dpo_filter.min_margin}")
    print(f"Epochs: {args.epochs}, Episodes/epoch: {args.episodes_per_epoch}")
    if start_epoch > 0:
        print(f"Resuming from epoch {start_epoch + 1}")
    print(f"{'='*60}\n")
    
    # ============================
    # STAGE 1: GRPO Training
    # ============================
    print("\n" + "="*60)
    print("STAGE 1: GRPO (Group Relative Policy Optimization)")
    print("="*60 + "\n")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_rewards = []
        epoch_scores = []
        
        profiles = dataset.get_curriculum_batch(
            epoch, args.epochs, args.episodes_per_epoch
        )
        
        for ep_idx, profile in enumerate(profiles):
            model.train()
            
            try:
                # Generate K trajectories
                trajectories = generate_k_responses(
                    model, tokenizer, simulator, profile,
                    K=args.K, has_thinking=args.thinking,
                    max_turns=args.max_turns, device=device
                )
                
                valid_trajs = [t for t in trajectories if t[0]]  # has sequences
                if len(valid_trajs) < 2:
                    print(f"    (skipped - need >=2 valid trajectories)")
                    continue
                
                # Compute GRPO loss
                grpo_loss = compute_grpo_loss(model, valid_trajs, device=device)
                
                if not grpo_loss.requires_grad:
                    continue
                
                optimizer.zero_grad()
                grpo_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0
                )
                optimizer.step()
                
                # Collect preference pairs for DPO (with quality filter)
                pair = collect_preference_pairs(valid_trajs)
                if pair is not None:
                    pair_rewards = [t[4] for t in valid_trajs]
                    r_chosen = max(pair_rewards)
                    r_rejected = min(pair_rewards)
                    if dpo_filter.should_keep(r_chosen, r_rejected):
                        all_preference_pairs.append(pair)
                
                rewards = [t[4] for t in valid_trajs]
                best_reward = max(rewards)
                mean_reward = float(np.mean(rewards))
                best_emo = valid_trajs[int(np.argmax(rewards))][3]
                
                epoch_rewards.append(mean_reward)
                epoch_scores.append(best_emo)
                
                # Track stability
                stability.log_step(mean_reward, float(grpo_loss.item()))
                
                train_log["grpo_episodes"].append({
                    "epoch": epoch + 1, "episode": ep_idx + 1,
                    "best_reward": float(best_reward),
                    "mean_reward": mean_reward,
                    "best_emotion": float(best_emo),
                    "k_valid": len(valid_trajs),
                    "loss": float(grpo_loss.item()),
                })
                
                print(f"  [GRPO E{epoch+1}][{ep_idx+1}/{len(profiles)}] "
                      f"best_emo={best_emo:.0f} mean_r={mean_reward:.4f} "
                      f"best_r={best_reward:.4f} K={len(valid_trajs)} "
                      f"loss={grpo_loss.item():.4f}")
            
            except Exception as e:
                print(f"  [GRPO E{epoch+1}][{ep_idx+1}/{len(profiles)}] ERROR: {e}")
                torch.cuda.empty_cache()
                gc.collect()
                continue
        
        # Epoch summary
        if epoch_rewards:
            avg_r = np.mean(epoch_rewards)
            avg_s = np.mean(epoch_scores)
            train_log["epoch_stats"].append({
                "epoch": epoch + 1, "stage": "GRPO",
                "mean_reward": float(avg_r),
                "mean_score": float(avg_s),
            })
            print(f"\n  === GRPO Epoch {epoch+1} ===")
            print(f"  Mean Reward: {avg_r:.4f} | Mean Score: {avg_s:.1f}")
            print(f"  Preference pairs collected: {len(all_preference_pairs)}")
            dpo_filter.log_status()
            
            if avg_r > best_avg_reward:
                best_avg_reward = avg_r
                save_path = os.path.join(args.output_dir, "grpo_model")
                os.makedirs(save_path, exist_ok=True)
                if HAS_PEFT:
                    model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                with open(os.path.join(save_path, "base_model_info.json"), 'w') as f:
                    json.dump(base_model_info, f)
                print(f"  * GRPO model saved! (reward={avg_r:.4f})")
        
        # Save checkpoint after each epoch for fault tolerance
        save_training_state(
            args.output_dir, epoch + 1, 0, model, optimizer,
            best_avg_reward, train_log, tokenizer
        )
    
    # ============================
    # STAGE 2: DPO Refinement
    # ============================
    if all_preference_pairs and args.dpo_epochs > 0:
        print(f"\n{'='*60}")
        print(f"STAGE 2: DPO (Direct Preference Optimization)")
        print(f"Preference pairs: {len(all_preference_pairs)}")
        print(f"DPO epochs: {args.dpo_epochs}, beta: {args.dpo_beta}")
        print(f"{'='*60}\n")
        
        dpo_optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=args.lr * 0.5, weight_decay=0.01  # Lower LR for DPO
        )
        
        for dpo_epoch in range(args.dpo_epochs):
            random.shuffle(all_preference_pairs)
            
            # Mini-batch DPO
            batch_size = min(4, len(all_preference_pairs))
            epoch_dpo_losses = []
            
            for batch_start in range(0, len(all_preference_pairs), batch_size):
                batch = all_preference_pairs[batch_start:batch_start + batch_size]
                
                try:
                    model.train()
                    dpo_loss = compute_dpo_loss(
                        model, ref_model, batch,
                        beta=args.dpo_beta, device=device
                    )
                    
                    if dpo_loss.requires_grad:
                        dpo_optimizer.zero_grad()
                        dpo_loss.backward()
                        torch.nn.utils.clip_grad_norm_(
                            [p for p in model.parameters() if p.requires_grad], 1.0
                        )
                        dpo_optimizer.step()
                        
                        epoch_dpo_losses.append(float(dpo_loss.item()))
                        
                        train_log["dpo_steps"].append({
                            "dpo_epoch": dpo_epoch + 1,
                            "batch": batch_start // batch_size + 1,
                            "loss": float(dpo_loss.item()),
                        })
                
                except Exception as e:
                    print(f"  [DPO] ERROR: {e}")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
            
            if epoch_dpo_losses:
                avg_dpo = np.mean(epoch_dpo_losses)
                print(f"  [DPO Epoch {dpo_epoch+1}/{args.dpo_epochs}] "
                      f"avg_loss={avg_dpo:.4f} steps={len(epoch_dpo_losses)}")
                
                train_log["epoch_stats"].append({
                    "epoch": dpo_epoch + 1, "stage": "DPO",
                    "mean_dpo_loss": float(avg_dpo),
                })
    else:
        print("\nSkipping DPO (no preference pairs collected or dpo_epochs=0)")
    
    # Save final model
    final_path = os.path.join(args.output_dir, "best_model")
    os.makedirs(final_path, exist_ok=True)
    if HAS_PEFT:
        model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    with open(os.path.join(final_path, "base_model_info.json"), 'w') as f:
        json.dump(base_model_info, f)
    
    # Save training log with stability and filter data
    train_log["stability"] = stability.get_summary()
    train_log["dpo_filter_stats"] = dpo_filter.get_stats()
    
    log_path = os.path.join(args.output_dir, "training_log.json")
    with open(log_path, 'w') as f:
        json.dump(train_log, f, indent=2)
    
    # Save preference pairs metadata
    pref_path = os.path.join(args.output_dir, "preference_pairs_count.json")
    with open(pref_path, 'w') as f:
        json.dump(dpo_filter.get_stats(), f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"GRPO+DPO Training Complete!")
    print(f"Best model: {final_path}")
    print(f"GRPO model: {os.path.join(args.output_dir, 'grpo_model')}")
    print(f"Preference pairs: {len(all_preference_pairs)}")
    print(f"{'='*60}")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RLVER GRPO+DPO Training")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--sim-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./checkpoints/grpo_dpo")
    parser.add_argument("--epochs", type=int, default=3, help="GRPO epochs")
    parser.add_argument("--dpo-epochs", type=int, default=2, help="DPO refinement epochs")
    parser.add_argument("--episodes-per-epoch", type=int, default=15)
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--K", type=int, default=4, help="Group size for GRPO")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--dpo-beta", type=float, default=0.1, help="DPO temperature")
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--thinking", action="store_true", default=True)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=42, help="Global random seed")
    parser.add_argument("--dpo-margin", type=float, default=0.05, help="Min reward margin for DPO pairs")
    args = parser.parse_args()
    
    if args.no_thinking:
        args.thinking = False
    
    os.makedirs(args.output_dir, exist_ok=True)
    train_grpo_dpo(args)


if __name__ == "__main__":
    main()
