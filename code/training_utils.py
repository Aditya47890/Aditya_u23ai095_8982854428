"""
RLVER Training Utilities
========================
Shared utilities for all training scripts:
- Global seed locking (reproducibility)
- Stateful checkpoint save/resume
- Training stability logger
- DPO quality filter
"""

import os
import gc
import json
import random
import time
import numpy as np
from datetime import datetime

import torch
import torch.nn.functional as F


# ============================================================
# 1. REPRODUCIBILITY: Global Seed
# ============================================================

def set_global_seed(seed=42):
    """Lock all random number generators for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"  [SEED] Global seed set to {seed}")


# ============================================================
# 2. STATEFUL CHECKPOINT: Save & Resume
# ============================================================

def save_training_state(output_dir, epoch, episode, model, optimizer,
                        best_reward, train_log, tokenizer=None,
                        extra_state=None):
    """Save full training state for fault-tolerant resumption.
    
    Saves:
        - LoRA adapter weights (model)
        - Optimizer state dict
        - Training progress (epoch, episode, best_reward)
        - Training log
        - Any extra state (e.g., preference pairs count)
    """
    state_dir = os.path.join(output_dir, "_resume_state")
    os.makedirs(state_dir, exist_ok=True)
    
    state = {
        "epoch": epoch,
        "episode": episode,
        "best_reward": best_reward,
        "timestamp": datetime.now().isoformat(),
    }
    if extra_state:
        state.update(extra_state)
    
    # Save state metadata
    with open(os.path.join(state_dir, "state.json"), 'w') as f:
        json.dump(state, f, indent=2)
    
    # Save training log
    with open(os.path.join(state_dir, "training_log.json"), 'w') as f:
        json.dump(train_log, f, indent=2)
    
    # Save optimizer state
    torch.save(optimizer.state_dict(), os.path.join(state_dir, "optimizer.pt"))
    
    # Save model (LoRA weights)
    try:
        model.save_pretrained(os.path.join(state_dir, "model"))
        if tokenizer is not None:
            tokenizer.save_pretrained(os.path.join(state_dir, "model"))
    except Exception as e:
        print(f"  [CHECKPOINT] Warning: Could not save model: {e}")
    
    print(f"  [CHECKPOINT] State saved at epoch={epoch}, episode={episode}")


def load_training_state(output_dir, model, optimizer):
    """Load saved training state if it exists.
    
    Returns:
        (start_epoch, start_episode, best_reward, train_log) or None if no state.
    """
    state_dir = os.path.join(output_dir, "_resume_state")
    state_file = os.path.join(state_dir, "state.json")
    
    if not os.path.exists(state_file):
        return None
    
    try:
        # Load state metadata
        with open(state_file, 'r') as f:
            state = json.load(f)
        
        # Load training log
        log_file = os.path.join(state_dir, "training_log.json")
        if os.path.exists(log_file):
            with open(log_file, 'r') as f:
                train_log = json.load(f)
        else:
            train_log = None
        
        # Load optimizer state
        opt_file = os.path.join(state_dir, "optimizer.pt")
        if os.path.exists(opt_file):
            try:
                opt_state = torch.load(opt_file, map_location="cpu", weights_only=True)
                optimizer.load_state_dict(opt_state)
            except Exception:
                print("  [RESUME] Warning: Could not load optimizer state, using fresh optimizer")
        
        # Load model weights
        model_dir = os.path.join(state_dir, "model")
        if os.path.exists(os.path.join(model_dir, "adapter_config.json")):
            try:
                from peft import PeftModel
                # The model already has LoRA, just load the weights
                model.load_adapter(model_dir, adapter_name="default")
                print(f"  [RESUME] Loaded LoRA weights from checkpoint")
            except Exception as e:
                print(f"  [RESUME] Warning: Could not load LoRA weights: {e}")
        
        epoch = state.get("epoch", 0)
        episode = state.get("episode", 0)
        best_reward = state.get("best_reward", -float('inf'))
        
        print(f"  [RESUME] Resuming from epoch={epoch}, episode={episode}, "
              f"best_reward={best_reward:.4f}")
        print(f"  [RESUME] Saved at: {state.get('timestamp', 'unknown')}")
        
        return epoch, episode, best_reward, train_log
    
    except Exception as e:
        print(f"  [RESUME] Could not load state: {e}. Starting fresh.")
        return None


# ============================================================
# 3. TRAINING STABILITY LOGGER
# ============================================================

class StabilityLogger:
    """Tracks training metrics for stability monitoring.
    
    Detects:
    - Reward collapse (mean reward drops below threshold)
    - Loss explosion (loss exceeds threshold)
    - Gradient issues (NaN/Inf)
    """
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.reward_history = []
        self.loss_history = []
        self.grad_norms = []
        self.warnings = []
    
    def log_step(self, reward, loss, grad_norm=None):
        """Log one training step."""
        self.reward_history.append(float(reward))
        self.loss_history.append(float(loss))
        if grad_norm is not None:
            self.grad_norms.append(float(grad_norm))
        
        # Check for instability
        self._check_stability()
    
    def _check_stability(self):
        """Check for training instability signals."""
        if len(self.reward_history) < self.window_size:
            return
        
        recent = self.reward_history[-self.window_size:]
        
        # Check reward collapse
        if np.mean(recent) < 0.05:
            msg = f"WARNING: Possible reward collapse! Mean reward={np.mean(recent):.4f}"
            self.warnings.append(msg)
            print(f"  [STABILITY] {msg}")
        
        # Check reward variance spike
        if len(self.reward_history) >= 2 * self.window_size:
            prev = self.reward_history[-2*self.window_size:-self.window_size]
            if np.std(recent) > 3 * np.std(prev) and np.std(prev) > 0:
                msg = f"WARNING: Reward variance spike! std={np.std(recent):.4f} vs prev={np.std(prev):.4f}"
                self.warnings.append(msg)
                print(f"  [STABILITY] {msg}")
        
        # Check loss explosion
        recent_loss = self.loss_history[-self.window_size:]
        if any(abs(l) > 1000 for l in recent_loss):
            msg = f"WARNING: Loss explosion! max={max(abs(l) for l in recent_loss):.1f}"
            self.warnings.append(msg)
            print(f"  [STABILITY] {msg}")
    
    def get_summary(self):
        """Return training stability summary."""
        return {
            "total_steps": len(self.reward_history),
            "final_mean_reward": float(np.mean(self.reward_history[-10:])) if self.reward_history else 0,
            "final_std_reward": float(np.std(self.reward_history[-10:])) if self.reward_history else 0,
            "reward_trend": self._compute_trend(self.reward_history),
            "loss_trend": self._compute_trend(self.loss_history),
            "warnings": self.warnings,
            "reward_curve": self.reward_history,
            "loss_curve": self.loss_history,
        }
    
    def _compute_trend(self, values):
        """Compute linear trend (slope) of values."""
        if len(values) < 2:
            return 0.0
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        return float(slope)


# ============================================================
# 4. DPO QUALITY FILTER
# ============================================================

class DPOQualityFilter:
    """Filters preference pairs for DPO training quality.
    
    Only keeps pairs where the reward margin exceeds a threshold,
    ensuring DPO trains on meaningful preference signals.
    """
    
    def __init__(self, min_margin=0.05):
        self.min_margin = min_margin
        self.total_proposed = 0
        self.total_accepted = 0
        self.total_rejected = 0
        self.margins = []
    
    def should_keep(self, reward_chosen, reward_rejected):
        """Check if a preference pair meets quality threshold."""
        self.total_proposed += 1
        margin = reward_chosen - reward_rejected
        self.margins.append(float(margin))
        
        if margin > self.min_margin:
            self.total_accepted += 1
            return True
        else:
            self.total_rejected += 1
            return False
    
    def get_stats(self):
        """Return quality filter statistics."""
        acceptance_rate = self.total_accepted / max(1, self.total_proposed)
        return {
            "total_proposed": self.total_proposed,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "acceptance_rate": float(acceptance_rate),
            "mean_margin": float(np.mean(self.margins)) if self.margins else 0,
            "min_margin_threshold": self.min_margin,
        }
    
    def log_status(self):
        """Print current filter status."""
        stats = self.get_stats()
        print(f"  [DPO Filter] Accepted: {stats['total_accepted']}/{stats['total_proposed']} "
              f"({stats['acceptance_rate']*100:.0f}%) | "
              f"Mean margin: {stats['mean_margin']:.4f}")


# ============================================================
# 5. REWARD DIAGNOSTICS
# ============================================================

def compute_length_normalized_reward(response_text, raw_reward):
    """Compute length-normalized reward to detect reward hacking.
    
    If model generates longer text just to inflate scores,
    the length-normalized reward will be lower.
    """
    word_count = len(response_text.split())
    if word_count == 0:
        return 0.0
    return raw_reward / np.log(word_count + 1)  # Log-normalize


GENERIC_EMPATHY_PHRASES = [
    "i understand", "i hear you", "that must be", "i'm sorry to hear",
    "it's okay", "it sounds like", "i can imagine", "you're not alone",
    "take your time", "i appreciate you sharing", "that's valid",
    "your feelings are valid", "i'm here for you",
    "我理解", "我听到了", "你不是一个人", "没关系", "我能感受到",
    "你的感受是", "慢慢来", "我在这里",
]


def detect_generic_responses(response_text):
    """Count how many generic empathy templates appear in a response.
    
    Returns:
        (template_count, unique_phrases_used)
    """
    text_lower = response_text.lower()
    found = [p for p in GENERIC_EMPATHY_PHRASES if p in text_lower]
    return len(found), found


# ============================================================
# 6. STATISTICAL VALIDATION
# ============================================================

def compute_confidence_interval(data, confidence=0.95):
    """Compute confidence interval for a list of values.
    
    Returns (mean, ci_lower, ci_upper, std, n)
    """
    if data is None or (hasattr(data, '__len__') and len(data) == 0):
        return 0, 0, 0, 0, 0
    
    data = np.array(data, dtype=float)
    n = len(data)
    mean = np.mean(data)
    std = np.std(data, ddof=1) if n > 1 else 0
    
    # t-distribution critical value approximation for 95% CI
    if n <= 1:
        return float(mean), float(mean), float(mean), 0, n
    
    from scipy import stats as scipy_stats
    t_crit = scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1)
    margin = t_crit * std / np.sqrt(n)
    
    return float(mean), float(mean - margin), float(mean + margin), float(std), n


def compute_win_rate(scores_a, scores_b):
    """Compute win rate of model A vs model B.
    
    Returns (win_rate_a, win_rate_b, tie_rate)
    A win is when score_a > score_b for paired observations.
    """
    n = min(len(scores_a), len(scores_b))
    if n == 0:
        return 0.5, 0.5, 0.0
    
    wins_a = sum(1 for i in range(n) if scores_a[i] > scores_b[i])
    wins_b = sum(1 for i in range(n) if scores_b[i] > scores_a[i])
    ties = n - wins_a - wins_b
    
    return wins_a / n, wins_b / n, ties / n
