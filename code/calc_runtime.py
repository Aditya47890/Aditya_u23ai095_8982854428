"""Precise HPC runtime calculator based on local test measurements."""

print("=" * 70)
print("PRECISE HPC RUNTIME ESTIMATE")
print("=" * 70)

# ---- MEASURED LOCAL BASELINE (RTX 4050 6GB, 1.5B + 1.5B) ----
local_train_min_per_episode = 5.0   # 2 episodes in ~10 min
local_eval_min_per_dialogue = 4.5   # 6 dialogues in ~27 min

# ---- HPC vs LOCAL speed factors ----
gpu_speedup = 2.5  # RTX PRO 4500 Blackwell 32GB vs RTX 4050 Laptop 6GB
sim_slowdown = 4.67  # 7B vs 1.5B parameters

# Training: 60% policy (same size), 40% simulator (bigger)
train_factor = 0.6 / gpu_speedup + 0.4 * sim_slowdown / gpu_speedup
hpc_train = local_train_min_per_episode * train_factor

# Evaluation: 50% policy, 50% simulator
eval_factor = 0.5 / gpu_speedup + 0.5 * sim_slowdown / gpu_speedup
hpc_eval = local_eval_min_per_dialogue * eval_factor

print(f"\nPer-unit times on HPC:")
print(f"  Training: {hpc_train:.1f} min/episode")
print(f"  Evaluation: {hpc_eval:.1f} min/dialogue")

# ---- PHASE-BY-PHASE (from setup_and_run.sh) ----
print("\n" + "=" * 70)
print("PHASE-BY-PHASE BREAKDOWN")
print("=" * 70)

# Phase 1+2: Setup + Download
setup = 30 + 6   # 30 min setup, ~6 min download on fast HPC network
print(f"\n[Phase 1+2] Setup + Download: ~{setup} min")

# Phase 3: Evaluate baselines
# 3 models x 2 modes (--both) x 6 scenarios x 5 dialogues = 180 dialogues
baseline_dialogues = 3 * 2 * 6 * 5  # = 180
baseline_min = baseline_dialogues * hpc_eval
print(f"[Phase 3] Evaluate baselines:")
print(f"  3 models x 2 modes x 6 scenarios x 5 dialogues = {baseline_dialogues}")
print(f"  {baseline_dialogues} x {hpc_eval:.1f} min = {baseline_min:.0f} min ({baseline_min/60:.1f} hrs)")

# Phase 4: Train
# 2 runs (no-curriculum + curriculum) x 3 epochs x 30 episodes = 180 episodes
train_episodes = 2 * 3 * 30  # = 180
train_min = train_episodes * hpc_train
print(f"[Phase 4] Training:")
print(f"  2 runs x 3 epochs x 30 episodes = {train_episodes}")
print(f"  {train_episodes} x {hpc_train:.1f} min = {train_min:.0f} min ({train_min/60:.1f} hrs)")

# Phase 5: Evaluate improved
# 2 models x 2 modes x 6 scenarios x 5 dialogues = 120 dialogues
improved_dialogues = 2 * 2 * 6 * 5  # = 120
improved_min = improved_dialogues * hpc_eval
print(f"[Phase 5] Evaluate improved:")
print(f"  2 models x 2 modes x 6 scenarios x 5 dialogues = {improved_dialogues}")
print(f"  {improved_dialogues} x {hpc_eval:.1f} min = {improved_min:.0f} min ({improved_min/60:.1f} hrs)")

# Phase 6
print(f"[Phase 6] Analysis: ~1 min")

total = setup + baseline_min + train_min + improved_min + 1
print(f"\n{'=' * 70}")
print(f"TOTAL: {total:.0f} min = {total/60:.1f} hours")
print(f"{'=' * 70}")
print(f"\n  Setup + Download:  {setup} min")
print(f"  Baseline eval:     {baseline_min:.0f} min ({baseline_min/60:.1f} hrs)")
print(f"  Training:          {train_min:.0f} min ({train_min/60:.1f} hrs)")
print(f"  Improved eval:     {improved_min:.0f} min ({improved_min/60:.1f} hrs)")
print(f"  Analysis:          1 min")
print(f"\n  >>> START IN THE EVENING, RESULTS BY NEXT MORNING <<<")
