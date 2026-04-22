#!/usr/bin/env bash
#
# RLVER Full Pipeline: Setup, Train, Evaluate, Ablation Study
# ============================================================
# Runs on HPC with single GPU (32GB VRAM)
#
# Usage:
#   ./setup_and_run.sh setup       # Install env + deps
#   ./setup_and_run.sh download    # Download models
#   ./setup_and_run.sh evaluate    # Evaluate baselines
#   ./setup_and_run.sh train       # Train REINFORCE models
#   ./setup_and_run.sh train_grpo  # Train GRPO+DPO models
#   ./setup_and_run.sh eval_all    # Evaluate all trained models
#   ./setup_and_run.sh analyze     # Generate figures + tables
#   ./setup_and_run.sh full        # Run everything
# ============================================================

# NOTE: We intentionally do NOT use 'set -e' here.
# A 20+ hour pipeline must survive non-fatal errors (e.g., a single
# training episode OOM) without killing the entire run.

# ===== CONFIGURATION =====
PROJECT_DIR="$HOME/RLVER_PROJECT"
CONDA_ENV="rlver"
BASE_MODEL_ID="Qwen/Qwen2.5-1.5B-Instruct"
SIM_MODEL_ID="Qwen/Qwen2.5-7B-Instruct"

# Paths relative to project root
# After git clone, repo will be at: $PROJECT_DIR/Aditya_u23ai095_8982854428/
REPO_DIR="$PROJECT_DIR/Aditya_u23ai095_8982854428"
CODE_DIR="$REPO_DIR/code"
DATA_DIR="$REPO_DIR/data"
MODEL_DIR="$PROJECT_DIR/models"
RESULTS_DIR="$PROJECT_DIR/results"
FIGURES_DIR="$PROJECT_DIR/figures"
CHECKPOINT_DIR="$PROJECT_DIR/checkpoints"

# Optimized settings (reduced runtime)
N_DIALOGUES=3          # Was 5, reduced for speed
EPISODES_PER_EPOCH=15  # Was 30, reduced for speed
N_EPOCHS=3
MAX_TURNS=6            # Was 8, reduced for speed (25% fewer inference calls per dialogue)
GRPO_K=2               # Group size for GRPO (K=2 proven to work, saves 50% gen time)
DPO_BETA=0.1
DPO_EPOCHS=2
TRAIN_DATA="$DATA_DIR/train_250.jsonl"  # 250 profiles (reduced from 500 for speed)


# =============================================================
# Phase 1: Environment Setup
# =============================================================
setup_env() {
    echo "============================================"
    echo "Phase 1: Setting up environment"
    echo "============================================"
    
    eval "$(conda shell.bash hook)"
    
    if conda env list | grep -q "$CONDA_ENV"; then
        echo "Environment '$CONDA_ENV' exists. Activating..."
        conda activate "$CONDA_ENV"
    else
        echo "Creating conda environment: $CONDA_ENV"
        conda create -n "$CONDA_ENV" python=3.10 -y
        conda activate "$CONDA_ENV"
    fi
    
    echo "Installing dependencies..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
    pip install transformers==4.46.0 accelerate bitsandbytes \
        peft trl datasets scipy numpy pandas matplotlib seaborn \
        huggingface_hub
    
    mkdir -p "$MODEL_DIR" "$RESULTS_DIR" "$FIGURES_DIR" "$CHECKPOINT_DIR"
    
    # Prepare reduced dataset (250 profiles for training)
    if [ ! -f "$TRAIN_DATA" ]; then
        echo "Creating reduced training dataset (250 profiles)..."
        head -250 "$DATA_DIR/train_profile.jsonl" > "$TRAIN_DATA"
        echo "  Created $TRAIN_DATA with $(wc -l < "$TRAIN_DATA") profiles"
    fi
    
    echo ""
    echo "Environment ready!"
    python -c "import torch; print(f'  GPU: {torch.cuda.get_device_name(0)}'); print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')"
    echo ""
}


# =============================================================
# Phase 2: Download Models
# =============================================================
download_models() {
    echo "============================================"
    echo "Phase 2: Downloading models"
    echo "============================================"
    
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    
    # Download base policy model
    echo "--- Downloading Qwen2.5-1.5B-Instruct ---"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$BASE_MODEL_ID', local_dir='$MODEL_DIR/qwen2.5-1.5b')
print('Base model downloaded!')
"
    
    # Download simulator model
    echo "--- Downloading Qwen2.5-7B-Instruct ---"
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$SIM_MODEL_ID', local_dir='$MODEL_DIR/qwen2.5-7b-sim')
print('Simulator model downloaded!')
"
    
    echo ""
    echo "All models downloaded!"
    echo ""
}


# =============================================================
# Phase 3: Evaluate Baseline Models
# =============================================================
evaluate_baselines() {
    echo "============================================"
    echo "Phase 3: Evaluating baseline models"
    echo "============================================"
    
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    cd "$CODE_DIR"
    
    # 3a. Evaluate Qwen2.5-1.5B base (no RL training)
    echo "--- Evaluating Qwen2.5-1.5B Base ---"
    python evaluate_adversarial.py \
        --policy-model "$MODEL_DIR/qwen2.5-1.5b" \
        --sim-model "$MODEL_DIR/qwen2.5-7b-sim" \
        --model-name "Baseline" \
        --n-dialogues $N_DIALOGUES \
        --max-turns $MAX_TURNS \
        --both \
        --output "$RESULTS_DIR"
    
    echo ""
    echo "Baseline evaluation complete!"
    echo ""
}


# =============================================================
# Phase 4a: Train REINFORCE Models (with/without curriculum)
# =============================================================
train_reinforce() {
    echo "============================================"
    echo "Phase 4a: Training REINFORCE models"
    echo "============================================"
    
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    cd "$CODE_DIR"
    
    # Without curriculum
    echo "--- Training: REINFORCE (no curriculum) ---"
    python train_improved.py \
        --base-model "$MODEL_DIR/qwen2.5-1.5b" \
        --sim-model "$MODEL_DIR/qwen2.5-7b-sim" \
        --train-data "$TRAIN_DATA" \
        --output-dir "$CHECKPOINT_DIR/reinforce_no_curriculum" \
        --epochs $N_EPOCHS \
        --episodes-per-epoch $EPISODES_PER_EPOCH \
        --lr 2e-5 \
        --seed 42 \
        --thinking
    
    # With curriculum
    echo "--- Training: REINFORCE + Curriculum ---"
    python train_improved.py \
        --base-model "$MODEL_DIR/qwen2.5-1.5b" \
        --sim-model "$MODEL_DIR/qwen2.5-7b-sim" \
        --train-data "$TRAIN_DATA" \
        --output-dir "$CHECKPOINT_DIR/reinforce_curriculum" \
        --epochs $N_EPOCHS \
        --episodes-per-epoch $EPISODES_PER_EPOCH \
        --lr 2e-5 \
        --curriculum \
        --seed 42 \
        --thinking
    
    echo ""
    echo "REINFORCE training complete!"
    echo ""
}


# =============================================================
# Phase 4b: Train GRPO + DPO Models
# =============================================================
train_grpo() {
    echo "============================================"
    echo "Phase 4b: Training GRPO+DPO models"
    echo "============================================"
    
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    cd "$CODE_DIR"
    
    # GRPO+DPO without curriculum
    echo "--- Training: GRPO+DPO (no curriculum) ---"
    python train_grpo_dpo.py \
        --base-model "$MODEL_DIR/qwen2.5-1.5b" \
        --sim-model "$MODEL_DIR/qwen2.5-7b-sim" \
        --train-data "$TRAIN_DATA" \
        --output-dir "$CHECKPOINT_DIR/grpo_dpo_no_curriculum" \
        --epochs $N_EPOCHS \
        --dpo-epochs $DPO_EPOCHS \
        --episodes-per-epoch $EPISODES_PER_EPOCH \
        --K $GRPO_K \
        --dpo-beta $DPO_BETA \
        --lr 2e-5 \
        --seed 42 \
        --dpo-margin 0.05 \
        --thinking
    
    # GRPO+DPO with curriculum
    echo "--- Training: GRPO+DPO + Curriculum ---"
    python train_grpo_dpo.py \
        --base-model "$MODEL_DIR/qwen2.5-1.5b" \
        --sim-model "$MODEL_DIR/qwen2.5-7b-sim" \
        --train-data "$TRAIN_DATA" \
        --output-dir "$CHECKPOINT_DIR/grpo_dpo_curriculum" \
        --epochs $N_EPOCHS \
        --dpo-epochs $DPO_EPOCHS \
        --episodes-per-epoch $EPISODES_PER_EPOCH \
        --K $GRPO_K \
        --dpo-beta $DPO_BETA \
        --lr 2e-5 \
        --curriculum \
        --seed 42 \
        --dpo-margin 0.05 \
        --thinking
    
    echo ""
    echo "GRPO+DPO training complete!"
    echo ""
}


# =============================================================
# Phase 5: Evaluate ALL Trained Models
# =============================================================
evaluate_all() {
    echo "============================================"
    echo "Phase 5: Evaluating all trained models"
    echo "============================================"
    
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    cd "$CODE_DIR"
    
    # REINFORCE (no curriculum)
    echo "--- Evaluating: REINFORCE ---"
    python evaluate_adversarial.py \
        --policy-model "$CHECKPOINT_DIR/reinforce_no_curriculum/best_model" \
        --sim-model "$MODEL_DIR/qwen2.5-7b-sim" \
        --model-name "REINFORCE" \
        --n-dialogues $N_DIALOGUES \
        --max-turns $MAX_TURNS \
        --both \
        --output "$RESULTS_DIR"
    
    # REINFORCE + Curriculum
    echo "--- Evaluating: REINFORCE+Curriculum ---"
    python evaluate_adversarial.py \
        --policy-model "$CHECKPOINT_DIR/reinforce_curriculum/best_model" \
        --sim-model "$MODEL_DIR/qwen2.5-7b-sim" \
        --model-name "REINFORCE-Curriculum" \
        --n-dialogues $N_DIALOGUES \
        --max-turns $MAX_TURNS \
        --both \
        --output "$RESULTS_DIR"
    
    # GRPO+DPO (no curriculum)
    echo "--- Evaluating: GRPO+DPO ---"
    python evaluate_adversarial.py \
        --policy-model "$CHECKPOINT_DIR/grpo_dpo_no_curriculum/best_model" \
        --sim-model "$MODEL_DIR/qwen2.5-7b-sim" \
        --model-name "GRPO-DPO" \
        --n-dialogues $N_DIALOGUES \
        --max-turns $MAX_TURNS \
        --both \
        --output "$RESULTS_DIR"
    
    # GRPO+DPO + Curriculum (our best model)
    echo "--- Evaluating: GRPO+DPO+Curriculum ---"
    python evaluate_adversarial.py \
        --policy-model "$CHECKPOINT_DIR/grpo_dpo_curriculum/best_model" \
        --sim-model "$MODEL_DIR/qwen2.5-7b-sim" \
        --model-name "GRPO-DPO-Curriculum" \
        --n-dialogues $N_DIALOGUES \
        --max-turns $MAX_TURNS \
        --both \
        --output "$RESULTS_DIR"
    
    echo ""
    echo "All evaluations complete!"
    echo ""
}


# =============================================================
# Phase 6: Generate Figures & Tables
# =============================================================
analyze() {
    echo "============================================"
    echo "Phase 6: Generating analysis"
    echo "============================================"
    
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    cd "$CODE_DIR"
    
    python analyze_results.py \
        --results-dir "$RESULTS_DIR" \
        --output-dir "$FIGURES_DIR"
    
    echo ""
    echo "============================================================"
    echo "ALL PHASES COMPLETE!"
    echo "============================================================"
    echo ""
    echo "Results:  $RESULTS_DIR/"
    echo "Figures:  $FIGURES_DIR/"
    echo "Models:   $CHECKPOINT_DIR/"
    echo ""
    echo "Copy results to your laptop with:"
    echo "  scp -r RL@172.21.1.220:~/RLVER_PROJECT/figures/ ."
    echo "  scp -r RL@172.21.1.220:~/RLVER_PROJECT/results/ ."
    echo ""
}


# =============================================================
# FULL PIPELINE
# =============================================================
full() {
    setup_env
    download_models
    evaluate_baselines
    train_reinforce
    train_grpo
    evaluate_all
    analyze
}


# =============================================================
# COMMAND DISPATCHER
# =============================================================
case "$1" in
    setup)     setup_env ;;
    download)  download_models ;;
    evaluate)  evaluate_baselines ;;
    train)     train_reinforce ;;
    train_grpo) train_grpo ;;
    eval_all)  evaluate_all ;;
    analyze)   analyze ;;
    full)      full ;;
    *)
        echo "RLVER Pipeline"
        echo ""
        echo "Usage: $0 {setup|download|evaluate|train|train_grpo|eval_all|analyze|full}"
        echo ""
        echo "  setup       - Create conda env + install dependencies"
        echo "  download    - Download Qwen2.5 models"
        echo "  evaluate    - Evaluate baseline models"
        echo "  train       - Train REINFORCE models (with/without curriculum)"
        echo "  train_grpo  - Train GRPO+DPO models (with/without curriculum)"
        echo "  eval_all    - Evaluate ALL trained models"
        echo "  analyze     - Generate figures and tables"
        echo "  full        - Run entire pipeline end-to-end"
        echo ""
        echo "Config: N_DIALOGUES=$N_DIALOGUES, EPISODES=$EPISODES_PER_EPOCH, EPOCHS=$N_EPOCHS"
        ;;
esac
