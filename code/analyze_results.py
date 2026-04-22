"""
RLVER Results Analysis & Visualization
========================================
Generates publication-quality figures and tables from evaluation results.

Produces:
- Model comparison tables (LaTeX-ready)
- Emotion trajectory plots
- Radar charts (multi-dimensional performance)
- Per-scenario heatmaps
- Think vs NoThink comparisons
- Training curve plots

Usage:
    python analyze_results.py --results-dir ./results --output-dir ./figures
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for HPC
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("WARNING: matplotlib not installed. Install with: pip install matplotlib")

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


# ============================================================
# STYLE CONFIGURATION
# ============================================================

COLORS = {
    "Baseline": "#95a5a6",
    "RLVER-PPO": "#3498db", 
    "RLVER-GRPO": "#e74c3c",
    "Improved": "#2ecc71",
    "Improved-LoRA": "#9b59b6",
    "REINFORCE": "#e67e22",
    "REINFORCE-Curriculum": "#f39c12",
    "GRPO-DPO": "#1abc9c",
    "GRPO-DPO-Curriculum": "#16a085",
    "LocalTest": "#7f8c8d",
}

def setup_style():
    if HAS_MATPLOTLIB:
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'figure.figsize': (10, 6),
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
        })


# ============================================================
# DATA LOADING
# ============================================================

def load_results(results_dir):
    """Load all JSON result files from a directory.
    
    Filters out summary/aggregate files (which lack 'turns' data)
    to only include actual dialogue results.
    """
    all_results = []
    results_dir = Path(results_dir)
    
    for f in results_dir.glob("*.json"):
        # Skip summary/aggregate and combined files — they duplicate individual results
        if f.name.startswith("summary_") or f.name.startswith("combined_"):
            continue
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    # Only include entries that are actual dialogue results (have 'turns')
                    for entry in data:
                        if isinstance(entry, dict) and "turns" in entry:
                            all_results.append(entry)
                elif isinstance(data, dict) and "turns" in data:
                    all_results.append(data)
        except Exception as e:
            print(f"  Warning: Could not load {f}: {e}")
    
    print(f"Loaded {len(all_results)} dialogue results")
    return all_results


def results_to_dataframe(results):
    """Convert results list to pandas DataFrame."""
    rows = []
    for r in results:
        model = r.get("model", "unknown")
        thinking = r.get("has_thinking", False)
        traj_type = r.get("trajectory_type", "unknown")
        
        # Compute average reward breakdown across turns
        turns = r.get("turns", [])
        if turns:
            avg_validation = np.mean([t.get("reward_breakdown", {}).get("validation", 0) for t in turns])
            avg_diversity = np.mean([t.get("reward_breakdown", {}).get("diversity", 0) for t in turns])
            avg_trajectory = np.mean([t.get("reward_breakdown", {}).get("trajectory", 0) for t in turns])
            avg_quality = np.mean([t.get("reward_breakdown", {}).get("quality", 0) for t in turns])
            avg_reward = np.mean([t.get("reward_breakdown", {}).get("total", 0) for t in turns])
            
            # Count unique strategies
            all_strategies = set()
            for t in turns:
                all_strategies.update(t.get("strategies_detected", []))
            strategy_count = len(all_strategies)
        else:
            avg_validation = avg_diversity = avg_trajectory = avg_quality = avg_reward = 0
            strategy_count = 0
        
        # Emotion trajectory analysis
        emo_traj = r.get("emotion_trajectory", [50])
        volatility = np.std(emo_traj) if len(emo_traj) > 1 else 0
        
        rows.append({
            "model": model,
            "has_thinking": thinking,
            "mode": "Think" if thinking else "NoThink",
            "model_mode": f"{model}-{'Think' if thinking else 'NoThink'}",
            "trajectory_type": traj_type,
            "num_turns": r.get("num_turns", 0),
            "final_score": r.get("final_score", 0),
            "termination": r.get("termination", "unknown"),
            "success": r.get("termination") in ("SUCCESS", "PARTIAL_SUCCESS"),
            "collapsed": r.get("termination") in ("FAILURE", "PARTIAL_FAILURE"),
            "avg_reward": avg_reward,
            "avg_validation": avg_validation,
            "avg_diversity": avg_diversity, 
            "avg_trajectory": avg_trajectory,
            "avg_quality": avg_quality,
            "strategy_count": strategy_count,
            "volatility": volatility,
        })
    
    return pd.DataFrame(rows)


# ============================================================
# TABLE GENERATORS
# ============================================================

def generate_model_summary_table(df, output_dir):
    """Generate model-level summary table."""
    summary = df.groupby("model").agg(
        mean_score=("final_score", "mean"),
        std_score=("final_score", "std"),
        success_rate=("success", "mean"),
        collapse_rate=("collapsed", "mean"),
        mean_reward=("avg_reward", "mean"),
        mean_validation=("avg_validation", "mean"),
        mean_diversity=("avg_diversity", "mean"),
        strategy_count=("strategy_count", "mean"),
        mean_turns=("num_turns", "mean"),
    ).round(4)
    
    summary.to_csv(os.path.join(output_dir, "table_model_summary.csv"))
    print("\n=== Model Summary ===")
    print(summary.to_string())
    
    # LaTeX version
    latex = summary.to_latex(float_format="%.3f", bold_rows=True)
    with open(os.path.join(output_dir, "table_model_summary.tex"), 'w') as f:
        f.write(latex)
    
    return summary


def generate_think_vs_nothink_table(df, output_dir):
    """Generate think vs nothink comparison."""
    comparison = df.groupby(["model", "mode"]).agg(
        mean_score=("final_score", "mean"),
        success_rate=("success", "mean"),
        mean_reward=("avg_reward", "mean"),
    ).round(4)
    
    comparison.to_csv(os.path.join(output_dir, "table_think_vs_nothink.csv"))
    print("\n=== Think vs NoThink ===")
    print(comparison.to_string())
    return comparison


def generate_per_scenario_table(df, output_dir):
    """Generate per-scenario performance table."""
    scenario = df.groupby(["model", "trajectory_type"]).agg(
        mean_score=("final_score", "mean"),
        success_rate=("success", "mean"),
        mean_reward=("avg_reward", "mean"),
    ).round(4)
    
    scenario.to_csv(os.path.join(output_dir, "table_per_scenario.csv"))
    print("\n=== Per-Scenario ===")
    print(scenario.to_string())
    return scenario


# ============================================================
# VISUALIZATION GENERATORS
# ============================================================

def plot_model_comparison_bar(df, output_dir):
    """Bar chart comparing models on key metrics."""
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    models = df["model"].unique()
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = [
        ("final_score", "Final Emotion Score", axes[0]),
        ("avg_reward", "Multi-Dimensional Reward", axes[1]),
        ("strategy_count", "Avg. Strategies Used", axes[2]),
    ]
    
    for metric, title, ax in metrics:
        means = df.groupby("model")[metric].mean()
        stds = df.groupby("model")[metric].std()
        colors = [COLORS.get(m, "#95a5a6") for m in means.index]
        
        bars = ax.bar(range(len(means)), means.values, yerr=stds.values,
                      color=colors, alpha=0.85, capsize=5, edgecolor='white', linewidth=1.5)
        ax.set_xticks(range(len(means)))
        ax.set_xticklabels(means.index, rotation=30, ha='right')
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel(metric.replace("_", " ").title())
        ax.grid(axis='y', alpha=0.3)
    
    plt.suptitle("Model Performance Comparison", fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_model_comparison.png"))
    plt.close()
    print("Saved: fig_model_comparison.png")


def plot_emotion_trajectories(results, output_dir):
    """Plot example emotion trajectories for each model."""
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    
    # Group by model
    model_trajs = {}
    for r in results:
        model = r.get("model", "unknown")
        if model not in model_trajs:
            model_trajs[model] = []
        model_trajs[model].append(r.get("emotion_trajectory", [50]))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for model, trajs in model_trajs.items():
        # Average trajectory (pad to same length)
        max_len = max(len(t) for t in trajs)
        padded = [t + [t[-1]] * (max_len - len(t)) for t in trajs]
        mean_traj = np.mean(padded, axis=0)
        std_traj = np.std(padded, axis=0)
        
        x = range(len(mean_traj))
        color = COLORS.get(model, "#95a5a6")
        ax.plot(x, mean_traj, label=model, color=color, linewidth=2.5)
        ax.fill_between(x, mean_traj - std_traj, mean_traj + std_traj, 
                        alpha=0.15, color=color)
    
    ax.set_xlabel("Turn", fontsize=13)
    ax.set_ylabel("Emotion Score", fontsize=13)
    ax.set_title("Average Emotion Trajectory per Model", fontsize=15, fontweight='bold')
    ax.legend(loc='lower right', fontsize=11)
    ax.set_ylim(0, 105)
    ax.axhline(y=95, color='green', linestyle='--', alpha=0.4, label='Success threshold')
    ax.axhline(y=10, color='red', linestyle='--', alpha=0.4, label='Failure threshold')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_emotion_trajectory.png"))
    plt.close()
    print("Saved: fig_emotion_trajectory.png")


def plot_radar_chart(df, output_dir):
    """Radar chart comparing models across 5 dimensions."""
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    models = df["model"].unique()
    
    categories = ['Validation', 'Diversity', 'Trajectory', 'Quality', 'Score']
    N = len(categories)
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for model in models:
        model_df = df[df["model"] == model]
        values = [
            model_df["avg_validation"].mean(),
            model_df["avg_diversity"].mean(),
            model_df["avg_trajectory"].mean(),
            model_df["avg_quality"].mean(),
            model_df["final_score"].mean() / 100,
        ]
        values += values[:1]
        
        color = COLORS.get(model, "#95a5a6")
        ax.plot(angles, values, 'o-', linewidth=2.5, label=model, color=color)
        ax.fill(angles, values, alpha=0.1, color=color)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=12)
    ax.set_ylim(0, 1)
    ax.set_title("Multi-Dimensional Performance Radar", fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_radar_chart.png"))
    plt.close()
    print("Saved: fig_radar_chart.png")


def plot_scenario_heatmap(df, output_dir):
    """Heatmap showing model performance per scenario."""
    if not HAS_MATPLOTLIB or not HAS_SEABORN:
        return
    
    setup_style()
    
    pivot = df.pivot_table(
        values="final_score", index="model", 
        columns="trajectory_type", aggfunc="mean"
    )
    
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", 
                linewidths=1, ax=ax, vmin=30, vmax=100,
                cbar_kws={'label': 'Final Score'})
    ax.set_title("Performance per Adversarial Scenario", fontsize=14, fontweight='bold')
    ax.set_ylabel("")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_scenario_heatmap.png"))
    plt.close()
    print("Saved: fig_scenario_heatmap.png")


def plot_reward_breakdown(df, output_dir):
    """Stacked bar chart showing reward breakdown by dimension."""
    if not HAS_MATPLOTLIB:
        return
    
    setup_style()
    models = df["model"].unique()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dimensions = ["avg_validation", "avg_diversity", "avg_trajectory", "avg_quality"]
    labels = ["Validation", "Diversity", "Trajectory", "Quality"]
    colors_dim = ["#3498db", "#e74c3c", "#2ecc71", "#f39c12"]
    
    x = np.arange(len(models))
    width = 0.6
    
    bottom = np.zeros(len(models))
    for dim, label, color in zip(dimensions, labels, colors_dim):
        values = np.array([df[df["model"] == m][dim].mean() for m in models])
        ax.bar(x, values, width, bottom=bottom, label=label, color=color, alpha=0.85)
        bottom += values
    
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha='right')
    ax.set_ylabel("Reward Score")
    ax.set_title("Reward Breakdown by Dimension", fontsize=14, fontweight='bold')
    ax.legend(loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "fig_reward_breakdown.png"))
    plt.close()
    print("Saved: fig_reward_breakdown.png")


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Analyze RLVER evaluation results")
    parser.add_argument("--results-dir", type=str, default="./results")
    parser.add_argument("--output-dir", type=str, default="./figures")
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    results = load_results(args.results_dir)
    if not results:
        print("No results found!")
        return
    
    df = results_to_dataframe(results)
    
    # Generate tables
    generate_model_summary_table(df, args.output_dir)
    generate_think_vs_nothink_table(df, args.output_dir)
    generate_per_scenario_table(df, args.output_dir)
    
    # Generate standard figures
    if HAS_MATPLOTLIB:
        plot_model_comparison_bar(df, args.output_dir)
        plot_emotion_trajectories(results, args.output_dir)
        plot_radar_chart(df, args.output_dir)
        plot_reward_breakdown(df, args.output_dir)
        
        if HAS_SEABORN:
            plot_scenario_heatmap(df, args.output_dir)
    
    # ============================
    # AA-GRADE ANALYSIS EXTENSIONS
    # ============================
    
    # 1. Training reward curves from logs
    if HAS_MATPLOTLIB:
        plot_training_curves(args.results_dir, args.output_dir)
    
    # 2. Win-rate heatmap
    if HAS_MATPLOTLIB and HAS_SEABORN:
        plot_win_rate_heatmap(df, args.output_dir)
    
    # 3. Ablation table with CIs
    generate_ablation_table(df, args.output_dir)
    
    # 4. Diagnostics comparison
    plot_diagnostics(results, args.output_dir)
    
    print(f"\n{'='*60}")
    print(f"All outputs saved to: {args.output_dir}")
    print(f"{'='*60}")


# ============================================================
# PUBLICATION-QUALITY EXTENSIONS
# ============================================================

def plot_training_curves(results_dir, output_dir):
    """Plot training reward/loss curves from training_log.json files."""
    parent = Path(results_dir).parent / "checkpoints"
    if not parent.exists():
        # Try relative to results_dir
        parent = Path(results_dir).parent / "checkpoints"
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    found = False
    for checkpoint_dir in [parent] + list(parent.iterdir()) if parent.exists() else []:
        log_file = checkpoint_dir / "training_log.json" if checkpoint_dir.is_dir() else None
        if log_file is None or not log_file.exists():
            continue
        
        try:
            with open(log_file) as f:
                log = json.load(f)
            
            algo = log.get("algorithm", checkpoint_dir.name)
            color = COLORS.get(algo, None)
            
            # Extract reward curve from stability data or episodes
            if "stability" in log and "reward_curve" in log["stability"]:
                rewards = log["stability"]["reward_curve"]
                losses = log["stability"].get("loss_curve", [])
            elif "episodes" in log:
                rewards = [ep.get("avg_reward", 0) for ep in log["episodes"]]
                losses = [ep.get("loss", 0) for ep in log["episodes"]]
            elif "grpo_episodes" in log:
                rewards = [ep.get("mean_reward", 0) for ep in log["grpo_episodes"]]
                losses = [ep.get("loss", 0) for ep in log["grpo_episodes"]]
            else:
                continue
            
            if rewards:
                # Smooth with moving average
                window = min(5, len(rewards))
                if window > 1:
                    smoothed = np.convolve(rewards, np.ones(window)/window, mode='valid')
                else:
                    smoothed = rewards
                axes[0].plot(smoothed, label=algo, color=color, linewidth=2)
                found = True
            
            if losses:
                window = min(5, len(losses))
                if window > 1:
                    smoothed_l = np.convolve(losses, np.ones(window)/window, mode='valid')
                else:
                    smoothed_l = losses
                axes[1].plot(smoothed_l, label=algo, color=color, linewidth=2)
        except Exception:
            continue
    
    if found:
        axes[0].set_title("Training Reward Curve", fontweight='bold')
        axes[0].set_xlabel("Episode")
        axes[0].set_ylabel("Mean Reward")
        axes[0].legend(fontsize=9)
        axes[0].grid(True, alpha=0.3)
        
        axes[1].set_title("Training Loss Curve", fontweight='bold')
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Loss")
        axes[1].legend(fontsize=9)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        path = os.path.join(output_dir, "fig_training_curves.png")
        plt.savefig(path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")


def plot_win_rate_heatmap(df, output_dir):
    """Generate pairwise win-rate heatmap between all models."""
    from training_utils import compute_win_rate
    
    models = sorted(df['model'].unique())
    if len(models) < 2:
        return
    
    win_matrix = np.zeros((len(models), len(models)))
    
    for i, m1 in enumerate(models):
        for j, m2 in enumerate(models):
            if i == j:
                win_matrix[i][j] = 0.5
                continue
            scores_1 = df[df['model'] == m1]['final_score'].values
            scores_2 = df[df['model'] == m2]['final_score'].values
            n = min(len(scores_1), len(scores_2))
            if n > 0:
                wr, _, _ = compute_win_rate(scores_1[:n], scores_2[:n])
                win_matrix[i][j] = wr
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(win_matrix, cmap='RdYlGn', vmin=0, vmax=1)
    
    # Labels
    ax.set_xticks(range(len(models)))
    ax.set_yticks(range(len(models)))
    ax.set_xticklabels(models, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(models, fontsize=9)
    
    # Add values
    for i in range(len(models)):
        for j in range(len(models)):
            text = f"{win_matrix[i][j]:.2f}"
            color = 'white' if win_matrix[i][j] < 0.3 or win_matrix[i][j] > 0.7 else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)
    
    ax.set_title("Pairwise Win-Rate Heatmap (Row vs Column)", fontweight='bold')
    ax.set_xlabel("Opponent")
    ax.set_ylabel("Model")
    plt.colorbar(im, ax=ax, label="Win Rate")
    plt.tight_layout()
    
    path = os.path.join(output_dir, "fig_win_rate_heatmap.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def generate_ablation_table(df, output_dir):
    """Generate complete ablation results table with CIs in LaTeX format."""
    from training_utils import compute_confidence_interval
    
    rows = []
    for (model, mode), group in df.groupby(['model', 'mode']):
        scores = group['final_score'].values
        mean, ci_lo, ci_hi, std, n = compute_confidence_interval(scores)
        success = (scores >= 95).sum() / len(scores) * 100
        failure = (scores <= 10).sum() / len(scores) * 100
        
        rows.append({
            "Model": model,
            "Mode": mode,
            "N": n,
            "Score": f"{mean:.1f}",
            "95% CI": f"[{ci_lo:.1f}, {ci_hi:.1f}]",
            "Std": f"{std:.1f}",
            "Success%": f"{success:.0f}",
            "Failure%": f"{failure:.0f}",
        })
    
    ablation_df = pd.DataFrame(rows)
    
    # Save as CSV
    csv_path = os.path.join(output_dir, "table_ablation_full.csv")
    ablation_df.to_csv(csv_path, index=False)
    
    # Save as LaTeX
    tex_path = os.path.join(output_dir, "table_ablation_full.tex")
    with open(tex_path, 'w') as f:
        f.write("\\begin{table}[ht]\n")
        f.write("\\centering\n")
        f.write("\\caption{Complete Ablation Results with 95\\% Confidence Intervals}\n")
        f.write("\\label{tab:ablation}\n")
        f.write("\\begin{tabular}{llccccc}\n")
        f.write("\\toprule\n")
        f.write("Model & Mode & N & Score & 95\\% CI & Success\\% & Failure\\% \\\\\n")
        f.write("\\midrule\n")
        for _, row in ablation_df.iterrows():
            f.write(f"{row['Model']} & {row['Mode']} & {row['N']} & "
                    f"{row['Score']} & {row['95% CI']} & "
                    f"{row['Success%']} & {row['Failure%']} \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"  Saved: {csv_path}")
    print(f"  Saved: {tex_path}")


def plot_diagnostics(results, output_dir):
    """Plot reward hacking diagnostics: response length vs reward."""
    if not HAS_MATPLOTLIB:
        return
    
    # Collect per-model diagnostics
    model_data = {}
    results_list = results if isinstance(results, list) else [r for v in results.values() for r in (v if isinstance(v, list) else [v])]
    for r in results_list:
        model = r.get("model", "Unknown")
        diag = r.get("diagnostics", {})
        if not diag:
            continue
        if model not in model_data:
            model_data[model] = {"lengths": [], "rewards": [], "generic": []}
        model_data[model]["lengths"].append(diag.get("avg_response_length_words", 0))
        model_data[model]["rewards"].append(r.get("final_score", 0))
        model_data[model]["generic"].append(diag.get("generic_templates_per_turn", 0))
    
    if not model_data:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Response Length vs Final Score
    for model, data in model_data.items():
        color = COLORS.get(model, '#333333')
        axes[0].scatter(data["lengths"], data["rewards"], label=model,
                       color=color, alpha=0.6, s=40)
    axes[0].set_xlabel("Avg Response Length (words)")
    axes[0].set_ylabel("Final Emotion Score")
    axes[0].set_title("Response Length vs Score\n(Reward Hacking Check)", fontweight='bold')
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Generic Template Rate by Model
    models_sorted = sorted(model_data.keys())
    generic_means = [np.mean(model_data[m]["generic"]) for m in models_sorted]
    bars_colors = [COLORS.get(m, '#333333') for m in models_sorted]
    axes[1].barh(models_sorted, generic_means, color=bars_colors)
    axes[1].set_xlabel("Generic Templates per Turn")
    axes[1].set_title("Generic Empathy Template Usage\n(Lower = More Original)", fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    path = os.path.join(output_dir, "fig_diagnostics.png")
    plt.savefig(path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


if __name__ == "__main__":
    main()

