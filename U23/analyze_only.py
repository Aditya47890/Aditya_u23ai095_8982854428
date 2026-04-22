import json
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path

RESULTS_DIR = Path("/home/RL/rlver_adversial_manipulation/results")
ROLLING_JSON = RESULTS_DIR / "all_results_rolling.json"

print("Loading results...")
with open(ROLLING_JSON) as f:
    data = json.load(f)
print(f"Total entries: {len(data)}")

rows = []
for r in data:
    m = r.get("metrics", {})
    rows.append({
        "checkpoint": r["checkpoint"],
        "has_thinking": r["has_thinking"],
        "trajectory_type": r["trajectory_type"],
        "dialogue_index": r["dialogue_index"],
        "num_turns": r["num_turns"],
        "final_score": r["final_score"],
        "termination": r["termination"],
        "ecs": m.get("ecs", np.nan),
        "ecs_drop": m.get("ecs_drop", np.nan),
        "volatility": m.get("volatility", np.nan),
        "tqs": m.get("tqs", np.nan),
    })

df = pd.DataFrame(rows)
df["model_name"] = df["checkpoint"].str.replace("-Think","").str.replace("-NoThink","")
df["collapsed"] = df["termination"] == "FAILURE"

ts = datetime.now().strftime("%Y%m%d_%H%M%S")

# full CSV
df.to_csv(RESULTS_DIR / f"full_1p5b_{ts}.csv", index=False)
print(f"Saved full CSV: full_1p5b_{ts}.csv")

# model summary
model_summary = df.groupby("model_name").agg(
    mean_ecs=("ecs","mean"),
    std_ecs=("ecs","std"),
    mean_final_score=("final_score","mean"),
    collapse_rate=("collapsed","mean"),
    mean_ecs_drop=("ecs_drop","mean"),
).round(4)
model_summary.to_csv(RESULTS_DIR / f"model_summary_1p5b_{ts}.csv")
print(f"\nModel Summary:")
print(model_summary.to_string())

# think vs nothink
tv = df.groupby(["model_name","has_thinking"]).agg(
    mean_final_score=("final_score","mean"),
    mean_ecs=("ecs","mean"),
    collapse_rate=("collapsed","mean"),
).round(4)
tv.to_csv(RESULTS_DIR / f"think_vs_nothink_1p5b_{ts}.csv")
print(f"\nThink vs NoThink:")
print(tv.to_string())

# trajectory summary
traj_summary = df.groupby(["model_name","trajectory_type"]).agg(
    mean_final_score=("final_score","mean"),
    mean_ecs=("ecs","mean"),
).round(4)
traj_summary.to_csv(RESULTS_DIR / f"summary_1p5b_{ts}.csv")
print(f"\nTrajectory Summary saved.")

print("\nDone.")
