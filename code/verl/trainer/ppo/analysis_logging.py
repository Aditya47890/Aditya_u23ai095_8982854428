import json
import os
from typing import Any, Dict

import numpy as np
import torch


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_jsonable(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def use_virtual_dataset(config) -> bool:
    return bool(config.data.get("use_virtual_dataset", True))


def dataset_kwargs_from_config(config) -> Dict[str, Any]:
    analysis_cfg = config.get("analysis", {})
    return {
        "simulator_save_dir": analysis_cfg.get("simulator_log_dir", "simulator_logs"),
        "role_file": config.data.get("role_file", "data/train_profile.jsonl"),
        "scenario_file": analysis_cfg.get("failure_suite_path"),
        "deterministic_sampling": config.data.get("deterministic_sampling", False),
        "simulator_backend": config.data.get("simulator_backend", "live"),
        "simulator_seed": int(config.data.get("simulator_seed", 0)),
    }


def rollout_log_dir(config, global_step: int) -> str:
    analysis_cfg = config.get("analysis", {})
    base_dir = analysis_cfg.get("rollout_log_dir", os.path.join("analysis", "rlver_failures", "reports", "rollouts"))
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(os.getcwd(), base_dir)
    return os.path.join(base_dir, config.trainer.experiment_name, f"global_step_{global_step}")


def maybe_write_rollout_artifacts(config, global_step: int, gen_batch_output, batch=None) -> None:
    if not (config.trainer.save_rollout or config.get("analysis", {}).get("enable_rich_logging", False)):
        return

    output_dir = rollout_log_dir(config=config, global_step=global_step)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "rollout.jsonl")

    reward_details = None
    if batch is not None:
        reward_details = batch.non_tensor_batch.get("reward_details")

    with open(output_path, "w", encoding="utf-8") as handle:
        for index in range(len(gen_batch_output)):
            row = {
                "messages": _jsonable(gen_batch_output.non_tensor_batch.get("messages", [])[index]),
                "emo_point": _jsonable(gen_batch_output.non_tensor_batch.get("emo_point", [])[index]),
                "dialogue_turns": _jsonable(gen_batch_output.non_tensor_batch.get("dialogue_turns", [])[index])
                if "dialogue_turns" in gen_batch_output.non_tensor_batch
                else None,
                "termination_reason": _jsonable(gen_batch_output.non_tensor_batch.get("termination_reason", [])[index])
                if "termination_reason" in gen_batch_output.non_tensor_batch
                else None,
                "emotion_trace": _jsonable(gen_batch_output.non_tensor_batch.get("emotion_trace", [])[index])
                if "emotion_trace" in gen_batch_output.non_tensor_batch
                else None,
                "strategy_tags": _jsonable(gen_batch_output.non_tensor_batch.get("strategy_tags", [])[index])
                if "strategy_tags" in gen_batch_output.non_tensor_batch
                else None,
            }
            if reward_details is not None:
                row["reward_details"] = _jsonable(reward_details[index])
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def collect_analysis_metrics(batch) -> Dict[str, float]:
    details = batch.non_tensor_batch.get("reward_details")
    if details is None:
        return {}

    records = [_jsonable(item) for item in details.tolist()]
    if not records:
        return {}

    emotions = [float(record.get("raw_final_emotion", 0.0)) for record in records]
    rewards = [float(record.get("clipped_reward", 0.0)) for record in records]
    turn_counts = [float(record.get("turn_count", 0.0)) for record in records]

    metrics = {
        "analysis/final_emotion_mean": float(sum(emotions) / len(emotions)),
        "analysis/clipped_reward_mean": float(sum(rewards) / len(rewards)),
        "analysis/turn_count_mean": float(sum(turn_counts) / len(turn_counts)),
    }

    termination_reasons = {}
    strategy_counts = {}
    for record in records:
        reason = str(record.get("termination_reason") or "none")
        termination_reasons[reason] = termination_reasons.get(reason, 0) + 1
        for tag in record.get("strategy_tags", []) or []:
            strategy_counts[tag] = strategy_counts.get(tag, 0) + 1

    total = float(len(records))
    for reason, count in termination_reasons.items():
        metrics[f"analysis/termination_rate/{reason}"] = count / total
    for tag, count in strategy_counts.items():
        metrics[f"analysis/strategy_rate/{tag}"] = count / total

    return metrics
