# RLVER Paper-to-Code Claim Map

| Intended capability | Repo evidence | Failure-analysis hook |
| --- | --- | --- |
| RL with verifiable emotion rewards | `code/verl/environments/url_environment.py` converts final `emo_point` into scalar reward | Reward hacking, threshold saturation, sparse terminal reward |
| Simulated user environment | `code/verl/workers/rollout/vllm_rollout/hard_player_simulator_dsv3.py` drives user replies and emotion transitions | Parser fuzzing, persona bias, hidden-topic exploitation |
| PPO / GRPO style policy optimization | `code/verl/trainer/ppo/ray_trainer.py`, `ray_trainer_think.py`, `core_algos.py` | PPO instability, mode collapse, reward-length pathologies |
| Think-then-say behavior | `code/verl/trainer/ppo/ray_trainer_think.py` and `thinking=True` rollout path | Token inefficiency, shallow reasoning, reasoning leakage |
| Multi-turn empathetic dialogue | `vllm_multi_turn_via_chat` rollout path and simulator loop | Long-horizon drift, contradiction blindness, premature goodbye |
| End-to-end reproducibility | `README.md`, `code/train_rlver.sh`, external SAGE instructions | Placeholder API dependency, external evaluation gap, hardcoded virtual data |

Proof levels used in the failure framework:

- `code_audit`: directly supported by the shipped code
- `repo_experiment`: demonstrated by the mock or replay failure harness
- `framework_ready`: instrumented now and ready for stronger checkpoint-backed validation later
