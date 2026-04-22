from analysis.rlver_failures.policies import build_policies
from analysis.rlver_failures.run_failure_suite import run_episode
from analysis.rlver_failures.scenario_builders import load_named_suite


def test_failure_suite_mock_backend_runs(tmp_path):
    scenario = load_named_suite("reward_hacking")[0]
    policy = build_policies(["praise_spam"])[0]

    result = run_episode(
        scenario=scenario,
        policy=policy,
        backend_name="mock",
        max_turns=3,
        save_dir=tmp_path,
    )

    assert result["backend"] == "mock"
    assert result["metrics"]["emotion_reward"] >= 0.0
    assert result["transcript"]
    assert "simulator_snapshot" in result
