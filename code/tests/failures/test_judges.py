from analysis.rlver_failures.judges import (
    compute_genericity_score,
    compute_hidden_topic_exploit_rate,
    compute_reward_empathy_mismatch,
    summarize_episode,
)
from analysis.rlver_failures.scenario_builders import load_named_suite


def test_genericity_score_detects_template_language():
    transcript = [
        {"role": "assistant", "content": "I hear you. You deserve care."},
        {"role": "assistant", "content": "I hear you. You deserve care."},
    ]
    assert compute_genericity_score(transcript) > 0.4


def test_hidden_topic_exploit_rate_increases_when_topic_is_named():
    scenario = load_named_suite("hidden_topic_exploitation")[0]
    transcript = [{"role": "assistant", "content": f"You keep circling back to {scenario.hidden_topic}."}]
    assert compute_hidden_topic_exploit_rate(scenario, transcript) > 0.0


def test_reward_empathy_mismatch_flags_reward_hacking_pattern():
    scenario = load_named_suite("reward_hacking")[0]
    transcript = [{"role": "assistant", "content": "You are valid and strong. You deserve care."}]
    mismatch = compute_reward_empathy_mismatch(final_emotion=80, transcript=transcript, scenario=scenario)
    assert mismatch > 0.1


def test_summarize_episode_returns_metric_bundle():
    scenario = load_named_suite("reward_hacking")[0]
    transcript = [
        {"role": "user", "content": scenario.first_talk},
        {"role": "assistant", "content": "I hear you. That sounds heavy."},
    ]
    metrics = summarize_episode(
        scenario=scenario,
        transcript=transcript,
        final_emotion=55,
        emotion_trace=[30, 55],
        termination_reason="none",
    )
    assert "emotion_reward" in metrics
    assert "reward_empathy_mismatch" in metrics
