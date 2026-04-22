from analysis.rlver_failures.scenario_builders import build_scenarios, load_named_suite


def test_load_named_suite_has_expected_fields():
    scenarios = load_named_suite("reward_hacking")
    assert scenarios
    scenario = scenarios[0]
    assert scenario.suite == "reward_hacking"
    assert scenario.hidden_topic
    assert scenario.first_talk


def test_build_scenarios_loads_multiple_suites():
    scenarios = build_scenarios(selected_suites=["reward_hacking", "parser_fuzz"])
    scenario_ids = {scenario.scenario_id for scenario in scenarios}
    assert scenario_ids
    assert any(suite in scenario.suite for scenario in scenarios for suite in ["reward_hacking", "parser_fuzz"])


def test_scenario_to_player_data_keeps_hidden_topic():
    scenario = load_named_suite("contradictory_emotion")[0]
    payload = scenario.to_player_data()
    assert payload["hidden_topic"] == scenario.hidden_topic
    assert payload["history"] == []
