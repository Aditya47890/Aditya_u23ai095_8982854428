from verl.workers.rollout.vllm_rollout.hard_player_simulator_dsv3 import (
    parse_planning_output,
    parse_player_reply_output,
)


def test_parse_planning_output_recovers_signed_change():
    parsed = parse_planning_output(
        "Content:\n[comfort]\nReason:\n[generic]\nActivity:\n[guarded]\nAnalyse:\n[still unsure]\nChange:\n[+8]"
    )
    assert parsed["content"] == "comfort"
    assert parsed["change"] == 8


def test_parse_planning_output_tolerates_missing_sections():
    parsed = parse_planning_output("Analyse:\n[badly formatted planner output]\nChange:\n[-3]\nExtra: ignored")
    assert parsed["analyse"] == "badly formatted planner output"
    assert parsed["change"] == -3


def test_parse_player_reply_output_falls_back_to_plain_text():
    parsed = parse_player_reply_output("I do not think you get it yet.")
    assert parsed["response"] == "I do not think you get it yet."
    assert parsed["origin"] == "I do not think you get it yet."


def test_parse_player_reply_output_accepts_dict_payload():
    parsed = parse_player_reply_output({"thinking": "brief", "response": "Thanks for listening."})
    assert parsed["thinking"] == "brief"
    assert parsed["response"] == "Thanks for listening."
