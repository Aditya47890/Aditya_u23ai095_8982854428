"""
RLVER Improved Multi-Dimensional Reward Function
=================================================
Replaces the single-scalar emo_point/100 reward with a 4-dimensional
empathy score that captures emotional validation, strategy diversity,
conversation progression, and response quality.

This is a DROP-IN replacement for url_environment.py's reward computation.
"""

import numpy as np
import re
from difflib import SequenceMatcher


# ============================================================
# EMPATHY STRATEGY DETECTION
# ============================================================

EMPATHY_STRATEGIES = {
    "emotional_validation": [
        "i understand", "that must be", "i can see why", "it makes sense that",
        "your feelings are valid", "that sounds really", "i hear you",
        "it's okay to feel", "it's natural to", "anyone would feel",
        "that sounds", "how disappointing", "how frustrating", "frustrating",
        "discouraging", "overwhelming", "i can imagine", "that's tough",
        "that's hard", "i appreciate you sharing"
    ],
    "reflective_listening": [
        "so what you're saying", "it sounds like", "if i understand correctly",
        "what i'm hearing is", "let me make sure i understand", "you're saying that"
    ],
    "open_questioning": [
        "can you tell me more", "how did that make you feel", "what do you think",
        "what would help", "how are you feeling about", "what happened next",
        "could you share more"
    ],
    "reframing": [
        "one way to look at", "another perspective", "have you considered",
        "what if we think about", "on the other hand", "looking at it differently"
    ],
    "normalization": [
        "it's completely normal", "many people feel", "you're not alone",
        "it's very common to", "lots of people go through", "that's a natural reaction"
    ],
    "practical_suggestion": [
        "have you tried", "one thing that might help", "you could consider",
        "it might be worth", "a good approach could be", "some people find it helpful"
    ],
    "empowerment": [
        "you've already shown", "you're stronger than", "the fact that you",
        "it takes courage", "you should be proud", "that shows real strength"
    ],
    "presence": [
        "i'm here for you", "take your time", "there's no rush",
        "i'm listening", "you can share whenever", "i'm not going anywhere"
    ]
}


def detect_strategies(text: str) -> set:
    """Detect which empathy strategies are present in a response."""
    text_lower = text.lower()
    found = set()
    for strategy, keywords in EMPATHY_STRATEGIES.items():
        for keyword in keywords:
            if keyword in text_lower:
                found.add(strategy)
                break
    return found


def compute_validation_score(response: str) -> float:
    """Score 0-1: How much emotional validation the response contains."""
    strategies = detect_strategies(response)
    validation_strategies = {"emotional_validation", "normalization", "presence"}
    found = strategies & validation_strategies
    return min(len(found) / 2.0, 1.0)  # Cap at 1.0


def compute_strategy_diversity(agent_responses: list) -> float:
    """Score 0-1: How diverse the empathy strategies are across turns."""
    if len(agent_responses) < 2:
        return 0.5  # Neutral for first turn
    
    all_strategies = set()
    per_turn_strategies = []
    for resp in agent_responses:
        turn_strategies = detect_strategies(resp)
        all_strategies |= turn_strategies
        per_turn_strategies.append(turn_strategies)
    
    # Reward: using many different strategies
    strategy_coverage = len(all_strategies) / len(EMPATHY_STRATEGIES)
    
    # Penalty: repeating exact same text patterns
    if len(agent_responses) >= 2:
        last_two_sim = SequenceMatcher(
            None, agent_responses[-1].lower(), agent_responses[-2].lower()
        ).ratio()
        repetition_penalty = max(0, last_two_sim - 0.5) * 2  # Penalty if >50% similar
    else:
        repetition_penalty = 0
    
    return min(max(strategy_coverage - repetition_penalty * 0.3, 0), 1.0)


def compute_trajectory_score(emo_point: float, num_turns: int) -> float:
    """Score 0-1: Reward for emotion trajectory improvement and efficiency."""
    # Base: how high is the emotion?
    base = emo_point / 100.0
    
    # Efficiency bonus: reaching high emotion in fewer turns is better
    if emo_point >= 80 and num_turns <= 5:
        efficiency_bonus = 0.15
    elif emo_point >= 60 and num_turns <= 6:
        efficiency_bonus = 0.05
    else:
        efficiency_bonus = 0.0
    
    return min(base + efficiency_bonus, 1.0)


def compute_response_quality(response: str) -> float:
    """Score 0-1: Response quality based on length, structure, and naturalness."""
    words = response.split()
    word_count = len(words)
    
    # Optimal length: 30-150 words
    if 30 <= word_count <= 150:
        length_score = 1.0
    elif word_count < 10:
        length_score = 0.2  # Way too short
    elif word_count < 30:
        length_score = 0.5 + (word_count - 10) / 40  # Ramp up
    elif word_count <= 200:
        length_score = 0.8  # Slightly verbose but ok
    else:
        length_score = 0.5  # Too verbose
    
    # Check for question marks (engagement)
    has_question = '?' in response
    question_bonus = 0.1 if has_question else 0.0
    
    # Penalty for generic filler responses
    generic_phrases = [
        "i'm sorry to hear that", "that must be really hard",
        "everything will be okay", "things will get better"
    ]
    generic_count = sum(1 for p in generic_phrases if p in response.lower())
    generic_penalty = generic_count * 0.1
    
    return min(max(length_score + question_bonus - generic_penalty, 0), 1.0)


# ============================================================
# MAIN REWARD FUNCTION
# ============================================================

def compute_multi_dimensional_reward(
    messages: list,
    emo_point: float,
    num_turns: int = 0,
    weights: dict = None
) -> float:
    """
    Compute multi-dimensional empathy reward.
    
    Args:
        messages: Full conversation history [{"role": ..., "content": ...}, ...]
        emo_point: Current emotion score (0-100)
        num_turns: Number of turns completed
        weights: Optional custom weights for each dimension
    
    Returns:
        float: Reward in range [0, 1]
    """
    if weights is None:
        weights = {
            "validation": 0.30,     # Emotional validation
            "diversity": 0.20,      # Strategy diversity
            "trajectory": 0.30,     # Emotion trajectory
            "quality": 0.20,        # Response quality
        }
    
    # Extract agent responses 
    agent_responses = [
        m["content"] for m in messages 
        if m.get("role") == "assistant" and m.get("content")
    ]
    
    if not agent_responses:
        return 0.0
    
    last_response = agent_responses[-1]
    
    # Compute each dimension
    validation = compute_validation_score(last_response)
    diversity = compute_strategy_diversity(agent_responses)
    trajectory = compute_trajectory_score(emo_point, num_turns)
    quality = compute_response_quality(last_response)
    
    # Weighted combination
    reward = (
        weights["validation"] * validation +
        weights["diversity"] * diversity +
        weights["trajectory"] * trajectory +
        weights["quality"] * quality
    )
    
    return float(np.clip(reward, 0.0, 1.0))


def compute_reward_with_breakdown(
    messages: list,
    emo_point: float,
    num_turns: int = 0
) -> dict:
    """Same as above but returns detailed breakdown for analysis."""
    agent_responses = [
        m["content"] for m in messages 
        if m.get("role") == "assistant" and m.get("content")
    ]
    
    if not agent_responses:
        return {"total": 0.0, "validation": 0.0, "diversity": 0.0, 
                "trajectory": 0.0, "quality": 0.0}
    
    last_response = agent_responses[-1]
    
    validation = compute_validation_score(last_response)
    diversity = compute_strategy_diversity(agent_responses)
    trajectory = compute_trajectory_score(emo_point, num_turns)
    quality = compute_response_quality(last_response)
    
    total = 0.30 * validation + 0.20 * diversity + 0.30 * trajectory + 0.20 * quality
    
    return {
        "total": float(np.clip(total, 0.0, 1.0)),
        "validation": round(validation, 4),
        "diversity": round(diversity, 4),
        "trajectory": round(trajectory, 4),
        "quality": round(quality, 4),
        "strategies_used": list(detect_strategies(last_response)),
    }


# ============================================================
# QUICK TEST
# ============================================================
if __name__ == "__main__":
    # Test with a sample conversation
    test_messages = [
        {"role": "user", "content": "I just failed my exam and I feel terrible about it."},
        {"role": "assistant", "content": "I understand how disappointing that must feel. Failing an exam when you've worked hard can be really discouraging. Can you tell me more about what happened? Sometimes talking through it helps."},
        {"role": "user", "content": "I studied so hard but I just blanked out during the test."},
        {"role": "assistant", "content": "That sounds incredibly frustrating - you put in the effort but your mind just wouldn't cooperate. That's actually a very common experience, especially under pressure. It doesn't mean you don't know the material. Have you considered trying some relaxation techniques before exams? Many students find deep breathing really helps with test anxiety."},
    ]
    
    result = compute_reward_with_breakdown(test_messages, emo_point=65, num_turns=2)
    print("=== Reward Breakdown ===")
    for k, v in result.items():
        print(f"  {k}: {v}")
