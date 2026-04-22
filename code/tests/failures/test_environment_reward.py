import numpy as np
import torch

try:
    from tensordict import TensorDict
    from verl import DataProto
    from verl.environments.url_environment import URLEnvironment
    DEPENDENCIES_AVAILABLE = DataProto is not None
except ModuleNotFoundError:
    TensorDict = None
    DataProto = None
    URLEnvironment = None
    DEPENDENCIES_AVAILABLE = False


class DummyTokenizer:
    pass


def test_environment_attaches_reward_details_and_sparse_reward():
    if not DEPENDENCIES_AVAILABLE:
        return
    env = URLEnvironment(config=None, tokenizer=DummyTokenizer())
    batch = TensorDict(
        {
            "prompts": torch.tensor([[11, 12]]),
            "responses": torch.tensor([[21, 22, 23]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 0]]),
        },
        batch_size=[1],
    )
    data = DataProto(
        batch=batch,
        non_tensor_batch={
            "messages": np.array([[{"role": "assistant", "content": "reply"}]], dtype=object),
            "emo_point": np.array([70], dtype=object),
            "dialogue_turns": np.array([2], dtype=object),
            "termination_reason": np.array(["user_goodbye"], dtype=object),
            "emotion_trace": np.array([[30, 70]], dtype=object),
            "strategy_tags": np.array([["praise_or_validation"]], dtype=object),
        },
    )

    original_reward, penalized_reward = env.get_reward_batched(data)

    assert float(original_reward.sum()) == 0.7
    assert float(penalized_reward.sum()) == 0.7
    reward_details = data.non_tensor_batch["reward_details"][0]
    assert reward_details["raw_final_emotion"] == 70.0
    assert reward_details["turn_count"] == 2
    assert reward_details["termination_reason"] == "user_goodbye"


def test_environment_clips_negative_reward_to_zero():
    if not DEPENDENCIES_AVAILABLE:
        return
    env = URLEnvironment(config=None, tokenizer=DummyTokenizer())
    batch = TensorDict(
        {
            "prompts": torch.tensor([[11, 12]]),
            "responses": torch.tensor([[21, 22]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
        },
        batch_size=[1],
    )
    data = DataProto(
        batch=batch,
        non_tensor_batch={
            "messages": np.array([[{"role": "assistant", "content": "reply"}]], dtype=object),
            "emo_point": np.array([-5], dtype=object),
        },
    )

    original_reward, penalized_reward = env.get_reward_batched(data)
    assert float(original_reward.sum()) == 0.0
    assert float(penalized_reward.sum()) == 0.0
