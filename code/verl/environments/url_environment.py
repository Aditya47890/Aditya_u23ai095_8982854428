from verl import DataProto
import torch
import numpy as np


class URLEnvironment():

    def __init__(self, config, tokenizer):
        self.config = config
        self.tokenizer = tokenizer


    def get_reward_batched(self, data: DataProto):  #batched
        messages_batched = []
        reward_locs = []
        reward_details = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            messages = data_item.non_tensor_batch['messages']
            if isinstance(messages, np.ndarray):
                messages = messages.tolist()
            messages_batched.append(messages)

            attention_mask = data_item.batch['attention_mask']
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_response_length = attention_mask[prompt_length:].sum()
            reward_locs.append(valid_response_length - 1)

            raw_final_emotion = float(data_item.non_tensor_batch['emo_point'])
            clipped_reward = max(raw_final_emotion / 100.0, 0.0)
            emotion_trace = data_item.non_tensor_batch.get('emotion_trace', [])
            if isinstance(emotion_trace, np.ndarray):
                emotion_trace = emotion_trace.tolist()
            strategy_tags = data_item.non_tensor_batch.get('strategy_tags', [])
            if isinstance(strategy_tags, np.ndarray):
                strategy_tags = strategy_tags.tolist()
            termination_reason = data_item.non_tensor_batch.get('termination_reason')
            if isinstance(termination_reason, np.ndarray):
                termination_reason = termination_reason.tolist()
            dialogue_turns = data_item.non_tensor_batch.get('dialogue_turns')
            if isinstance(dialogue_turns, np.ndarray):
                dialogue_turns = dialogue_turns.tolist()
            if isinstance(dialogue_turns, list):
                turn_count = int(dialogue_turns[-1]) if dialogue_turns else 0
            elif dialogue_turns is None:
                turn_count = sum(1 for message in messages if message.get('role') == 'assistant')
            else:
                turn_count = int(dialogue_turns)

            existing_details = data_item.non_tensor_batch.get('reward_details', {})
            if isinstance(existing_details, np.ndarray):
                existing_details = existing_details.tolist()
            if isinstance(existing_details, list):
                existing_details = existing_details[0] if existing_details else {}
            if not isinstance(existing_details, dict):
                existing_details = {}
            reward_details.append({
                **existing_details,
                'raw_final_emotion': raw_final_emotion,
                'clipped_reward': clipped_reward,
                'turn_count': turn_count,
                'termination_reason': termination_reason,
                'emotion_trace': emotion_trace,
                'strategy_tags': strategy_tags,
                'sparse_reward_loc': int(reward_locs[-1]),
            })

        # reward_batched = requests.post(url, json=payload).json()
        reward_batched = data.non_tensor_batch['emo_point']/100
        reward_batched = np.maximum(reward_batched, 0)
        original_reward_batched = reward_batched.copy()



        original_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        penalized_reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)
        
        for i in range(len(data)):
            original_reward_tensor[i, reward_locs[i]] = original_reward_batched[i]
            penalized_reward_tensor[i, reward_locs[i]] = reward_batched[i]

        data.non_tensor_batch['reward_details'] = np.array(reward_details, dtype=object)
        
        return original_reward_tensor, penalized_reward_tensor
