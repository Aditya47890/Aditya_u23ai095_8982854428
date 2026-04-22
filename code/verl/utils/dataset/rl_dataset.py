# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from omegaconf import ListConfig
import os
from pathlib import Path
from typing import List, Union
import copy
import pandas as pd

import torch
import numpy as np
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from verl.workers.rollout.vllm_rollout.system_prompt import *
from verl.workers.rollout.vllm_rollout.hard_player_simulator_dsv3 import (
    LiveAPISimulatorBackend,
    MockDeterministicBackend,
    PlayerSimulator,
    TranscriptReplayBackend,
)


from verl.utils.py_functional import to_1d_np_array
import json


def _resolve_runtime_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    return os.path.join(os.getcwd(), path)


def _load_failure_scenarios(scenario_file: str | None):
    if not scenario_file:
        return None
    from analysis.rlver_failures.scenario_builders import load_scenarios_from_file

    return [scenario.to_player_data() for scenario in load_scenarios_from_file(Path(scenario_file))]


def _make_backend(backend_name: str, scenario: dict | None):
    backend_name = (backend_name or "live").lower()
    if backend_name == "mock":
        return MockDeterministicBackend()
    if backend_name == "replay":
        replay_steps = []
        if scenario is not None:
            replay_steps = scenario.get("metadata", {}).get("replay_steps", [])
        return TranscriptReplayBackend(replay_steps=replay_steps)
    return LiveAPISimulatorBackend()

def recursive_convert(item):#removes all np arrays
    if isinstance(item, np.ndarray):
        # Convert numpy array to list, and recursively convert its elements
        return recursive_convert(item.tolist())
    elif isinstance(item, list):
        return [recursive_convert(element) for element in item]
    elif isinstance(item, dict):
        return {key: recursive_convert(value) for key, value in item.items()}
    else:
        return item

def collate_fn(data_list: list[dict]) -> dict:
    tensors = {}
    non_tensors = {}

    for data in data_list:
        for key, val in data.items():
            if isinstance(val, torch.Tensor):
                if key not in tensors:
                    tensors[key] = []
                tensors[key].append(val)
            else:
                if key not in non_tensors:
                    non_tensors[key] = []
                non_tensors[key].append(val)

    for key, val in tensors.items():
        tensors[key] = torch.stack(val, dim=0)

    for key, val in non_tensors.items():
        non_tensors[key] = np.array(val, dtype=object)

    output = {}
    output.update(tensors)
    output.update(non_tensors)
    return output


class RLHFDataset(Dataset):
    """
    We assume the dataset contains a column that contains prompts and other information
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 response_key='gt_response',
                 max_prompt_length=1024,
                 max_response_length=1024,
                 filter_prompts=True,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 trajectory_injection=False,
                 simulator_save_dir='simulator_logs',
                 role_file='data/train_profile.jsonl',
                 scenario_file=None,
                 deterministic_sampling=False,
                 simulator_backend='live',
                 simulator_seed=0):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]

        self.parquet_files = copy.deepcopy(parquet_files)
        self.original_parquet_files = copy.deepcopy(parquet_files)  # use for resume
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer

        self.prompt_key = prompt_key
        self.response_key = response_key
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.filter_prompts = filter_prompts

        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.trajectory_injection = trajectory_injection
        self.simulator_save_dir = simulator_save_dir
        self.role_file = role_file
        self.scenario_file = scenario_file
        self.deterministic_sampling = deterministic_sampling
        self.simulator_backend = simulator_backend
        self.simulator_seed = simulator_seed
        self.failure_scenarios = _load_failure_scenarios(scenario_file)

        # whether to store the dataset in state_dict()
        # default not store
        self.serialize_dataset = False
        self._download()
        self._read_files_and_tokenize()

    def _scenario_for_item(self, item):
        if not self.failure_scenarios:
            return None
        return copy.deepcopy(self.failure_scenarios[item % len(self.failure_scenarios)])

    def _create_player_simulator(self, item):
        simulator_dir = _resolve_runtime_path(self.simulator_save_dir)
        os.makedirs(simulator_dir, exist_ok=True)
        scenario = self._scenario_for_item(item)
        backend = _make_backend(self.simulator_backend, scenario=scenario)
        seed = self.simulator_seed + int(item)
        return PlayerSimulator(
            simulator_dir,
            backend=backend,
            scenario=scenario,
            role_file=self.role_file,
            deterministic_sampling=self.deterministic_sampling,
            seed=seed,
        )

    def _download(self, use_origin_parquet=False):
        from verl.utils.fs import copy_to_local
        parquet_files = self.parquet_files if not use_origin_parquet else self.original_parquet_files
        for i, parquet_file in enumerate(parquet_files):
            self.parquet_files[i] = copy_to_local(src=parquet_file, cache_dir=self.cache_dir)

    def _read_files_and_tokenize(self):
        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)

        print(f'original dataset len: {len(self.dataframe)}')

        # filter out too long prompts
        tokenizer = self.tokenizer
        prompt_key = self.prompt_key
        self.dataframe = self.dataframe[self.dataframe.apply(lambda doc: len(
            tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True)) <= self.max_prompt_length,
                                                             axis=1)]

        print(f'filter dataset len: {len(self.dataframe)}')

    def resume_dataset_state(self):
        self.serialize_dataset = False if hasattr(self, 'original_parquet_files') else True
        # resume dataframe if not it's serialized in data.pt
        if not self.serialize_dataset:
            self._download(use_origin_parquet=True)  # download and resume from original parquet files
            self._read_files_and_tokenize()
        else:
            print(r'old dataloader ckpt file is used, please train from scratch for better ckpt performance')

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        player_simulator = self._create_player_simulator(item)
        user_reply=player_simulator.reply(None)
        chat=[{"role":"system","content":system_prompt_trained},{"role":"user","content":user_reply["content"]}]

        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)
        row_dict = self.dataframe.iloc[item].to_dict()
        
        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        if self.trajectory_injection:
            assert self.response_key in row_dict, f"response_key ({self.response_key}) not found in dataset"
            gt_response = row_dict.pop(self.response_key)
            gt_input_ids, gt_attention_mask = verl_F.tokenize_and_postprocess_data(prompt=gt_response + self.tokenizer.eos_token,
                                                                                  tokenizer=self.tokenizer,
                                                                                  max_length=self.max_response_length,
                                                                                  pad_token_id=self.tokenizer.pad_token_id,
                                                                                  left_pad=False,
                                                                                  truncation="right")
            gt_position_ids = compute_position_id_with_mask(gt_attention_mask)
            row_dict['gt_response'] = gt_input_ids[0]
            row_dict['gt_attention_mask'] = gt_attention_mask[0]
            row_dict['gt_position_ids'] = gt_position_ids[0]

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = recursive_convert([{"role":"system","content":system_prompt_trained}]+[user_reply])#makes all internal np arrays into lists
            row_dict['simulator']=player_simulator
            row_dict['scenario_id'] = player_simulator.role.get('id')
        # add index for each prompt
        index = row_dict.get("extra_info", {}).get("index", 0)
        row_dict["index"] = index

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()

            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()


class VirtualRLHFDataset(RLHFDataset):


    def __init__(self,
                 virtual_size: int,
                 tokenizer: PreTrainedTokenizer,
                 prompt_key='prompt',
                 response_key='gt_response',
                 max_prompt_length=1024,
                 max_response_length=1024,
                 cache_dir='~/.cache/verl/rlhf',
                 chat_template_func=None,
                 return_raw_chat=False,
                 truncation='error',
                 trajectory_injection=False,
                 simulator_save_dir='simulator_logs',
                 role_file='data/train_profile.jsonl',
                 scenario_file=None,
                 deterministic_sampling=False,
                 simulator_backend='live',
                 simulator_seed=0):

        self.virtual_size = virtual_size
        self.cache_dir = os.path.expanduser(cache_dir)
        self.tokenizer = tokenizer
        
        self.prompt_key = prompt_key
        self.response_key = response_key
        self.max_prompt_length = max_prompt_length
        self.max_response_length = max_response_length
        self.filter_prompts = False  
        self.return_raw_chat = return_raw_chat
        self.chat_template_func = chat_template_func
        self.truncation = truncation
        self.trajectory_injection = trajectory_injection
        self.simulator_save_dir = simulator_save_dir
        self.role_file = role_file
        self.scenario_file = scenario_file
        self.deterministic_sampling = deterministic_sampling
        self.simulator_backend = simulator_backend
        self.simulator_seed = simulator_seed
        self.failure_scenarios = _load_failure_scenarios(scenario_file)
        
        self.serialize_dataset = False
        
        self._create_virtual_dataframe()
        
        print(f'Created virtual dataset with size: {self.virtual_size}')

    def _create_virtual_dataframe(self):
        import pandas as pd
        
        virtual_data = []
        for i in range(self.virtual_size):
            virtual_data.append({
                'index': i,
                'extra_info': {'index': i},
            })
        
        self.dataframe = pd.DataFrame(virtual_data)

    def _download(self, use_origin_parquet=False):
        pass

    def _read_files_and_tokenize(self):
        pass

    def resume_dataset_state(self):
        self.serialize_dataset = False
        self._create_virtual_dataframe()

    def __len__(self):
        return self.virtual_size

    def __getitem__(self, item):
        player_simulator = self._create_player_simulator(item)
        user_reply = player_simulator.reply(None)

        chat = [{"role":"system","content":system_prompt_trained},{"role":"user","content":user_reply["content"]}]

        prompt_with_chat_template = self.tokenizer.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                         tokenizer=self.tokenizer,
                                                                         max_length=self.max_prompt_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.truncation)

        position_ids = compute_position_id_with_mask(attention_mask)
        
        row_dict = {
            'index': item,
            'extra_info': {'index': item},
        }
        
        row_dict['input_ids'] = input_ids[0]
        row_dict['attention_mask'] = attention_mask[0]
        row_dict['position_ids'] = position_ids[0]

        if self.trajectory_injection:

            pass

        # encode prompts without chat template
        if self.return_raw_chat:
            row_dict['raw_prompt'] = recursive_convert([{"role":"system","content":system_prompt_trained}]+[user_reply])
            row_dict['simulator'] = player_simulator
            row_dict['scenario_id'] = player_simulator.role.get('id')

        return row_dict

    def __getstate__(self):
        if not self.serialize_dataset:
            state = self.__dict__.copy()
            if 'dataframe' in state:
                del state['dataframe']
            return state
        return self.__dict__.copy()
