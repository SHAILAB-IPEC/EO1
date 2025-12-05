# Copyright 2025 EO-Robotics Team. All rights reserved.
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

import copy
import math
import random
import re

import torch
import ujson as json
from datasets import load_dataset
from torch.utils.data import Dataset

from eo.constants import (
    ACTION_END_TOKEN,
    ACTION_START_TOKEN,
    DEFAULT_ACTION_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_STATE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    LLAVA_ACTION_TOKEN,
    LLAVA_IMAGE_TOKEN,
    LLAVA_STATE_TOKEN,
    LLAVA_VIDEO_TOKEN,
    LLAVA_VLA_TOKEN,
    STATE_END_TOKEN,
    STATE_START_TOKEN,
    TASK_VLA_TOKEN,
    VISION_END_TOKEN,
    VISION_START_TOKEN,
)
from eo.data.schema import MMDatasetConfig


class MultimodaDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        data_configs: list[MMDatasetConfig],
        max_seq_length: int = 16384,
        max_action_dim: int = 32,
        chunk_size: int = 50,
    ):
        super().__init__()
        self.data_configs = data_configs
        self.max_action_dim = max_action_dim
        self.chunk_size = chunk_size

        list_data_dict, dataset_lens = [], []
        list_hf_datasets = []
        seq_lengths = []

        for i, dataset in enumerate(data_configs):
            json_path = dataset.json_path
            sampling_strategy = dataset.sampling_strategy
            sampling_number = None

            if json_path.endswith(".jsonl"):
                cur_data_dict = []
                for line in open(json_path):
                    cur_data_dict.append(json.loads(line.strip()))
            elif json_path.endswith(".json"):
                cur_data_dict = json.load(open(json_path))
            else:
                print(f"Loading HF dataset from {json_path}")
                hf_dataset = load_dataset(json_path, split="train")
                len_hf_ds = len(hf_dataset)

                # set seq_length for packing
                hf_dataset_lens = hf_dataset["seq_length"]
                cur_data_dict = []

                for idx in range(len_hf_ds):
                    cur_data_dict.append({
                        "hf_idx": len(list_hf_datasets),
                        "data_idx": idx,
                        "seq_length": hf_dataset_lens[idx]
                    })
                list_hf_datasets.append(hf_dataset)

            # NOTE: filter out lines above MAX_SEQ_LENGTH
            cur_data_dict = [
                line for line in cur_data_dict if line.get("seq_length", 0) <= max_seq_length
            ]

            if ":" in sampling_strategy:
                sampling_strategy, sampling_number = sampling_strategy.split(":")
                if "%" in sampling_number:
                    sampling_number = math.ceil(
                        float(sampling_number.split("%")[0]) * len(cur_data_dict) / 100
                    )
                else:
                    sampling_number = int(sampling_number)

            # sampling
            if sampling_strategy == "first" and sampling_number is not None:
                cur_data_dict = cur_data_dict[:sampling_number]
            elif sampling_strategy == "end" and sampling_number is not None:
                cur_data_dict = cur_data_dict[-sampling_number:]
            elif sampling_strategy == "random" and sampling_number is not None:
                random.shuffle(cur_data_dict)
                cur_data_dict = cur_data_dict[:sampling_number]

            print(f"Loaded {len(cur_data_dict)} samples from {json_path}")
            dataset_lens.append(len(cur_data_dict))
            for data in cur_data_dict:
                data["vision_base_path"] = dataset.vision_base_path
            list_data_dict.extend(cur_data_dict)

            # prepare lens for packing
            for line in cur_data_dict:
                seq_len = line.get("seq_length", 0)
                seq_lengths.append(seq_len)
                if seq_len == 0:
                    print(f"[Warning] {seq_len=}, {json_path=}, {line=}, \
                    please group length for data packing usage")

        self.json_data = list_data_dict
        self.hf_datas = list_hf_datasets
        self.dataset_lens = dataset_lens
        self.cached_lengths = seq_lengths

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, i) -> dict[str, torch.Tensor]:
        sources = self.json_data[i]
        key = "conversations"
        if "hf_idx" in sources:
            hf_idx, data_idx = sources["hf_idx"], sources["data_idx"]
            vision_base_path = sources["vision_base_path"]
            sources = self.hf_datas[hf_idx][data_idx]
            sources["vision_base_path"] = vision_base_path
            key = "conversation"

        transformed_source = copy.deepcopy(sources)
        transformed_source["conversations"] = llava_to_openai(
            transformed_source[key], "video" in sources
        )
        return transformed_source

    @property
    def lengths(self) -> list[int]:
        return self.cached_lengths


def replace_image_tokens(input_string, is_video=False):
    if is_video:
        pattern = r"\s*" + re.escape(LLAVA_VIDEO_TOKEN) + r"\n?"
        replacement = VISION_START_TOKEN + DEFAULT_VIDEO_TOKEN + VISION_END_TOKEN
    else:
        pattern = r"\s*" + re.escape(LLAVA_IMAGE_TOKEN) + r"\n?"
        replacement = VISION_START_TOKEN + DEFAULT_IMAGE_TOKEN + VISION_END_TOKEN
    return re.sub(pattern, replacement, input_string)


def replace_action_tokens(input_string):
    pattern = r"\s*" + re.escape(LLAVA_ACTION_TOKEN) + r"\n?"
    replacement = f"{ACTION_START_TOKEN}{DEFAULT_ACTION_TOKEN}{ACTION_END_TOKEN}"
    return re.sub(pattern, replacement, input_string)


def replace_vla_tokens(input_string):
    pattern = r"\s*" + re.escape(LLAVA_VLA_TOKEN) + r"\n?"
    replacement = TASK_VLA_TOKEN
    return re.sub(pattern, replacement, input_string)


def replace_state_tokens(input_string):
    pattern = r"\s*" + re.escape(LLAVA_STATE_TOKEN) + r"\n?"
    replacement = f"{STATE_START_TOKEN}{DEFAULT_STATE_TOKEN}{STATE_END_TOKEN}"
    return re.sub(pattern, replacement, input_string)


def llava_to_openai(conversations, is_video=False):
    role_mapping = {"human": "user", "gpt": "assistant"}
    transformed_data = []
    for conversation in conversations:
        transformed_content = replace_image_tokens(conversation["value"], is_video=is_video)
        transformed_content = replace_action_tokens(transformed_content)
        transformed_content = replace_state_tokens(transformed_content)
        transformed_content = replace_vla_tokens(transformed_content)
        transformed_entry = {
            "role": role_mapping.get(conversation["from"], conversation["from"]),
            "content": transformed_content,
        }
        transformed_data.append(transformed_entry)
    return transformed_data


def pad_vector(vector, new_dim):
    """Can be (b s e) or (b e)"""
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector
