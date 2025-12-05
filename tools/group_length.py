"""caculate parquet dataset length
. /mnt/shared-storage-user/eorobotics-shared/miniconda3/etc/profile.d/conda.sh
conda activate eo
"""

import torch
import os
import copy
import logging
from tqdm import tqdm
import os.path as osp
from datasets import load_dataset

from eo.model.processing_eo1 import EO1VisionProcessor
from eo.data.dataset import get_image_info, get_video_info
from eo.data.multim_dataset import llava_to_openai
from eo.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    DEFAULT_VIDEO_TOKEN,
    IGNORE_INDEX,
    SYSTEM_MESSAGE,
)

logger = logging.getLogger(__name__)

data_base = ""
save_base = ""
video_folder = ""

datas = [
    # your json paths
]

processor = EO1VisionProcessor.from_pretrained(
    "IPEC-COMMUNITY/eo1-qwen2_5_vl_hf",
    trust_remote_code=True,
    padding_side="right",
    use_fast=True,
)


def data_len_fn(sources):
    if "image" in sources:
        videos = None
        grid_key = "image_grid_thw"
        pixel_key = "pixel_values"
        image_files = sources["image"]
        images = []
        
        if not isinstance(image_files, list):
            image_files = [image_files]
        
        for image_file in image_files:
            images.append(
                get_image_info(
                    image_file,
                    64 * 28 * 28,
                    128 * 28 * 28,
                    None,
                    None,
                )
            )

    elif "video" in sources:
        images = None
        grid_key = "video_grid_thw"
        pixel_key = "pixel_values_videos"
        video_files = sources["video"]
        if isinstance(video_files, str):
            video_files = [video_files]
        videos = []
        for video_file in video_files:
            if isinstance(video_file, str) and not video_file.startswith("http"):
                video_file = os.path.join(video_folder, video_file)
            video_input, video_kwargs = get_video_info(
                video_file,
                64 * 28 * 28,
                128 * 28 * 28,
                None,
                None,
                1.0,
            )
            videos.append(video_input)
    else:
        grid_key = pixel_key = images = videos = None

    conversations = copy.deepcopy(llava_to_openai(sources["conversation"], "video" in sources))

    all_input_ids = []
    all_labels = []
    all_pixel_values = []
    all_image_grid_thw = []
    all_second_gird = []

    if len(SYSTEM_MESSAGE) > 0:
        system_message = f"{DEFAULT_IM_START_TOKEN}system\n{SYSTEM_MESSAGE}{DEFAULT_IM_END_TOKEN}\n"
        system_message_input_ids = processor.tokenizer(
            system_message, add_special_tokens=False, return_tensors="pt"
        )["input_ids"]
        system_labels = torch.full_like(system_message_input_ids, IGNORE_INDEX)
        all_input_ids.append(system_message_input_ids.squeeze(0))
        all_labels.append(system_labels.squeeze(0))

    img_start = 0

    for _, j in enumerate(range(0, len(conversations), 2)):
        user_input = conversations[j]
        gpt_response = conversations[j + 1]
        user_input = f"{DEFAULT_IM_START_TOKEN}{user_input['role']}\n{user_input['content']}{DEFAULT_IM_END_TOKEN}\n{DEFAULT_IM_START_TOKEN}{gpt_response['role']}\n"

        if DEFAULT_IMAGE_TOKEN in user_input:
            img_num = user_input.count(DEFAULT_IMAGE_TOKEN)
            inputs = processor(
                text=[user_input],
                images=images[img_start : img_start + img_num] if images else None,
                videos=videos,
                padding=False,
                do_resize=False,
                return_tensors="pt",
            )
            prompt_input_ids = inputs["input_ids"]
            all_pixel_values.append(inputs[pixel_key])
            all_image_grid_thw.append(inputs[grid_key])
            img_start += img_num

        elif DEFAULT_VIDEO_TOKEN in user_input:
            inputs = processor(
                text=[user_input],
                images=images,
                videos=videos,
                padding=False,
                do_resize=False,
                return_tensors="pt",
                **video_kwargs,
            )
            all_second_gird.extend(inputs["second_per_grid_ts"])
            prompt_input_ids = inputs["input_ids"]
            all_pixel_values.append(inputs[pixel_key])
            all_image_grid_thw.append(inputs[grid_key])

        else:
            prompt_input_ids = processor.tokenizer(
                user_input, add_special_tokens=False, padding=False, return_tensors="pt"
            )["input_ids"]

        gpt_response = f"{gpt_response['content']}{DEFAULT_IM_END_TOKEN}\n"
        response_input_ids = processor(text=[gpt_response], padding=False, return_tensors="pt")[
            "input_ids"
        ]
        input_ids = torch.cat([prompt_input_ids, response_input_ids], dim=1).squeeze(0)
        all_input_ids.append(input_ids)

    input_ids = torch.cat(all_input_ids, dim=0)
    return {"seq_length": input_ids.shape[0]}


if __name__ == "__main__":
    for dataset in tqdm(datas, desc="Grouping datase"):
        data_path = osp.join(data_base, dataset)
        save_path = osp.join(save_base, dataset)
        os.makedirs(save_path, exist_ok=True)

        print(f"loaded {dataset} from {data_path}")

        ds = load_dataset(data_path)

        if "interleave" in dataset:
            cols_to_remove = []
        else:
            cols_to_remove = [
                c for c in ["action", "state", "action_is_pad"] if c in ds["train"].column_names
            ]
            print(f"removing columns: {cols_to_remove} from {dataset}: {ds['train'].column_names}")

        ds = ds.map(
            data_len_fn,
            num_proc=32,
            remove_columns=cols_to_remove,
        )

        train = ds["train"]
        print(f"gp train: {train}")

        n = len(train)
        rows_per_shard = 1024
        num_shards = (n + rows_per_shard - 1) // rows_per_shard

        for i in range(num_shards):
            shard = train.select(range(i * rows_per_shard, min((i + 1) * rows_per_shard, n)))
            shard.to_parquet(osp.join(save_path, f"train-{i:05d}-of-{num_shards:05d}.parquet"))
        print(f"saved {num_shards} shards to {save_path}")

    print("Done")
