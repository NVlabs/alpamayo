# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""End-to-end example script for the inference pipeline.

Loads a dataset, runs inference, and computes the minADE. With no arguments
it reproduces the prior hardcoded behavior exactly; flags exist so users can
vary the clip, sampling parameters, model name, seed, etc. without editing
this file.

Examples:
    # Default behavior (matches the pre-CLI script).
    python src/alpamayo_r1/test_inference.py

    # Different clip with 8 samples.
    python src/alpamayo_r1/test_inference.py \\
        --clip-id 030c760c-ae38-49aa-9ad8-f5650a545d26 --num-traj-samples 8

    # Hotter sampling at a different keyframe.
    python src/alpamayo_r1/test_inference.py --t0-us 6000000 --temperature 0.9
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

from alpamayo_r1 import helper
from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1


# Defaults preserve the pre-CLI hardcoded behavior.
DEFAULT_CLIP_ID = "030c760c-ae38-49aa-9ad8-f5650a545d26"
DEFAULT_T0_US = 5_100_000
DEFAULT_MODEL = "nvidia/Alpamayo-R1-10B"
DEFAULT_NUM_TRAJ_SAMPLES = 1
DEFAULT_TOP_P = 0.98
DEFAULT_TEMPERATURE = 0.6
DEFAULT_MAX_GENERATION_LENGTH = 256
DEFAULT_SEED = 42

DTYPE_MAP: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments. Defaults reproduce the prior hardcoded behavior."""
    parser = argparse.ArgumentParser(
        description=(
            "Run Alpamayo end-to-end inference on a single PAI clip and report minADE."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--clip-id",
        type=str,
        default=DEFAULT_CLIP_ID,
        help="PAI clip id to load.",
    )
    parser.add_argument(
        "--t0-us",
        type=int,
        default=DEFAULT_T0_US,
        help="Keyframe timestamp in microseconds.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help="HuggingFace model id (or local path) for AlpamayoR1.from_pretrained.",
    )
    parser.add_argument(
        "--num-traj-samples",
        type=int,
        default=DEFAULT_NUM_TRAJ_SAMPLES,
        help="Number of trajectory samples per CoC rollout. Higher = more variety + memory.",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=DEFAULT_TOP_P,
        help="Nucleus sampling probability.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--max-generation-length",
        type=int,
        default=DEFAULT_MAX_GENERATION_LENGTH,
        help="Maximum number of generated tokens for the chain-of-causation rollout.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="CUDA RNG seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for model + inputs (e.g. cuda, cuda:1).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=sorted(DTYPE_MAP.keys()),
        default="bfloat16",
        help="Model dtype + autocast dtype.",
    )
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    """Run inference on a single clip and print minADE."""
    dtype = DTYPE_MAP[args.dtype]

    print(f"Loading dataset for clip_id: {args.clip_id}...")
    data = load_physical_aiavdataset(args.clip_id, t0_us=args.t0_us)
    print("Dataset loaded.")

    messages = helper.create_message(data["image_frames"].flatten(0, 1))

    model = AlpamayoR1.from_pretrained(args.model, dtype=dtype).to(args.device)
    processor = helper.get_processor(model.tokenizer)

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, args.device)

    torch.cuda.manual_seed_all(args.seed)
    with torch.autocast(args.device.split(":")[0], dtype=dtype):
        pred_xyz, pred_rot, extra = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=args.top_p,
            temperature=args.temperature,
            num_traj_samples=args.num_traj_samples,
            max_generation_length=args.max_generation_length,
            return_extra=True,
        )

    # the size is [batch_size, num_traj_sets, num_traj_samples]
    print("Chain-of-Causation (per trajectory):\n", extra["cot"][0])

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].T.numpy()
    pred_xy = pred_xyz.cpu().numpy()[0, 0, :, :, :2].transpose(0, 2, 1)
    diff = np.linalg.norm(pred_xy - gt_xy[None, ...], axis=1).mean(-1)
    min_ade = diff.min()
    print("minADE:", min_ade, "meters")
    print(
        "Note: VLA-reasoning models produce nondeterministic outputs due to trajectory sampling, "
        "hardware differences, etc. With num_traj_samples=1 (set for GPU memory compatibility), "
        "variance in minADE is expected. For visual sanity checks, see notebooks/inference.ipynb"
    )


if __name__ == "__main__":
    main(parse_args())
