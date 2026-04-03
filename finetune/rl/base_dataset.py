# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Cosmos-RL dataset wrapper for Alpamayo models."""

from __future__ import annotations

from typing import Any

from cosmos_rl.policy.config import Config
from cosmos_rl.utils.logging import logger
from torch.utils.data import Dataset
from transformers import AutoTokenizer

import rl.state as alp_state
from rl.prefetch.server import set_custom_cfg


class AlpamayoCosmosDataset(Dataset):
    """Lightweight dataset adapter between Alpamayo and Cosmos-RL.

    Returns only index identifiers; actual data loading is handled by
    the DataPacker. Both ReasoningVLA and ExpertModel use this class.
    """

    def __init__(self, split: str = "train"):
        self.split = split

    def setup(self, config: Config, tokenizer: AutoTokenizer, *args, **kwargs):
        """Initialize config, dataloader, and tokenizer from global state."""
        self.config = config
        set_custom_cfg(config)
        self.dataset = alp_state.get_dataloaders()[self.split].dataset
        self.tokenizer = alp_state.get_tokenizer()

    def __len__(self) -> int:
        """Return the number of samples in the underlying dataset."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> dict[str, str]:
        """Return an index-only stub; real data is loaded by the DataPacker."""
        return {"idx": str(idx), "split": self.split}

    def get_reference_answer(self, idx: int) -> dict[str, Any]:
        """Retrieve ground-truth trajectory and metadata for reward computation.

        Returns an empty dict when the sample cannot be fetched or lacks
        required trajectory keys. The reward function asserts on this and
        will fail loudly if the dataset is misconfigured.
        """
        try:
            sample = self.dataset[idx]
        except Exception as e:
            logger.error(f"[AlpamayoCosmosDataset] Error getting reference answer: {e}")
            return {}
        if not isinstance(sample, dict) or "ego_future_xyz" not in sample:
            return {}
        return {
            "ego_future_xyz": sample["ego_future_xyz"],
            "ego_future_rot": sample["ego_future_rot"],
            "ego_history_xyz": sample["ego_history_xyz"],
            "ego_history_rot": sample["ego_history_rot"],
            "egomotion_road_boundaries": sample.get("egomotion_road_boundaries", None),
            "egomotion_lanelines": sample.get("egomotion_lanelines", None),
            "ego_lwh": sample.get("ego_lwh", None),
            "ego_length_offset": sample.get("ego_length_offset", None),
            "obstacle_bbox_history": sample.get("obstacle_bbox_history", None),
            "obstacle_bbox_future": sample.get("obstacle_bbox_future", None),
            "cot": sample.get("cot", ""),
        }
