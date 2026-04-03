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

"""Shared data packer base class for Alpamayo RL models.

Inherits from cosmos_rl DataPacker and adds RL-specific sample fetching
with prefetch support.  Reuses the tokenizer/processor already initialized
by rl.state.init_once() (called in launcher before setup).
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Literal, cast

import rl.state as alp_state
import torch
from cosmos_rl.dispatcher.data.packer.base import DataPacker
from cosmos_rl.policy.config import Config

SampleRole = Literal["rollout", "policy"]


class BaseRLDataPacker(DataPacker):
    """Shared base for ReasoningVLA and ExpertModel RL data packers.

    Reuses the tokenizer already built by rl.state.init_once() (which runs
    before cosmos-rl calls setup()).  Subclasses implement the model-specific
    RL API (get_rollout_input, get_policy_input, policy_collate_fn, etc.).
    """

    def setup(self, config: Config, *args, **kwargs) -> None:
        """Bind the already-initialized tokenizer and configure prefetch.

        Args:
            config: CosmosConfig object containing the model configuration.
        """
        self.config = config
        self.tokenizer = alp_state.get_tokenizer()

        from rl.prefetch.server import set_custom_cfg

        set_custom_cfg(config)

    def _get_sample_raw(self, *, split: str, n: int) -> dict:
        """Fetch a raw sample dict (no per-role processing)."""
        dataloaders = alp_state.get_dataloaders()
        sample = dataloaders[split].dataset[int(n)]
        if not isinstance(sample, dict):
            raise TypeError(f"Expected Alpamayo sample to be a dict, but got {type(sample)}")
        return sample

    def _prepare_policy_sample(self, sample: dict) -> dict:
        """Normalize policy-related fields (input_ids shape)."""
        tokenized = sample.get("tokenized_data")
        if tokenized is None:
            raise KeyError("Missing tokenized_data")

        alp_tok = alp_state.get_tokenizer()
        input_ids = tokenized.get("input_ids")
        if input_ids is None:
            raw_text = tokenized.get("text")
            if raw_text is None:
                raise KeyError(
                    "Expected key 'input_ids' or 'text' inside sample['tokenized_data'] "
                    "for policy input."
                )
            input_ids = torch.tensor(alp_tok.encode(cast(str, raw_text)), dtype=torch.long)

        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.tensor(input_ids, dtype=torch.long)

        if input_ids.ndim == 1:
            input_ids = input_ids.unsqueeze(0)
        elif input_ids.ndim == 2:
            if input_ids.shape[0] != 1:
                raise ValueError(
                    f"input_ids must be shape [1, L] or [L], got {tuple(input_ids.shape)}"
                )
        else:
            raise ValueError(f"input_ids must be shape [1, L] or [L], got {tuple(input_ids.shape)}")

        tokenized["input_ids"] = input_ids
        return sample

    def _get_sample(self, *, split: str, n: int, role: SampleRole) -> Any:
        """Fetch + per-role preprocess a sample (with prefetch support)."""
        dataloaders = alp_state.get_dataloaders()
        ds = dataloaders[split].dataset
        get_pf = getattr(ds, "get_prefetched", None)
        if callable(get_pf):
            return get_pf(n=int(n), role=str(role))

        sample = self._get_sample_raw(split=split, n=int(n))
        if role == "rollout":
            return self._process_rollout_sample(sample)
        if role == "policy":
            return self._prepare_policy_sample(sample)
        raise ValueError(f"Unknown role: {role}")

    @abstractmethod
    def _process_rollout_sample(self, sample: dict) -> Any:
        """Convert raw sample to rollout format. Override in subclasses."""
