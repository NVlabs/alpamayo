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

"""Cosmos-RL BaseModel wrapper for ReasoningVLA."""

from __future__ import annotations

import torch
from transformers import AutoConfig

from rl.base_cosmos_wrapper import BaseCosmosWrapper
from rl.utils.fsdp import shard_lm_layers, shard_visual_tower
from rl.utils.weight_loading import copy_state_into_dtensor_shards, detect_fsdp2_active
from rl.models.reasoning_vla.base_model import RLWrapperReasoningVLA


class RVLACosmos(BaseCosmosWrapper):
    """Cosmos BaseModel wrapper around RLWrapperReasoningVLA (FSDP2 sharding + input bridging)."""

    def __init__(self, hf_config: AutoConfig):
        super().__init__(hf_config)
        self.reasoning_vla = RLWrapperReasoningVLA(hf_config)

    @staticmethod
    def supported_model_types():
        """Return the HF model types this wrapper supports."""
        return ["alpamayo_reasoning_vla"]

    def _apply_fsdp2(self, dp_mesh, fsdp_config: dict, reshard_fn) -> None:
        """Shard visual tower, LM layers, and the top-level module with FSDP2."""
        from torch.distributed.fsdp import fully_shard

        rvla = self.reasoning_vla
        shard_visual_tower(rvla, fsdp_config, reshard_fn, model_name="ReasoningVLA")
        shard_lm_layers(rvla, fsdp_config, reshard_fn, model_name="ReasoningVLA")
        fully_shard(self, **fsdp_config, reshard_after_forward=True)

    def get_position_ids(self, **kwargs) -> tuple[torch.Tensor, torch.Tensor, int]:
        """Compute sequential position IDs from input_ids or inputs_embeds."""
        if "input_ids" in kwargs and kwargs["input_ids"] is not None:
            inputs = kwargs["input_ids"]
        else:
            inputs_embeds = kwargs.get("inputs_embeds", None)
            if inputs_embeds is None:
                raise ValueError("Either input_ids or inputs_embeds must be provided")
            seq_len = inputs_embeds.size(1)
            batch = inputs_embeds.size(0)
            inputs = torch.zeros(batch, seq_len, dtype=torch.long, device=inputs_embeds.device)

        position_ids = (
            torch.arange(inputs.size(-1), dtype=torch.long, device=inputs.device)
            .unsqueeze(0)
            .expand_as(inputs)
        )
        seq_dim_idx = 1
        return position_ids, inputs, seq_dim_idx

    def post_to_empty_hook(self, cosmos_config):
        """No-op: load_hf_weights handles all weight/buffer restoration via from_pretrained."""

    def load_hf_weights(
        self,
        model_name_or_path: str,
        parallel_dims=None,
        device: torch.device = None,
        revision: str | None = None,
    ):
        """Load checkpoint weights into self.reasoning_vla."""
        ckpt_model = RLWrapperReasoningVLA.from_pretrained(
            model_name_or_path, trust_remote_code=True
        ).to("cpu")
        ckpt_state = ckpt_model.state_dict()
        del ckpt_model

        if not detect_fsdp2_active(self.reasoning_vla):
            self.reasoning_vla.load_state_dict(ckpt_state, strict=True)
            if device is not None:
                self.reasoning_vla = self.reasoning_vla.to(device)
            return

        copy_state_into_dtensor_shards(self.reasoning_vla, ckpt_state, strict=True)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        **kwargs,
    ):
        """Bridge Cosmos-RL inputs to the underlying ReasoningVLA forward pass."""
        td = kwargs.get("tokenized_data", None)
        if isinstance(td, dict):
            if labels_mask is not None:
                td["labels_mask"] = labels_mask
            if position_ids is not None:
                td["position_ids"] = position_ids
            if input_ids is not None:
                td["input_ids"] = input_ids

            target_device = (
                input_ids.device if input_ids is not None else next(self.parameters()).device
            )
            try:
                target_dtype = next(self.reasoning_vla.vlm.parameters()).dtype
            except StopIteration:
                target_dtype = None

            for k, v in list(td.items()):
                if isinstance(v, torch.Tensor):
                    if torch.is_floating_point(v) and target_dtype is not None:
                        td[k] = v.to(device=target_device, dtype=target_dtype)
                    else:
                        td[k] = v.to(device=target_device)

            kwargs["tokenized_data"] = td

        outputs = self.reasoning_vla(
            tokenized_data=kwargs.get("tokenized_data", {}),
            ego_history_xyz=kwargs.get("ego_history_xyz", None),
            ego_history_rot=kwargs.get("ego_history_rot", None),
            ego_future_xyz=kwargs.get("ego_future_xyz", None),
            ego_future_rot=kwargs.get("ego_future_rot", None),
            labels_mask=labels_mask,
        )
        return outputs

    @classmethod
    def from_pretrained(
        cls,
        hf_config: AutoConfig,
        model_name_or_path: str,
        max_position_embeddings: int | None = None,
    ) -> RVLACosmos:
        """Construct an RVLACosmos instance from a pretrained checkpoint path."""
        checkpoint_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        if max_position_embeddings is not None:
            checkpoint_config.model_max_length = max_position_embeddings
        checkpoint_config._name_or_path = model_name_or_path
        return cls(checkpoint_config)
