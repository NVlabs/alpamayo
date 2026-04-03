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

"""Weight mapper for ReasoningVLA model (Policy <-> Rollout key normalization)."""

from __future__ import annotations

import torch
from cosmos_rl.policy.model.hf_models.weight_mapper import HFModelWeightMapper
from cosmos_rl.utils import util
from transformers import AutoConfig


class ReasoningVLAWeightMapper(HFModelWeightMapper):
    """Weight-name mapper between ReasoningVLA policy, vLLM rollout, and HF checkpoint formats."""

    def __init__(self, hf_config: AutoConfig):
        super().__init__(hf_config.get_llm_config())

    def policy_map_local_key_to_hf_key(self, name: str) -> str:
        """Map policy-side ReasoningVLA parameter names to HF checkpoint key-space.

        Examples:
          - reasoning_vla.vlm.model.language_model.* -> model.*
          - reasoning_vla.vlm.model.visual.* -> visual.*
        """
        name = util.clear_weight_name(name)

        def apply_rewrites(name: str, rules: list[tuple[str, str]]) -> str:
            for old, new in rules:
                name = name.replace(old, new, 1)
            return name

        name = apply_rewrites(
            name,
            [
                ("reasoning_vla.", ""),
                ("vlm.", ""),
                ("model.language_model.", "model."),
                ("model.visual.", "visual."),
            ],
        )

        return super().policy_map_local_key_to_hf_key(name)

    def rollout_map_local_key_to_hf_key(self, rollout_weight_name: str) -> str:
        """Map vLLM rollout parameter names to canonical HF checkpoint key-space."""
        name = rollout_weight_name
        if name.startswith("llm.model."):
            name = name.replace("llm.model.", "model.", 1)
        elif name.startswith("llm.lm_head."):
            name = name.replace("llm.lm_head.", "lm_head.", 1)
        elif name.startswith("model.vlm.model.visual."):
            name = name.replace("model.vlm.model.visual.", "visual.", 1)
        elif name.startswith("vlm.model.visual."):
            name = name.replace("vlm.model.visual.", "visual.", 1)
        elif name.startswith("vlm."):
            name = name[len("vlm.") :]

        if name.startswith("language_model.model."):
            name = name.replace("language_model.model.", "model.", 1)
        elif name.startswith("language_model.lm_head."):
            name = name.replace("language_model.lm_head.", "lm_head.", 1)
        elif name.startswith("language_model."):
            name = name.replace("language_model.", "model.", 1)

        if name.startswith("visual.") and ".attn.qkv_proj." in name:
            name = name.replace(".attn.qkv_proj.", ".attn.qkv.", 1)

        processed_name = self.policy_map_local_key_to_hf_key(name)
        return processed_name

    def rollout_split_local_key_n_param_to_hf_key_n_param(
        self, param_name: str, param: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Rollout-side mapping/splitting with vocab-padding trimming."""
        group = super().rollout_split_local_key_n_param_to_hf_key_n_param(param_name, param)
        vocab_size = getattr(self.config, "vocab_size", None)
        if vocab_size is None:
            return group

        trimmed: list[tuple[str, torch.Tensor]] = []
        for name, t in group:
            if (
                name in ("model.embed_tokens.weight", "lm_head.weight")
                and isinstance(t, torch.Tensor)
                and t.ndim == 2
                and t.shape[0] > vocab_size
                and t.shape[1] > 0
            ):
                trimmed.append((name, t[:vocab_size]))
            else:
                trimmed.append((name, t))
        return trimmed
