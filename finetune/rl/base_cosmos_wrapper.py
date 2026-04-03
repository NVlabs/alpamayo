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

"""BaseCosmosWrapper -- common Cosmos-RL BaseModel for both ReasoningVLA and ExpertModel."""

from __future__ import annotations

from typing import Callable

import torch
from cosmos_rl.policy.model.base import BaseModel
from cosmos_rl.utils.logging import logger
from transformers import AutoConfig


class BaseCosmosWrapper(BaseModel):
    """Shared base for Alpamayo Cosmos model wrappers.

    Provides the common ``parallelize_fn`` template (precondition checks,
    DDP fallback, dp_mesh construction) and delegates model-specific FSDP2
    sharding to the ``_apply_fsdp2`` hook.
    """

    def __init__(self, hf_config: AutoConfig):
        super().__init__(hf_config)
        self.hf_config = hf_config

    @staticmethod
    def supported_model_types():
        """Return the HF model types this wrapper supports."""
        raise NotImplementedError

    @property
    def parallelize_fn(self):
        """Build the FSDP2/DDP parallelization closure and return ``(fn, model)``."""
        from torch.distributed._composable.replicate import replicate

        from rl.utils.fsdp import (
            build_fsdp_config,
            build_reshard_fn,
            check_parallelism_preconditions,
            get_dp_mesh,
        )

        model_name = type(self).__name__

        def parallelize(
            model: torch.nn.Module,
            parallel_dims,
            config,
            pp_loss_fn: Callable | None = None,
        ):
            check_parallelism_preconditions(parallel_dims, config, model_name)

            if not parallel_dims.dp_shard_enabled:
                if parallel_dims.dp_replicate_enabled:
                    replicate(model, device_mesh=parallel_dims.mesh, bucket_cap_mb=100)
                    logger.info(f"[{model_name}] Applied DDP (replicate)")
                return None, None

            dp_mesh = get_dp_mesh(parallel_dims)
            fsdp_config = build_fsdp_config(dp_mesh, config)
            reshard_fn = build_reshard_fn(config.train.fsdp_reshard_after_forward)

            model._apply_fsdp2(dp_mesh, fsdp_config, reshard_fn)

            if parallel_dims.dp_replicate_enabled:
                logger.info(f"[{model_name}] Applied HSDP(FSDP2)")
            else:
                logger.info(f"[{model_name}] Applied FSDP2")
            if config.train.fsdp_offload:
                logger.info(f"[{model_name}] Applied CPU offloading")

            return None, None

        return parallelize, self

    def _apply_fsdp2(self, dp_mesh, fsdp_config: dict, reshard_fn) -> None:
        """Apply FSDP2 sharding to the model. Subclasses must override."""
        raise NotImplementedError("Subclasses must implement _apply_fsdp2")

    def get_position_ids(self, **kwargs) -> tuple[torch.Tensor | None, torch.Tensor, int]:
        """Default position_ids implementation for Alpamayo models."""
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

    def apply_pipeline_split(self, pp_rank, pp_size):
        """Apply pipeline-parallel split (not supported)."""
        raise NotImplementedError("Pipeline parallel is not supported.")

    def separate_model_parts(self) -> list[torch.nn.Module]:
        """Return model sub-modules for per-part parallelization."""
        return [self]

    @classmethod
    def get_nparams_and_flops(cls, seq_len: int) -> tuple[int, int]:
        """Return estimated parameter count and FLOPs for the given sequence length."""
        return 0, 0
