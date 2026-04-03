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


"""Wrapped Reasoning VLA model implementation for RL training."""

from dataclasses import dataclass
from typing import Any

import einops
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModel,
)
from transformers.utils import ModelOutput

from alpamayo_r1.common import logging
from alpamayo_r1.models.base_model import ReasoningVLA, IGNORE_INDEX
from alpamayo_r1.models.token_utils import extract_text_tokens, extract_traj_tokens
from rl.models.reasoning_vla.config import RLWrapperReasoningVLAConfig

logger = logging.RankedLogger(__name__, rank_zero_only=True)
logger.setLevel("INFO")


@dataclass
class ReasoningVLAOutput(ModelOutput):
    """Output of the ReasoningVLA model."""

    loss: torch.FloatTensor | None = None
    logits: torch.FloatTensor | None = None


class RLWrapperReasoningVLA(ReasoningVLA):
    """RL Wrapped ReasoningVLA model."""

    config_class: type[RLWrapperReasoningVLAConfig] = RLWrapperReasoningVLAConfig

    def __init__(
        self,
        config: RLWrapperReasoningVLAConfig,
        pretrained_modules: dict[str, torch.nn.Module] | None = None,
        original_vocab_size: int | None = None,
        print_param_count: bool = True,
    ) -> None:
        """Initialize the model."""
        super().__init__(config)

    def gradient_checkpointing_enable(
        self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
    ) -> None:
        """Enable gradient checkpointing for the model.

        Args:
            gradient_checkpointing_kwargs: Additional keyword arguments for gradient checkpointing.
        """
        if hasattr(self.vlm, "gradient_checkpointing_enable"):
            self.vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            raise ValueError(
                f"{self.vlm.__class__.__name__} does not support gradient checkpointing."
            )

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing for the model."""
        if hasattr(self.vlm, "gradient_checkpointing_disable"):
            self.vlm.gradient_checkpointing_disable()
        else:
            raise ValueError(
                f"{self.vlm.__class__.__name__} does not support gradient checkpointing."
            )

    def freeze_base_model_except_embeddings(self) -> None:
        """Only train the embeddings for new tokens."""
        for param in self.parameters():
            param.requires_grad = False

        self.vlm.language_model.embed_tokens.weight.requires_grad = True

        def reset_grad(grad: torch.Tensor) -> torch.Tensor:
            grad[: self.original_vocab_size] = 0
            return grad

        self.vlm.language_model.embed_tokens.weight.register_hook(reset_grad)

    @torch._dynamo.disable
    def _compute_next_token_loss(
        self,
        outputs: ModelOutput,
        labels: torch.Tensor,
        labels_mask: torch.Tensor | None = None,
        token_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the loss for the next token prediction.

        Args:
            outputs: ModelOutput containing logits of shape (B, L, V)
            labels: [B, L]
            labels_mask: [B, L], indicates which tokens in the sequence are valid for loss
                computation
            token_mask: [V], indicates which tokens ids are valid for logits computation

        Returns:
            torch.Tensor: (,) loss value
        """
        if labels_mask is None:
            labels_mask = torch.ones_like(labels, dtype=torch.bool)
        if labels_mask[:, 1:].sum() == 0:
            return torch.tensor(0.0, device=labels.device)
        # Shift labels to the left by 1 position (predict next token)
        shift_labels = labels[..., 1:]
        # The logits should also be trimmed to match the shifted labels
        # NOTE: we clone the logits to avoid in-place operations if token_mask is present that will
        # modify the original
        shift_logits = outputs.logits[..., :-1, :].clone()

        shift_labels = shift_labels[labels_mask[:, 1:]].contiguous()
        shift_logits = shift_logits[labels_mask[:, 1:]].contiguous().float()
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        if token_mask is not None:
            shift_logits[..., ~token_mask] = torch.finfo(shift_logits.dtype).min
        loss = torch.nan_to_num(
            torch.nn.functional.cross_entropy(
                shift_logits, shift_labels, ignore_index=IGNORE_INDEX, reduction="mean"
            ),
            nan=0.0,
        )
        return loss

    def forward(
        self,
        tokenized_data: dict[str, Any],
        ego_history_xyz: torch.Tensor | None = None,
        ego_history_rot: torch.Tensor | None = None,
        ego_future_xyz: torch.Tensor | None = None,
        ego_future_rot: torch.Tensor | None = None,
        labels_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> ReasoningVLAOutput:
        """Forward pass of the model."""
        # 1. tokenize trajectory and fuse into input_ids
        input_ids = tokenized_data.pop("input_ids")
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
            "ego_future_xyz": ego_future_xyz,
            "ego_future_rot": ego_future_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)

        # 2. get labels
        labels = input_ids.clone()
        if labels_mask is not None:
            labels = torch.where(labels_mask, labels, IGNORE_INDEX)

        # 3. vlm forward pass
        outputs = self.vlm(input_ids=input_ids, labels=labels, **tokenized_data)

        losses = {}
        # Identify trajectory tokens (tokens between traj_future and next special token)
        traj_mask = (
            (
                (labels >= self.future_token_start_idx)
                & (labels < self.future_token_start_idx + self.config.traj_vocab_size)
            )
            | (labels == self.special_token_ids["traj_future_start"])
            | (labels == self.special_token_ids["traj_future_end"])
        )
        losses["future_traj"] = self._compute_next_token_loss(
            outputs, labels, traj_mask
        ) * self.config.loss_weights.get("future_traj", 1.0)
        labels[traj_mask] = IGNORE_INDEX

        # Include all other tokens in the loss
        losses["others"] = self._compute_next_token_loss(
            outputs, labels, labels != IGNORE_INDEX
        ) * self.config.loss_weights.get("others", 1.0)

        # Replace the original loss
        outputs.loss = sum(losses.values())

        return ReasoningVLAOutput(
            loss=outputs.loss,
            logits=outputs.logits,
        )

    def sample_trajectories_from_data(
        self,
        data: dict[str, Any],
        top_p: float = 0.98,
        top_k: int | None = None,
        temperature: float = 0.6,
        num_traj_samples: int = 6,
        num_traj_sets: int = 1,
        last_component: str = "traj_future",
        *args: Any,
        **kwargs: Any,
    ) -> (
        tuple[torch.Tensor, torch.Tensor, torch.Tensor] | tuple[Any, Any, Any, dict[str, list[str]]]
    ):
        """Sample trajectories from the data.

        Args:
            data: The input data.
            top_p: The top-p value for sampling.
            top_k: The top-k value for sampling.
            temperature: The temperature for sampling.
            num_traj_samples: The number of trajectory samples.
            num_traj_sets: The number of trajectory sets.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            pred_xyz: The predicted xyz.
            pred_rot: The predicted rotation.
            logprob: The log probability.
        """
        ego_history_xyz = data["ego_history_xyz"]
        ego_history_rot = data["ego_history_rot"]
        B, n_traj_group, _, _ = ego_history_xyz.shape
        assert n_traj_group == 1, "Only one trajectory group is supported for inference."
        tokenized_data = data["tokenized_data"]
        input_ids = tokenized_data.pop("input_ids")
        traj_data = {
            "ego_history_xyz": ego_history_xyz,
            "ego_history_rot": ego_history_rot,
        }
        input_ids = self.fuse_traj_tokens(input_ids, traj_data)

        n_samples_total = num_traj_samples * num_traj_sets
        max_generation_length = kwargs.get(
            "max_generation_length", self.config.tokens_per_future_traj
        )
        assert max_generation_length >= self.config.tokens_per_future_traj
        generation_config = self.vlm.generation_config
        generation_config.top_p = top_p
        generation_config.temperature = temperature
        generation_config.do_sample = True
        generation_config.num_return_sequences = n_samples_total
        generation_config.max_new_tokens = max_generation_length
        generation_config.output_logits = True
        generation_config.return_dict_in_generate = True
        generation_config.top_k = top_k
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generated = self.vlm.generate(
            input_ids=input_ids, **tokenized_data, generation_config=generation_config
        )
        # remove input ids from the generated sequences
        generated_tokens = generated.sequences[:, input_ids.shape[1] :]

        # extract trajectory tokens from generated sequences
        traj_token_ids = extract_traj_tokens(
            generated_tokens,
            self.special_token_ids,
            self.config.tokens_per_future_traj,
            self.future_token_start_idx,
            self.traj_tokenizer.vocab_size,
        )

        pred_xyz, pred_rot, _ = self.traj_tokenizer.decode(
            hist_xyz=einops.repeat(
                ego_history_xyz[:, -1],
                "b ... -> (b n) ...",
                n=n_samples_total,
            ),
            hist_rot=einops.repeat(
                ego_history_rot[:, -1],
                "b ... -> (b n) ...",
                n=n_samples_total,
            ),
            tokens=traj_token_ids,
        )
        pred_xyz = einops.rearrange(
            pred_xyz,
            "(b ns nj) ... -> b ns nj ...",
            ns=num_traj_sets,
            nj=num_traj_samples,
        )
        pred_rot = einops.rearrange(
            pred_rot,
            "(b ns nj) ... -> b ns nj ...",
            ns=num_traj_sets,
            nj=num_traj_samples,
        )
        logger.warning(
            "logprob computation is not implemented; returning zeros. "
            "Do not use these values for ranking or importance weighting."
        )
        logprob = torch.zeros_like(pred_xyz[..., 0])

        # return additional information
        if kwargs.get("return_extra", False):
            extra = extract_text_tokens(self.tokenizer, generated_tokens)
            # rearrange text tokens to shape [B, ns, nj] to match trajectory shape
            for text_tokens in extra.keys():
                extra[text_tokens] = np.array(extra[text_tokens]).reshape(
                    [input_ids.shape[0], num_traj_sets, num_traj_samples]
                )
            return pred_xyz, pred_rot, logprob, extra
        return pred_xyz, pred_rot, logprob


# Register the model with Auto classes
AutoConfig.register("alpamayo_reasoning_vla", RLWrapperReasoningVLAConfig)
AutoModel.register(RLWrapperReasoningVLAConfig, RLWrapperReasoningVLA)
