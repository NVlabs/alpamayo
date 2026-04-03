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

"""Shared GRPO trainer base class for Alpamayo models."""

from __future__ import annotations

from typing import Any

import torch
from cosmos_rl.dispatcher.replica import Rollout
from cosmos_rl.policy.trainer import GRPOTrainer

from rl.utils.checkpoint import save_checkpoint_if_needed


class AlpamayoGRPOTrainer(GRPOTrainer):
    """Shared base for ExpertModel and ReasoningVLA GRPO trainers.

    Consolidates: rollout extraction, checkpoint saving, _move_to_device,
    and swap-based reference model phase setup.
    """

    def _extract_rollout_data(self, rollouts: list[Rollout]) -> tuple[list, list, list, list]:
        """Unpack Cosmos rollouts into parallel lists for the policy update.

        Args:
            rollouts: Batch of completed rollouts from the dispatcher.

        Returns:
            Tuple ``(payloads, completions, advantages, n_ignore_prefix_tokens)``, where each
            element is a list aligned by rollout index (same order as ``rollouts``).
        """
        payloads = [r.prompt for r in rollouts]
        completions = [r.completion for r in rollouts]
        advantages = [r.advantage for r in rollouts]
        n_ignore = [r.n_ignore_prefix_tokens for r in rollouts]
        return payloads, completions, advantages, n_ignore

    def _process_samples(self, payloads: list, completions: list, n_ignore: list) -> list[Any]:
        """Turn raw prompts/completions into policy training inputs via the data packer.

        Args:
            payloads: Per-rollout prompt payloads (model-specific structure).
            completions: Per-rollout generated completions.
            n_ignore: Per-rollout prefix token counts to mask in the loss.

        Returns:
            One packed policy input per rollout, in the same order as the input lists.
        """
        return [
            self.data_packer.get_policy_input(payloads[i], completions[i], n_ignore[i])
            for i in range(len(payloads))
        ]

    def _save_checkpoint(
        self,
        current_step: int,
        total_steps: int,
        remain_samples_num: int,
        is_master_replica: bool,
    ) -> None:
        """Persist a checkpoint when the configured save policy says so.

        Args:
            current_step: Optimizer/training step index.
            total_steps: Planned total steps for the run (used for scheduling).
            remain_samples_num: Samples left in the current dataset pass, if applicable.
            is_master_replica: Whether this replica should perform I/O (only master writes).
        """
        save_checkpoint_if_needed(
            self,
            current_step,
            total_steps,
            remain_samples_num,
            is_master_replica,
        )

    def _move_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Recursively move tensors (and tensor lists) in a batch to ``self.device``.

        Nested dicts are walked depth-first. Non-tensor leaves are left unchanged.

        Args:
            batch: Possibly nested structure of tensors and other objects.

        Returns:
            Same structure as ``batch`` with tensors moved to the trainer device.
        """
        result = {}
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                result[k] = v.to(self.device)
            elif isinstance(v, dict):
                result[k] = self._move_to_device(v)
            elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
                result[k] = [t.to(self.device) for t in v]
            else:
                result[k] = v
        return result

    @torch.no_grad()
    def _swap_model_state_dict(self) -> tuple[bool, float]:
        """Swap policy and reference weights in-place for the KL reference phase.

        When ``kl_beta`` is non-zero and a reference state dict exists, exchanges each
        overlapping parameter tensor between ``self.model`` and ``self.reference_state_dict``
        on ``self.train_stream`` (used so the next forward uses the reference weights).

        Returns:
            ``(need_compute_ref, kl_beta)``: whether reference logits should be computed
            next, and the configured KL coefficient (or ``0.0`` when swapping is skipped).
        """
        kl_beta = getattr(self.config.train.train_policy, "kl_beta", 0.0)
        if kl_beta != 0.0 and self.reference_state_dict:
            with torch.cuda.stream(self.train_stream):
                model_sd = self.model.state_dict()
                for key, value in model_sd.items():
                    if key not in self.reference_state_dict:
                        continue
                    ref_clone = self.reference_state_dict[key].clone()
                    self.reference_state_dict[key].copy_(value)
                    value.copy_(ref_clone)
            return True, kl_beta
        return False, 0.0

    def _reference_reset(self, current_step: int) -> None:
        """Initialize or refresh the KL reference weights on CPU.

        No-op when ``kl_beta`` is zero. If ``reference_state_dict`` is empty, snapshots
        the current ``self.model`` weights (detached CPU tensors). If
        ``reference_reset_interval`` is set and positive, copies the live model into the
        reference dict every ``reference_reset_interval`` steps (including alignment on
        ``current_step``).

        Args:
            current_step: Training step index used for first-time init and interval checks.
        """
        kl_beta = getattr(self.config.train.train_policy, "kl_beta", 0.0)
        if kl_beta == 0.0:
            return

        reset_interval = getattr(self.config.train.train_policy, "reference_reset_interval", None)

        if not self.reference_state_dict:
            from cosmos_rl.utils.logging import logger

            logger.info(f"[AlpamayoGRPO] Initializing reference state dict at step {current_step}")
            for key, value in self.model.state_dict().items():
                self.reference_state_dict[key] = value.detach().cpu()
            return

        if reset_interval and reset_interval > 0 and current_step % reset_interval == 0:
            from cosmos_rl.utils.logging import logger

            logger.info(f"[AlpamayoGRPO] Resetting reference model at step {current_step}")
            for key, value in self.model.state_dict().items():
                if key in self.reference_state_dict:
                    self.reference_state_dict[key] = value.detach().cpu()
