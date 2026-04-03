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

import os
from typing import Any, Dict, List

import cosmos_rl.utils.distributed as dist_util
import numpy as np
import torch
from cosmos_rl.dispatcher.replica import Rollout
from cosmos_rl.policy.trainer.base import TrainerRegistry
from cosmos_rl.policy.trainer.llm_trainer.grpo_trainer import compute_loss
from cosmos_rl.utils.distributed import HighAvailabilitylNccl
from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.ulysses import slice_inputs_for_ulysses
from cosmos_rl.utils.util import compute_mfu, is_master_rank

from rl.base_trainer import AlpamayoGRPOTrainer


@TrainerRegistry.register(trainer_type="reasoning_vla_grpo")
class ReasoningVLAGRPOTrainer(AlpamayoGRPOTrainer):
    """GRPO trainer for reasoning VLA models."""

    def step_training(
        self,
        rollouts: List[Rollout],
        current_step: int,
        total_steps: int,
        remain_samples_num: int,
        inter_policy_nccl: HighAvailabilitylNccl,
        is_master_replica: bool,
        do_save_checkpoint: bool = False,
        **kwargs,
    ) -> Dict[str, Any]:
        """Customized Reasoning VLA GRPO Trainer. Run one GRPO policy optimization step from a batch of rollouts.

        Args:
            rollouts: Completed rollouts (prompt, completion, advantage, masks, etc.).
            current_step: Global training step index (logging and checkpoint naming).
            total_steps: Planned total steps for the job.
            remain_samples_num: Samples remaining in the dataset pass (checkpoint policy).
            inter_policy_nccl: Communicator for cross-replica grad / loss all-reduces.
            is_master_replica: Whether this replica performs logging and checkpoint I/O.
            do_save_checkpoint: Reserved for the Cosmos trainer API (checkpointing uses
                ``_save_checkpoint`` and config inside this implementation).

        Returns:
            ``report_data``: metrics dict for logging (e.g. ``train_step``, ``train/loss_*``,
            timing, MFU) on the master rank; empty dict elsewhere.
        """
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()
        logger.debug("[Policy] Prepare training data.")
        self.metrics = {
            "entropy": 0.0,
            "effective_entropy": 0.0,
        }
        payloads_list = [rollout.prompt for rollout in rollouts]
        completions_list = [rollout.completion for rollout in rollouts]
        advantages_list = [rollout.advantage for rollout in rollouts]

        # Optional Positive-NLL support: only compute flags when coefficient > 0
        pos_coef_global = self.config.train.train_policy.positive_nll_coef
        if pos_coef_global is not None and pos_coef_global > 0.0:
            rewards_list = [rollout.reward for rollout in rollouts]
            self._positive_flags_t = torch.tensor(
                [1 if r > 0 else 0 for r in rewards_list],
                device=self.device,
                dtype=torch.bool,
            )
        else:
            self._positive_flags_t = None
        n_ignore_prefix_tokens_list = [rollout.n_ignore_prefix_tokens for rollout in rollouts]
        processed_samples: List[Any] = [
            self.data_packer.get_policy_input(
                payloads_list[i],
                completions_list[i],
                n_ignore_prefix_tokens_list[i],
            )
            for i in range(len(payloads_list))
        ]

        advantages_t = torch.tensor(advantages_list).to(self.device)
        batch_size = len(rollouts)
        mini_batch_size = min(self.mini_batch, batch_size) if self.mini_batch > 0 else batch_size
        assert batch_size % mini_batch_size == 0, (
            "Batch size should be divided evenly by mini_batch"
        )
        num_mini_batch = batch_size // mini_batch_size

        # Initialize placeholder for old per-token logprobs
        self.old_per_token_logps = [None for _ in range(num_mini_batch)]
        self.ref_per_token_logps = [None for _ in range(num_mini_batch)]

        acc_n_tokens = 0

        need_compute_ref, kl_beta = self._swap_model_state_dict()

        loss_sum = torch.tensor(0.0, device=self.device)
        kl_loss_sum = torch.tensor(0.0, device=self.device)
        grad_norm_sum = torch.tensor(0.0, device=self.device)
        loss_count = 0
        is_computing_refs = [True, False] if need_compute_ref else [False]
        for is_computing_ref in is_computing_refs:
            # Set model to eval mode if reference model is being used
            if is_computing_ref:
                self.model.eval()
            else:
                if need_compute_ref:
                    # Swap model state dict back to the original model
                    need_compute_ref = False
                    self._swap_model_state_dict()
                self.model.train()

            with torch.set_grad_enabled(not is_computing_ref):
                for i_mu in range(1 if is_computing_ref else self.mu_iterations):
                    local_mini_step = 0
                    with torch.cuda.stream(self.train_stream):
                        for i in range(0, batch_size, mini_batch_size):
                            end = min(i + mini_batch_size, batch_size)
                            # Convert advantages from [batch_size] -> [batch_size, max_len]
                            # by expanding along the sequence dimension.

                            minibatched_processed_samples = processed_samples[i:end]

                            computed_max_len = (
                                self.config.policy.model_max_length
                                if self.parallel_dims.pp_enabled
                                else self.data_packer.policy_compute_max_len(
                                    minibatched_processed_samples
                                )
                            )

                            computed_max_len = (
                                (computed_max_len + self.seq_len_multiple - 1)
                                // self.seq_len_multiple
                                * self.seq_len_multiple
                            )
                            minibatched_advantages = (
                                advantages_t[i:end]
                                .unsqueeze(1)
                                .expand(-1, computed_max_len)
                                .to(self.device)
                            )

                            user_mini_batch: Dict[str, Any] = self.data_packer.policy_collate_fn(
                                minibatched_processed_samples,
                                computed_max_len=computed_max_len,
                            )

                            # TP/CP will shard the sequence dimension into n-ranks.
                            # The interested_tokens will be unevenly distributed across ranks.
                            # So do not enable interested_tokens in TP.
                            if (
                                self.parallel_dims.dp_shard_coord[1]
                                == self.parallel_dims.world_size
                            ):
                                user_mini_batch["interested_tokens"] = user_mini_batch[
                                    "logprob_masks"
                                ]

                            # Move all tensor to device
                            for k, v in list(user_mini_batch.items()):
                                if isinstance(v, torch.Tensor) and v.device != self.device:
                                    user_mini_batch[k] = v.to(self.device)

                            # input_ids are different across ranks in dp_shard_cp
                            position_ids, input_ids, pos_seq_dim = self.model.get_position_ids(
                                **user_mini_batch
                            )
                            acc_n_tokens += np.prod(input_ids.shape)
                            user_mini_batch["position_ids"] = position_ids
                            padding_mask = user_mini_batch.get("padding_mask", None)

                            input_ids_before_cp = user_mini_batch["input_ids"]
                            position_ids_before_cp = user_mini_batch["position_ids"]
                            padding_mask_before_cp = padding_mask

                            if self.parallel_dims.cp_enabled:
                                [input_ids, position_ids, padding_mask] = slice_inputs_for_ulysses(
                                    [input_ids, position_ids, padding_mask],
                                    self.parallel_dims.mesh["cp"],
                                )
                                user_mini_batch["position_ids"] = position_ids
                                user_mini_batch["input_ids"] = input_ids
                                if padding_mask is not None:
                                    user_mini_batch["padding_mask"] = padding_mask

                            if self.parallel_dims.pp_enabled:
                                raise NotImplementedError(
                                    "Pipeline Parallel is not supported for Reasoning VLA"
                                )
                            else:
                                model_out = self.model(**user_mini_batch)

                                if self.parallel_dims.cp_enabled:
                                    # reset the position ids and input ids
                                    user_mini_batch["position_ids"] = position_ids_before_cp
                                    user_mini_batch["input_ids"] = input_ids_before_cp
                                    if padding_mask_before_cp is not None:
                                        user_mini_batch["padding_mask"] = padding_mask_before_cp

                                # Support HF ModelOutput or raw Tensor.
                                raw_logits = (
                                    model_out.logits if hasattr(model_out, "logits") else model_out
                                )
                                if self.config.train.train_policy.temperature > 1e-6:
                                    raw_logits = (
                                        raw_logits / self.config.train.train_policy.temperature
                                    )
                                # returned shape:
                                # current_per_token_logprobs: [n_tokens_of_logprobs]
                                # cu_seqlens: [batch_size + 1]
                                current_per_token_logprobs, cu_seqlens, metrics = (
                                    self.compute_logprobs(
                                        user_mini_batch,
                                        logits=raw_logits,
                                        is_full_logits=True
                                        if getattr(raw_logits, "ndim", 0) == 3
                                        else False,
                                    )
                                )
                                logprob_masks = user_mini_batch["logprob_masks"]
                                current_advantages = logprob_masks * minibatched_advantages

                                # Compute ref per-token logprobs if needed
                                if is_computing_ref:
                                    assert i_mu == 0, "Only first iteration should compute ref"
                                    self.ref_per_token_logps[local_mini_step] = (
                                        current_per_token_logprobs.detach()
                                    )
                                    # Skip the rest of the loop
                                    local_mini_step += 1
                                    continue
                                else:
                                    if self.old_per_token_logps[local_mini_step] is None:
                                        assert i_mu == 0, (
                                            "Only first iteration should append "
                                            "`old_per_token_logps`"
                                        )
                                        self.old_per_token_logps[local_mini_step] = (
                                            current_per_token_logprobs.detach()
                                        )
                                    else:
                                        assert i_mu > 0, (
                                            "Only inner iteration should reuse "
                                            "`old_per_token_logps`"
                                        )
                                    loss, per_token_loss, kl_loss = compute_loss(
                                        current_per_token_logprobs,
                                        self.old_per_token_logps[local_mini_step],
                                        self.ref_per_token_logps[local_mini_step],
                                        current_advantages,
                                        cu_seqlens,
                                        self.config,
                                        logprob_masks,
                                        dp_group=self.parallel_dims.mesh["dp"].get_group()
                                        if self.parallel_dims.dp_enabled
                                        else None,
                                        ddp_comm=inter_policy_nccl,
                                    )

                                    # Positive Example LM Loss
                                    if pos_coef_global is not None and pos_coef_global > 0.0:
                                        pos_flag_batch = self._positive_flags_t[i:end]
                                        pos_mask = pos_flag_batch.unsqueeze(1).expand_as(
                                            logprob_masks
                                        )
                                        pos_token_mask = pos_mask & logprob_masks
                                        if pos_token_mask.any():
                                            flat_mask = pos_token_mask[logprob_masks]
                                            l_nll = -current_per_token_logprobs[flat_mask].mean()
                                            loss = loss + pos_coef_global * l_nll

                                    loss = loss / num_mini_batch
                                    per_token_loss = per_token_loss / num_mini_batch
                                    kl_loss = kl_loss / num_mini_batch

                                    loss.backward()
                                    loss_sum += per_token_loss.item()
                                    kl_loss_sum += kl_loss.item()
                                    loss_count += 1
                                    for key in metrics:
                                        self.metrics[key] += metrics[key]

                            self.mini_step += 1
                            local_mini_step += 1

                            if (
                                local_mini_step
                                % int(os.environ.get("COSMOS_GRPO_STEP_INTERVAL", "10"))
                                == 0
                            ) and local_mini_step > 1:
                                all_reduced = True
                                grad_norm_sum += self.all_reduce_states(inter_policy_nccl)
                            else:
                                all_reduced = False
                        if not is_computing_ref and not all_reduced:
                            grad_norm_sum += self.all_reduce_states(inter_policy_nccl)
        self.old_per_token_logps = []
        self.ref_per_token_logps = []
        end_event.record()

        # Only step lr scheduler when all the mini-batches are processed
        self.lr_schedulers.step()

        loss = (loss_sum / loss_count) if loss_count > 0 else loss_sum
        kl_loss = (kl_loss_sum / loss_count) if loss_count > 0 else kl_loss_sum
        if (
            self.parallel_dims.dp_replicate_enabled
            or self.parallel_dims.dp_shard_enabled
            or self.parallel_dims.cp_enabled
        ):
            global_avg_loss, global_max_loss = (  # noqa: F841
                dist_util.dist_mean(loss, self.parallel_dims.mesh["dp_cp"]),
                dist_util.dist_max(loss, self.parallel_dims.mesh["dp_cp"]),
            )
            if self.config.train.train_policy.kl_beta != 0.0:
                global_avg_kl_loss, global_max_kl_loss = (  # noqa: F841
                    dist_util.dist_mean(kl_loss, self.parallel_dims.mesh["dp_cp"]),
                    dist_util.dist_max(kl_loss, self.parallel_dims.mesh["dp_cp"]),
                )
        else:
            global_avg_loss = global_max_loss = loss.item()  # noqa: F841
            if self.config.train.train_policy.kl_beta != 0.0:
                global_avg_kl_loss = global_max_kl_loss = kl_loss.item()  # noqa: F841

        report_data = {}
        if self.config.logging.logger:
            if is_master_rank(self.parallel_dims, self.global_rank):
                report_data = {"train_step": current_step}
                # Calculate the iteration time
                assert end_event.query()
                iter_time = start_event.elapsed_time(end_event) / 1000.0  # in seconds
                report_data["train/iteration_time"] = iter_time
                report_data["train/loss_avg"] = global_avg_loss
                report_data["train/loss_max"] = global_max_loss
                report_data["train/learning_rate"] = self.lr_schedulers.get_last_lr()[0]
                if self.config.train.train_policy.kl_beta != 0.0:
                    report_data["train/kl_loss_avg"] = global_avg_kl_loss
                    report_data["train/kl_loss_max"] = global_max_kl_loss
                report_data["train/grad_norm"] = grad_norm_sum.item()

                if self.config.logging.report_mfu:
                    mfu = compute_mfu(
                        model=self.model,
                        n_tokens=acc_n_tokens,
                        iter_time=iter_time,
                        num_gpus=self.world_size,
                        dtype=self.config.train.param_dtype,
                    )
                    for k, v in mfu.items():
                        report_data[f"train/{k}"] = v
                if len(self.metrics) > 0:
                    for k, v in self.metrics.items():
                        report_data[f"train/{k}"] = (
                            v.item() if isinstance(v, torch.Tensor) else v
                        ) / loss_count
        self._save_checkpoint(current_step, total_steps, remain_samples_num, is_master_replica)
        return report_data

    @property
    def pp_loss_fn(self):
        """Return a pipeline-parallel loss function that averages per-token losses."""

        def fake_compute_loss(
            loss: torch.Tensor,
            target: torch.Tensor,
        ) -> torch.Tensor:
            """loss: the loss of shape `[n_tokens]`"""
            return loss.mean()

        return fake_compute_loss
