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

"""Shared checkpoint saving logic for Alpamayo GRPO trainers."""

from __future__ import annotations

import os

from cosmos_rl.utils.logging import logger
from cosmos_rl.utils.util import str2torch_dtype


def save_checkpoint_if_needed(
    trainer,
    current_step: int,
    total_steps: int,
    remain_samples_num: int,
    is_master_replica: bool,
) -> None:
    """Save Cosmos-RL checkpoint with optional HF/safetensors export.

    HF export is disabled by default to avoid HuggingFace Hub access issues.
    Set ALPAMAYO_ENABLE_HF_EXPORT=1 to enable.
    """
    enable_hf_export = os.environ.get("ALPAMAYO_ENABLE_HF_EXPORT", "0") == "1"

    if (
        (not enable_hf_export)
        and is_master_replica
        and getattr(trainer.config.train.ckpt, "export_safetensors", False)
    ):
        if not getattr(trainer, "_hf_export_disabled_warned", False):
            logger.warning(
                "[Policy] HF/safetensors export is disabled "
                "(set ALPAMAYO_ENABLE_HF_EXPORT=1 to enable). "
                "Cosmos checkpoint saving remains enabled."
            )
            trainer._hf_export_disabled_warned = True

    should_save = is_master_replica and (
        (
            trainer.config.train.ckpt.enable_checkpoint
            and current_step % trainer.config.train.ckpt.save_freq == 0
            and current_step > 0
        )
        or (trainer.config.train.ckpt.enable_checkpoint and current_step == total_steps)
    )

    if not should_save:
        return

    if enable_hf_export and getattr(trainer.config.train.ckpt, "export_safetensors", False):
        logger.info(
            "[Policy] Saving HF checkpoint at step %s to %s...",
            current_step,
            trainer.config.train.output_dir,
        )
        trainer.export_safetensors(
            output_dir=trainer.config.train.output_dir,
            rel_path=os.path.join("safetensors", f"step_{current_step}"),
            trainable_only=False,
            is_final=current_step == total_steps,
            dtype=str2torch_dtype(trainer.config.train.param_dtype),
        )

    logger.info(f"[Policy] Saving cosmos checkpoint at step {current_step}...")
    trainer.ckpt_manager.save_checkpoint(
        model=trainer.model,
        optimizer=trainer.optimizers,
        scheduler=trainer.lr_schedulers,
        step=current_step,
        total_steps=total_steps,
        **{
            "remain_samples_num": remain_samples_num,
            "is_final": current_step == total_steps,
        },
    )
    trainer.ckpt_manager.save_check(step=current_step)
