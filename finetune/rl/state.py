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

"""Global state for Alpamayo Cosmos-RL entry.

Goal:
  - Keep all heavy, process-wide resources in ONE place (dataloaders/tokenizers/configs).
  - Provide an init-once API to avoid repeated Hydra initialization / instantiation.
  - Let Dataset / Packer / Reward modules depend on this state module, instead of
    importing the entry script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import hydra
import torch
from hydra import compose, initialize
from transformers import AutoConfig

from alpamayo_r1.models.base_model import TrajectoryFusionMixin
from alpamayo_r1.processor.qwen_processor import build_processor


@dataclass
class _State:
    initialized_ckpt_path: str | None = None
    dataloaders: Any | None = None
    tokenizer: Any | None = None
    traj_tokenizer: Any | None = None
    ckpt_cfg: Any | None = None
    traj_fuser: Any | None = None


_STATE = _State()


class _RolloutTrajectoryFusion(torch.nn.Module, TrajectoryFusionMixin):
    """Lightweight helper to reuse ReasoningVLA's fuse_traj_tokens logic in rollout."""

    def __init__(self, cfg: AutoConfig, traj_tokenizer, hist_traj_tokenizer=None) -> None:
        super().__init__()
        self.config = cfg
        self.traj_tokenizer = traj_tokenizer
        self.hist_traj_tokenizer = hist_traj_tokenizer or traj_tokenizer

        self.hist_token_start_idx = self.future_token_start_idx = self.config.traj_token_start_idx
        # Need to add vocab_size for history traj tokens, because
        # Future and history traj tokens share one tokenizer; model IDs must not overlap.
        # Future occupies [traj_token_start_idx, traj_token_start_idx + vocab_size);
        # history starts immediately after that block.
        self.hist_token_start_idx += self.traj_tokenizer.vocab_size


def is_initialized() -> bool:
    return _STATE.initialized_ckpt_path is not None


def reset() -> None:
    """Reset module-global state (use with care; mainly for notebooks/tests)."""
    _STATE.initialized_ckpt_path = None
    _STATE.dataloaders = None
    _STATE.tokenizer = None
    _STATE.traj_tokenizer = None
    _STATE.ckpt_cfg = None
    _STATE.traj_fuser = None


def init_once(
    ckpt_path: str,
    *,
    hydra_config_path: str,
    hydra_config_name: str,
    overrides: list[str],
    job_name: str = "cosmos_worker",
    force: bool = False,
) -> bool:
    """Initialize global state once per process.

    Args:
      ckpt_path: Local checkpoint path.
      hydra_config_path: Hydra config search path.
      hydra_config_name: Experiment YAML to compose.
      overrides: Hydra overrides to apply (pass [] for none).
      job_name: Hydra job name.
      force: If True and already initialized, reset and rebuild.
    """
    if _STATE.initialized_ckpt_path is not None and not force:
        if _STATE.initialized_ckpt_path != ckpt_path:
            raise RuntimeError(
                "rl.state is already initialized with a different ckpt_path. "
                f"existing={_STATE.initialized_ckpt_path!r}, requested={ckpt_path!r}"
            )
        return True

    if force:
        reset()

    ov = list(overrides)

    # Build data pipeline via Hydra (reuse Reasoning-VLA configs).
    with initialize(version_base=None, config_path=hydra_config_path, job_name=job_name):
        cfg = compose(config_name=hydra_config_name, overrides=ov)

    ckpt_cfg = AutoConfig.from_pretrained(ckpt_path, trust_remote_code=True)

    dataloaders = hydra.utils.instantiate(cfg.data, _convert_="partial")

    # Build tokenizer(s) from checkpoint config.
    tokenizer = build_processor(
        vlm_name_or_path=ckpt_cfg.vlm_name_or_path,
        traj_vocab_size=ckpt_cfg.traj_vocab_size,
        min_pixels=getattr(ckpt_cfg, "min_pixels", None),
        max_pixels=getattr(ckpt_cfg, "max_pixels", None),
        include_camera_ids=getattr(ckpt_cfg, "include_camera_ids", False),
        include_frame_nums=getattr(ckpt_cfg, "include_frame_nums", False),
    ).tokenizer

    traj_tokenizer = hydra.utils.instantiate(ckpt_cfg.traj_tokenizer_cfg)

    hist_traj_tokenizer = None
    if getattr(ckpt_cfg, "hist_traj_tokenizer_cfg", None) is not None:
        try:
            hist_traj_tokenizer = hydra.utils.instantiate(ckpt_cfg.hist_traj_tokenizer_cfg)
        except Exception:
            print("[WARN] Failed to instantiate history trajectory tokenizer")
            hist_traj_tokenizer = None

    traj_fuser = _RolloutTrajectoryFusion(
        cfg=ckpt_cfg,
        traj_tokenizer=traj_tokenizer,
        hist_traj_tokenizer=hist_traj_tokenizer,
    )

    _STATE.initialized_ckpt_path = ckpt_path
    _STATE.ckpt_cfg = ckpt_cfg
    _STATE.dataloaders = dataloaders
    _STATE.tokenizer = tokenizer
    _STATE.traj_tokenizer = traj_tokenizer
    _STATE.traj_fuser = traj_fuser
    # Apply node-prefetch wrapping only for training; validation reads directly from the
    # underlying dataset to avoid prefetch-server issues during eval.
    if "train" in (dataloaders if isinstance(dataloaders, dict) else {}):
        maybe_enable_node_prefetch(split="train")
    return True


def _require(name: str, value: Any):
    if value is None:
        raise RuntimeError(
            f"[rl.state] '{name}' is not initialized. Call state.init_once(...) first."
        )
    return value


def get_ckpt_cfg() -> Any:
    return _require("ckpt_cfg", _STATE.ckpt_cfg)


def get_dataloaders() -> Any:
    return _require("dataloaders", _STATE.dataloaders)


def get_tokenizer() -> Any:
    return _require("tokenizer", _STATE.tokenizer)


def get_traj_tokenizer() -> Any:
    return _require("traj_tokenizer", _STATE.traj_tokenizer)


def get_traj_fuser() -> Any:
    return _require("traj_fuser", _STATE.traj_fuser)


def maybe_enable_node_prefetch(*, split: str) -> None:
    """Wrap the underlying dataloader dataset so dataset[n] is served by the node prefetch server.

    This must be called after `data_prefetch._alpamayo_set_custom_cfg(config)` has been invoked
    somewhere in the process, so `_alpamayo_cfg_get("prefetch.capacity", ...)` can read TOML.
    """
    # Local imports to avoid hard dependency / circular imports at module import time.
    from rl.prefetch.dataset import NodePrefetchDatasetWrapper
    from rl.prefetch.server import _alpamayo_cfg_get

    cap = int(_alpamayo_cfg_get("prefetch.capacity", 0) or 0)
    if cap <= 0:
        return

    dls = get_dataloaders()
    loader = dls[str(split)]
    base = getattr(loader, "dataset", None)
    if base is None or isinstance(base, NodePrefetchDatasetWrapper):
        return

    wrapped = NodePrefetchDatasetWrapper(base, server_key=str(split))
    try:
        loader.dataset = wrapped
    except ValueError:
        # torch>=2.8 forbids mutating DataLoader.dataset after init: replace the loader instead.
        import copy

        new_loader = copy.copy(loader)
        object.__setattr__(new_loader, "dataset", wrapped)
        dls[str(split)] = new_loader
