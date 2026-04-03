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

"""Shared DTensor weight loading utilities for FSDP2 models."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def detect_fsdp2_active(module: torch.nn.Module) -> bool:
    """Return True if any parameter or buffer in *module* is a DTensor."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return False

    for _, p in module.named_parameters(recurse=True):
        t = getattr(p, "data", p)
        if isinstance(t, DTensor):
            return True
    for _, b in module.named_buffers(recurse=True):
        if isinstance(b, DTensor):
            return True
    return False


def copy_state_into_dtensor_shards(
    target_module: torch.nn.Module,
    ckpt_state: dict[str, torch.Tensor],
    *,
    strict: bool = True,
    pad_to_match: bool = False,
    key_prefix: str = "",
) -> None:
    """Copy a CPU state dict into FSDP2 DTensor shards in-place.

    Args:
        target_module: The nn.Module whose params/buffers are DTensors.
        ckpt_state: Checkpoint state dict (CPU tensors).
        strict: If True, raise KeyError when a parameter is missing from ckpt_state.
        pad_to_match: If True, zero-pad source tensors that are smaller than the
            target DTensor (e.g. action_out_proj after pad_linear_for_fsdp).
        key_prefix: Optional prefix to prepend when looking up keys in ckpt_state
            (e.g. "expert_model." for trainer checkpoint resume).
    """
    try:
        from torch.distributed.tensor import DTensor, distribute_tensor
    except ImportError:
        raise RuntimeError("FSDP2/DTensor appears active, but DTensor APIs are unavailable")

    with torch.no_grad():
        for name, p in target_module.named_parameters(recurse=True):
            full_key = f"{key_prefix}{name}" if key_prefix else name
            if full_key not in ckpt_state:
                if strict:
                    raise KeyError(f"Checkpoint missing parameter: {full_key}")
                continue
            src = ckpt_state[full_key]
            t = getattr(p, "data", p)
            if isinstance(t, DTensor):
                if pad_to_match and src.shape != t.shape:
                    pad_args: list[int] = []
                    for s_dim, t_dim in zip(reversed(src.shape), reversed(t.shape)):
                        pad_args.extend([0, t_dim - s_dim])
                    src = F.pad(src, pad_args)
                src_dt = distribute_tensor(src, t.device_mesh, t.placements)
                t.to_local().copy_(src_dt.to_local())
            else:
                t.copy_(src.to(t.device))

        for name, b in target_module.named_buffers(recurse=True):
            full_key = f"{key_prefix}{name}" if key_prefix else name
            if full_key not in ckpt_state:
                continue
            src = ckpt_state[full_key]
            if isinstance(b, DTensor):
                src_dt = distribute_tensor(src, b.device_mesh, b.placements)
                b.to_local().copy_(src_dt.to_local())
            elif isinstance(b, torch.Tensor):
                b.copy_(src.to(b.device))
