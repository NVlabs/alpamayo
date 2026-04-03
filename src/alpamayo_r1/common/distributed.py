# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Helping functions for distributed training."""

import functools
import os
from datetime import timedelta
from typing import Callable

import torch
import torch.distributed as dist

from alpamayo_r1.common import logging

logger = logging.RankedLogger(__name__, rank_zero_only=False)
logger.setLevel("INFO")


def initialize_distributed_simple():
    """Initialize distributed training with simple configuration."""
    torch.distributed.init_process_group(
        backend="nccl", init_method="env://", timeout=timedelta(minutes=60)
    )
    # necessary here because we initialize global process group by ourselves
    torch.cuda.set_device(torch.device(f"cuda:{get_local_rank()}"))


def barrier() -> None:
    """Barrier for all GPUs."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def get_local_rank() -> int:
    """Get the rank (GPU device) of the worker locally on the node.

    Returns:
        rank (int): The local rank of the worker.
    """
    local_rank = 0
    if dist.is_available() and dist.is_initialized() and "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
    return local_rank


def get_global_rank() -> int:
    """Get the rank (GPU device) of the worker.

    Returns:
        rank (int): The rank of the worker.
    """
    rank = 0
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    return rank


def get_world_size() -> int:
    """Get world size. How many GPUs are available in this job.

    Returns:
        world_size (int): The total number of GPUs available in this job.
    """
    world_size = 1
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    return world_size


def is_rank_zero() -> bool:
    """Check if current process is the rank zero process.

    Returns:
        (bool): True if this function is called from the rank zero process, else False.
    """
    return get_global_rank() == 0


def rank_zero_only(func: Callable) -> Callable:
    """Apply this function only on the rank zero process.

    Example usage:
        @rank_zero_only
        def func(x):
            ...

    Args:
        func (Callable): a function.

    Returns:
        (Callable): A function wrapper executing the function only on the rank zero process.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        if is_rank_zero():
            return func(*args, **kwargs)
        else:
            return None

    return wrapper


def is_initialized():
    return dist.is_initialized()
