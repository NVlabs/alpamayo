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

from typing import Any
from collections import defaultdict

from rich.console import Console
from rich.pretty import pprint
import torch
import random
import numpy as np


def pformat(obj: Any) -> str:
    """Pretty format an object."""
    console = Console()
    with console.capture() as capture:
        pprint(obj, console=console, expand_all=True)
    return capture.get()


def get_param_count(nn_model: torch.nn.Module, depth: int = 2) -> dict:
    """Get parameter counts for each module."""
    if depth < 1:
        raise ValueError("Provided depth must be greater than 0 (got %d)" % depth)
    param_counts = defaultdict(int, {"total_params": 0, "trainable_params": 0})
    for n, p in nn_model.named_parameters():
        names = ".".join(n.split(".")[:depth])
        param_counts[names] += p.numel()
        if p.requires_grad:
            param_counts["trainable_params"] += p.numel()
        param_counts["total_params"] += p.numel()

    return param_counts


def seed_everything(seed: int) -> None:
    """Seed all random number generators."""
    random.seed(seed)  # for Python random module.
    np.random.seed(seed)  # for NumPy.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
