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

from typing import Mapping

import torch


def summarize_metric(
    metric: Mapping[str, torch.Tensor], disable_summary: bool = False
) -> dict[str, torch.Tensor]:
    """Replaces dict containing multi-dimensional metrics with mean, mean_square, and stdev.

    Args:
        metric: dict[str,Tensor]
            name: shape [B, N]
        disable_summary: bool, if True, return metric without summarizing over groups.

    Returns:
        dict[str,Tensor]
            name: shape [B] # average
            name_sq: shape [B] # average square
            name_std: shape [B] # standard deviation over N
    """
    result = {}
    for metric_name, val in metric.items():
        if val.ndim != 2:
            raise ValueError(
                f"All keys of metric must have values of shape [B,N], "
                f"found {metric_name} with shape {val.shape}"
            )
        mean = val.mean(1)
        result[metric_name] = mean
        if val.shape[1] > 1 and not disable_summary:
            mean_sq = val.pow(2).mean(1)
            std = (mean_sq - mean**2).sqrt()
            result[metric_name + "_std"] = std

    return result


def apply_prefix(prefix: str, data: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Returns a dict with prefix prepended to every key in data."""
    return {prefix + k: v for k, v in data.items()}
