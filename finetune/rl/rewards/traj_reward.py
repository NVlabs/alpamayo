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

"""Trajectory distance metrics for Alpamayo Cosmos-RL reward computation."""

import torch


def calculate_ade(pred_trajectory: torch.Tensor, gt_trajectory: torch.Tensor) -> float:
    """Average Displacement Error (XY only)."""
    assert pred_trajectory.shape == gt_trajectory.shape, (
        f"Shape mismatch: pred {pred_trajectory.shape} vs gt {gt_trajectory.shape}"
    )
    pred_xy = pred_trajectory[..., :2]
    gt_xy = gt_trajectory[..., :2]
    distances = torch.linalg.norm(pred_xy - gt_xy, dim=-1)
    return float(distances.mean())
