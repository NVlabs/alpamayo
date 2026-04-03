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

"""Comfort reward for Alpamayo Cosmos-RL training.

Uses manually defined comfort thresholds for longitudinal/lateral acceleration,
jerk, yaw rate, and yaw acceleration.
"""

from typing import Final

import torch

MAX_ABS_MAG_JERK: Final[float] = 8.37  # [m/s^3]
MAX_ABS_LAT_ACCEL: Final[float] = 4.89  # [m/s^2]
MAX_LON_ACCEL: Final[float] = 2.40  # [m/s^2]
MIN_LON_ACCEL: Final[float] = -4.05  # [m/s^2]
MAX_ABS_YAW_ACCEL: Final[float] = 1.93  # [rad/s^2]
MAX_ABS_LON_JERK: Final[float] = 4.13  # [m/s^3]
MAX_ABS_YAW_RATE: Final[float] = 0.95  # [rad/s]
PLANNING_FREQ: Final[float] = 10.0  # [Hz]

COMFORT_METRIC_CONFIG_DICT = {
    "comfort/lon_accel": ["ego_dv_lon", MIN_LON_ACCEL, MAX_LON_ACCEL],
    "comfort/lat_accel": ["ego_dv_lat", -MAX_ABS_LAT_ACCEL, MAX_ABS_LAT_ACCEL],
    "comfort/jerk": ["ego_jerk", -MAX_ABS_MAG_JERK, MAX_ABS_MAG_JERK],
    "comfort/lon_jerk": ["ego_jerk_lon", -MAX_ABS_LON_JERK, MAX_ABS_LON_JERK],
    "comfort/yaw_accel": ["ego_yaw_accel", -MAX_ABS_YAW_ACCEL, MAX_ABS_YAW_ACCEL],
    "comfort/yaw_rate": ["ego_yaw_rate", -MAX_ABS_YAW_RATE, MAX_ABS_YAW_RATE],
}


def _diff(tensor: torch.Tensor) -> torch.Tensor:
    """Compute finite differences scaled by planning frequency."""
    delta = tensor[..., 1:] - tensor[..., :-1]
    last_delta = delta[..., -1:].clone()
    return torch.cat([delta, last_delta], dim=-1) * PLANNING_FREQ


def _diff_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Compute yaw rate from yaw angles, handling wraparound at +/-pi."""
    yaw_diff = torch.diff(yaw, dim=-1)
    yaw_diff = torch.where(yaw_diff > torch.pi, yaw_diff - 2 * torch.pi, yaw_diff)
    yaw_diff = torch.where(yaw_diff < -torch.pi, yaw_diff + 2 * torch.pi, yaw_diff)
    yaw_rate = yaw_diff * PLANNING_FREQ
    last_yaw_rate = yaw_rate[..., -1:].clone()
    yaw_rate = torch.cat((yaw_rate, last_yaw_rate), dim=-1)
    return yaw_rate


def _within_bound(
    metric: torch.Tensor,
    min_bound: float | None = None,
    max_bound: float | None = None,
) -> torch.Tensor:
    """Check whether all values in *metric* fall within [min_bound, max_bound]."""
    min_bound = min_bound if min_bound is not None else -float("inf")
    max_bound = max_bound if max_bound is not None else float("inf")
    metric_within_bound = (metric > min_bound) & (metric < max_bound)
    return torch.all(metric_within_bound, axis=-1).float()


def gather_dynamics(
    pred_xyz: torch.Tensor,
    pred_rot: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Gather ego dynamics from predicted trajectory.

    Args:
        pred_xyz: [B, N, K, T, 3]
        pred_rot: [B, N, K, T, 3, 3]
    """
    ego_x = pred_xyz[..., 0]
    ego_y = pred_xyz[..., 1]
    ego_h = torch.atan2(pred_rot[..., 1, 0], pred_rot[..., 0, 0])
    ego_dx = _diff(ego_x)
    ego_dy = _diff(ego_y)
    ego_yaw_rate = ego_dh = _diff_yaw(ego_h)
    ego_v = torch.linalg.norm(torch.stack([ego_dx, ego_dy], dim=-1), dim=-1)
    ego_dv = _diff(ego_v)
    ego_jerk = _diff(ego_dv)
    ego_yaw_accel = _diff(ego_dh)
    ego_v_lon = ego_dx * torch.cos(ego_h) + ego_dy * torch.sin(ego_h)
    ego_dv_lon = _diff(ego_v_lon)
    ego_jerk_lon = _diff(ego_dv_lon)
    ego_dv_lat = _diff(-ego_dx * torch.sin(ego_h) + ego_dy * torch.cos(ego_h))

    return {
        "ego_yaw_rate": ego_yaw_rate,
        "ego_dv": ego_dv,
        "ego_jerk": ego_jerk,
        "ego_yaw_accel": ego_yaw_accel,
        "ego_dv_lon": ego_dv_lon,
        "ego_jerk_lon": ego_jerk_lon,
        "ego_dv_lat": ego_dv_lat,
    }


def compute_comfort(
    pred_xyz: torch.Tensor,
    pred_rot: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Compute comfort metrics. Score equals 1 if all values are within bounds.

    Args:
        pred_xyz: [B, N, K, T, 3]
        pred_rot: [B, N, K, T, 3, 3]

    Returns:
        dict mapping comfort metric names to [B] tensors.
    """
    ego_dynamics = gather_dynamics(pred_xyz, pred_rot)

    comfort_metric_dict = {}
    for name, (dyn_key, lo, hi) in COMFORT_METRIC_CONFIG_DICT.items():
        comfort_metric_dict[name] = _within_bound(ego_dynamics[dyn_key], lo, hi).mean(2)

    for name in comfort_metric_dict:
        comfort_metric_dict[name] = comfort_metric_dict[name].mean(dim=-1)
    return comfort_metric_dict
