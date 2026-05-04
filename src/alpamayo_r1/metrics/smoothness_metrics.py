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

"""Trajectory smoothness metrics for evaluating predicted ego trajectories.

These metrics measure how comfortable / kinematically plausible a predicted
trajectory is, independent of how close it is to the ground truth. ADE and
FDE answer "is the trajectory accurate?"; the metrics here answer "is the
trajectory drivable?".

Each metric is reported as both a per-batch RMS (overall magnitude) and a
per-batch absolute max (worst peak). All metrics work on planar XY motion
plus yaw and assume a fixed planning frequency.

Note on duplication with ``alpamayo_r1.finetune.rl.rewards.comfort_reward``:
the RL reward turns these dynamics into within-bound booleans for use as a
training signal. This module reports the actual physical magnitudes for
post-hoc evaluation. The two share the same kinematic derivations but serve
different purposes; we keep the math here standalone to avoid a metrics ->
RL dependency.
"""

from __future__ import annotations

import torch

from alpamayo_r1.metrics.metric_utils import summarize_metric


def _diff_pad_last(tensor: torch.Tensor, freq_hz: float) -> torch.Tensor:
    """Forward finite difference, padded by repeating the last delta.

    Returns a tensor with the same shape as ``tensor``, scaled by ``freq_hz``.
    Padding the last delta keeps the output length identical to the input
    length, which matches the convention used by ``comfort_reward.gather_dynamics``.
    """
    delta = tensor[..., 1:] - tensor[..., :-1]
    last = delta[..., -1:].clone()
    return torch.cat([delta, last], dim=-1) * freq_hz


def _diff_yaw_pad_last(yaw: torch.Tensor, freq_hz: float) -> torch.Tensor:
    """Forward finite difference of yaw, handling wraparound at +/- pi."""
    diff = torch.diff(yaw, dim=-1)
    diff = torch.where(diff > torch.pi, diff - 2 * torch.pi, diff)
    diff = torch.where(diff < -torch.pi, diff + 2 * torch.pi, diff)
    rate = diff * freq_hz
    last = rate[..., -1:].clone()
    return torch.cat((rate, last), dim=-1)


def gather_dynamics(
    pred_xyz: torch.Tensor,
    pred_rot: torch.Tensor,
    planning_freq_hz: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Derive ego dynamics from a predicted trajectory.

    Args:
        pred_xyz: [..., T, 3] positions in the ego frame.
        pred_rot: [..., T, 3, 3] rotation matrices.
        planning_freq_hz: Inverse of the time step between samples (default 10 Hz,
            matching the model's standard 0.1 s output cadence).

    Returns:
        Dict of [..., T] tensors:

            - ``yaw_rate``         (rad/s)
            - ``yaw_accel``        (rad/s^2)
            - ``v_lon``            (m/s, along ego heading)
            - ``v_lat``            (m/s, perpendicular to ego heading)
            - ``accel_lon``        (m/s^2, longitudinal)
            - ``accel_lat``        (m/s^2, lateral)
            - ``jerk_lon``         (m/s^3, longitudinal)
    """
    ego_x = pred_xyz[..., 0]
    ego_y = pred_xyz[..., 1]
    ego_h = torch.atan2(pred_rot[..., 1, 0], pred_rot[..., 0, 0])

    dx = _diff_pad_last(ego_x, planning_freq_hz)
    dy = _diff_pad_last(ego_y, planning_freq_hz)
    yaw_rate = _diff_yaw_pad_last(ego_h, planning_freq_hz)
    yaw_accel = _diff_pad_last(yaw_rate, planning_freq_hz)

    v_lon = dx * torch.cos(ego_h) + dy * torch.sin(ego_h)
    v_lat = -dx * torch.sin(ego_h) + dy * torch.cos(ego_h)
    accel_lon = _diff_pad_last(v_lon, planning_freq_hz)
    accel_lat = _diff_pad_last(v_lat, planning_freq_hz)
    jerk_lon = _diff_pad_last(accel_lon, planning_freq_hz)

    return {
        "yaw_rate": yaw_rate,
        "yaw_accel": yaw_accel,
        "v_lon": v_lon,
        "v_lat": v_lat,
        "accel_lon": accel_lon,
        "accel_lat": accel_lat,
        "jerk_lon": jerk_lon,
    }


# Which dynamics to summarize as smoothness signals. Each entry maps a
# user-facing prefix to the ``gather_dynamics`` key it summarizes.
_SMOOTHNESS_SOURCES: dict[str, str] = {
    "smoothness/jerk_lon": "jerk_lon",
    "smoothness/accel_lon": "accel_lon",
    "smoothness/accel_lat": "accel_lat",
    "smoothness/yaw_rate": "yaw_rate",
    "smoothness/yaw_accel": "yaw_accel",
}


def compute_smoothness_metrics(
    pred_xyz: torch.Tensor,
    pred_rot: torch.Tensor,
    disable_summary: bool = False,
    planning_freq_hz: float = 10.0,
) -> dict[str, torch.Tensor]:
    """Compute trajectory smoothness metrics over the standard alpamayo shapes.

    For each kinematic signal (longitudinal/lateral acceleration, longitudinal
    jerk, yaw rate, yaw acceleration) report two per-batch numbers:

        - ``<name>_rms``  -- RMS magnitude across the trajectory (typical signal level)
        - ``<name>_max``  -- absolute peak magnitude across the trajectory (worst case)

    Each is averaged over the K samples per group (so a noisy single sample
    does not dominate) and then summarized over the N groups via
    ``summarize_metric`` (adds ``_std`` keys when N > 1).

    Args:
        pred_xyz: [B, N, K, T, 3] predicted ego positions.
        pred_rot: [B, N, K, T, 3, 3] predicted ego rotations.
        disable_summary: if True, skip the N-group summarization (no ``_std`` keys).
        planning_freq_hz: Inverse time step between samples (default 10 Hz).

    Returns:
        dict[str, torch.Tensor] of [B] tensors. Keys are
        ``smoothness/{jerk_lon,accel_lon,accel_lat,yaw_rate,yaw_accel}_{rms,max}``,
        plus ``_std`` variants when N > 1 and ``disable_summary`` is False.
    """
    if pred_xyz.ndim != 5 or pred_xyz.shape[-1] != 3:
        raise ValueError(
            f"pred_xyz must have shape [B, N, K, T, 3], got {tuple(pred_xyz.shape)}"
        )
    if pred_rot.ndim != 6 or pred_rot.shape[-2:] != (3, 3):
        raise ValueError(
            f"pred_rot must have shape [B, N, K, T, 3, 3], got {tuple(pred_rot.shape)}"
        )
    if pred_xyz.shape[:-1] != pred_rot.shape[:-2]:
        raise ValueError(
            f"pred_xyz and pred_rot leading dims must match: "
            f"{tuple(pred_xyz.shape[:-1])} vs {tuple(pred_rot.shape[:-2])}"
        )

    dynamics = gather_dynamics(pred_xyz, pred_rot, planning_freq_hz=planning_freq_hz)

    out: dict[str, torch.Tensor] = {}
    for prefix, src_key in _SMOOTHNESS_SOURCES.items():
        signal = dynamics[src_key]  # [B, N, K, T]
        rms = signal.pow(2).mean(dim=-1).sqrt()  # [B, N, K]
        peak = signal.abs().amax(dim=-1)  # [B, N, K]
        # Aggregate K -> [B, N] by averaging (worst-K is reported elsewhere via
        # _max already; here we want a representative per-sample value).
        out[f"{prefix}_rms"] = rms.mean(dim=2)
        out[f"{prefix}_max"] = peak.mean(dim=2)

    return summarize_metric(out, disable_summary)
