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

from typing import Iterable

import torch

from alpamayo_r1.geometry import coordinates
from alpamayo_r1.metrics.metric_utils import summarize_metric


def compute_ade(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    timestep_horizon: int | None = None,
    only_xy: bool = True,
) -> torch.Tensor:
    """Compute ade over K samples.

    Args:
        pred_xyz (torch.Tensor): [B, N, K, T, 3], where N is the number of groups of K candidates.
            For models that use sampling to generate trajectory, we can use these group numbers to
            compute mean/std.
        gt_xyz (torch.Tensor): [B, T, 3]
        only_xy (bool): if True, only compute ade over the BEV (XY) plane.

    Returns:
        ade: [B, N, K].
    """
    diff = pred_xyz - gt_xyz[:, None, None, :, :]  # [B, N, K, T, 3]
    if only_xy:
        diff = diff[..., :2]
    l2 = torch.linalg.norm(diff, ord=2, dim=-1)  # [B, N, K, T]
    if timestep_horizon is not None:
        if timestep_horizon > l2.shape[-1]:
            raise ValueError(f"{timestep_horizon=} must be less than {l2.shape[-1]=}")
        l2 = l2[..., :timestep_horizon]
    return l2.mean(dim=-1)  # [B, N, K]


def compute_minade(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    disable_summary: bool = False,
    timestep_horizons: Iterable[int] = [5, 10, 30, 50],
    only_xy: bool = True,
    time_step: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compute min_ade over K samples.

    Args:
        pred_xyz (torch.Tensor): [B, N, K, T, 3], where N is the number of groups of K candidates.
            For models that use sampling to generate trajectory, we can use these group numbers to
            compute mean/std.
        gt_xyz (torch.Tensor): [B, T, 3]
        disable_summary (bool): if True, return min_ade without summarizing over groups.
        timestep_horizons (Iterable[int]): List of time horizons to compute min_ade for.
            Horizons exceeding the available timesteps will be skipped.
        only_xy (bool): if True, only compute min_ade over the BEV (XY) plane.
        time_step (float): Time step of the trajectory.

    Returns:
        dict[str, torch.Tensor]:
            min_ade: [B], average min_ade over N groups for each batch element
            min_ade_sq: [B], average squared min_ade over N groups, for each batch element
    """
    _, N, _, T, _ = pred_xyz.shape
    # Filter timestep_horizons to only include those that don't exceed available timesteps
    valid_horizons = [t for t in timestep_horizons if t <= T]

    diff = pred_xyz - gt_xyz[:, None, None, :, :]  # [B, N, K, T, 3]
    if only_xy:
        diff = diff[..., :2]
    l2 = torch.linalg.norm(diff, ord=2, dim=-1)  # [B, N, K, T]
    idx = l2.mean(dim=-1).argmin(dim=2)  # [B, N]
    min_diff = torch.take_along_dim(l2, idx[:, :, None, None], dim=2)  # [B, N, 1, T]
    min_diff = min_diff.squeeze(2)  # [B, N, T]

    out = {"min_ade": min_diff.mean(dim=2)}
    for t in valid_horizons:
        min_ade_t = min_diff[:, :, :t].mean(dim=2)
        out[f"min_ade/by_t={t * time_step:.1f}"] = min_ade_t
    return summarize_metric(out, disable_summary)


def compute_fde(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    timestep_horizon: int | None = None,
    only_xy: bool = True,
) -> torch.Tensor:
    """Compute Final Displacement Error (FDE) over K samples.

    FDE is the L2 distance between the predicted and ground-truth positions at
    the **last** timestep of the trajectory. This is the standard companion to
    ADE in trajectory-prediction literature: ADE measures average error over
    the horizon, FDE measures error at the end of it.

    Args:
        pred_xyz (torch.Tensor): [B, N, K, T, 3], where N is the number of groups
            of K candidates.
        gt_xyz (torch.Tensor): [B, T, 3]
        timestep_horizon (int | None): If set, evaluate FDE at this timestep
            (1-indexed count from the start) instead of the final one. Useful
            for reporting FDE at, e.g., 3 s. Must be ``<= T``.
        only_xy (bool): if True, compute FDE over the BEV (XY) plane only.

    Returns:
        fde: [B, N, K].
    """
    T = pred_xyz.shape[3]
    if timestep_horizon is not None:
        if timestep_horizon > T:
            raise ValueError(f"{timestep_horizon=} must be less than or equal to {T=}")
        end_idx = timestep_horizon - 1
    else:
        end_idx = T - 1

    diff = pred_xyz[:, :, :, end_idx, :] - gt_xyz[:, None, None, end_idx, :]
    if only_xy:
        diff = diff[..., :2]
    return torch.linalg.norm(diff, ord=2, dim=-1)  # [B, N, K]


def compute_minfde(
    pred_xyz: torch.Tensor,
    gt_xyz: torch.Tensor,
    disable_summary: bool = False,
    timestep_horizons: Iterable[int] = [5, 10, 30, 50],
    only_xy: bool = True,
    time_step: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Compute minimum Final Displacement Error (minFDE) over K samples.

    Mirrors ``compute_minade``: per-(B, N) we take the K-sample minimum of FDE,
    then summarize over the N groups via ``summarize_metric`` (adds ``_std``
    keys when N > 1 and summarization is not disabled).

    Args:
        pred_xyz (torch.Tensor): [B, N, K, T, 3]
        gt_xyz (torch.Tensor): [B, T, 3]
        disable_summary (bool): if True, return only the per-group means.
        timestep_horizons (Iterable[int]): Per-horizon FDE keys to emit, in
            number of timesteps. Horizons exceeding T are silently skipped.
        only_xy (bool): if True, compute FDE on the BEV (XY) plane only.
        time_step (float): Seconds per step; used to label per-horizon keys.

    Returns:
        dict[str, torch.Tensor]: per-batch tensors of shape ``[B]``:

            - ``min_fde``: minFDE at the final available timestep, averaged
              over the N groups.
            - ``min_fde/by_t={H:.1f}``: minFDE at the timestep
              ``int(H / time_step)``, for each horizon ``t in timestep_horizons``
              that satisfies ``t <= T``. ``H = t * time_step`` is in seconds.
            - ``<key>_std``: stdev across the N groups, added by
              ``summarize_metric`` when ``N > 1`` and ``disable_summary`` is False.
    """
    T = pred_xyz.shape[3]
    valid_horizons = [t for t in timestep_horizons if t <= T]

    diff = pred_xyz - gt_xyz[:, None, None, :, :]  # [B, N, K, T, 3]
    if only_xy:
        diff = diff[..., :2]
    l2 = torch.linalg.norm(diff, ord=2, dim=-1)  # [B, N, K, T]

    out = {"min_fde": l2[..., -1].min(dim=2).values}  # [B, N]
    for t in valid_horizons:
        out[f"min_fde/by_t={t * time_step:.1f}"] = l2[..., t - 1].min(dim=2).values
    return summarize_metric(out, disable_summary)


def compute_grouped_corner_distance(
    pred_xyz: torch.Tensor,
    pred_rot: torch.Tensor,
    gt_xyz: torch.Tensor,
    gt_rot: torch.Tensor,
    dims: torch.Tensor,
    disable_summary: bool = False,
) -> dict[str, torch.Tensor]:
    """Compute corner distance over N groups, each containing K trajectory samples.

    Args:
        pred_xyz (torch.Tensor): [B, N, K, T, 3], where N is the number of groups of K candidates.
            For models that use sampling to generate trajectory, we can use these group numbers to
            compute mean/std.
        pred_rot (torch.Tensor): [B, N, K, T, 3, 3]
        gt_xyz (torch.Tensor): [B, T, 3]
        gt_rot (torch.Tensor): [B, T, 3, 3]
        dims (torch.Tensor): [3]
        disable_summary (bool): if True, return corner_distance without summarizing over groups.

    Returns:
        dict[str, torch.Tensor]:
            corner_distance: [B], average corner distance over N groups
            corner_distance_sq: [B], average squared corner distance over N groups
    """
    _, N = pred_xyz.shape[:2]
    corner_pred = coordinates.xyzrot_to_corners(
        pred_xyz,
        pred_rot,
        dims.view(1, 1, 1, 1, 3),
    )
    corner_gt = coordinates.xyzrot_to_corners(
        gt_xyz,
        gt_rot,
        dims.view(1, 1, 3),
    )
    distance = (corner_pred - corner_gt[:, None, None]).norm(dim=-1)  # [B, N, K, T, 8]
    distance = distance.min(dim=2)[0].mean(dim=(2, 3))
    if disable_summary or (N == 1):
        return {"corner_distance": distance.mean(dim=1)}
    else:
        return summarize_metric({"corner_distance": distance})
