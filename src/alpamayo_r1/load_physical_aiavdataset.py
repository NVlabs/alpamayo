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

"""Load data from physical_ai_av.PhysicalAIAVDatasetInterface for model inference."""

from typing import Any

import numpy as np
import physical_ai_av
import scipy.spatial.transform as spt
import torch
from einops import rearrange


def load_physical_aiavdataset(
    clip_id: str,
    t0_us: int = 5_100_000,
    avdi: physical_ai_av.PhysicalAIAVDatasetInterface | None = None,
    maybe_stream: bool = True,
    num_history_steps: int = 16,
    num_future_steps: int = 64,
    time_step: float = 0.1,
    camera_features: list | None = None,
    num_frames: int = 4,
) -> dict[str, Any]:
    """Load data from physical_ai_av for model inference.

    This function loads a sample from the physical_ai_av dataset and converts it
    to the format expected by AlpamayoR1 model inference.

    Args:
        clip_id: The clip ID to load data from. Can be obtained from vla_golden.parquet.
        t0_us: The timestamp (in microseconds) at which to sample the trajectory.
            If None, uses a timestamp 5.1s seconds into the clip.
        avdi: Optional pre-initialized PhysicalAIAVDatasetInterface. If None, creates one.
        maybe_stream: Whether to stream data from HuggingFace (if not downloaded locally).
        num_history_steps: Number of history trajectory steps (default: 16 for 1.6s at 10Hz).
        num_future_steps: Number of future trajectory steps (default: 64 for 6.4s at 10Hz).
        time_step: Time step between trajectory points in seconds (default: 0.1s = 10Hz).
        camera_features: List of camera features to load. If None, uses 4 cameras:
            [CAMERA_FRONT_WIDE_120FOV, CAMERA_FRONT_TELE_30FOV,
             CAMERA_CROSS_LEFT_120FOV, CAMERA_CROSS_RIGHT_120FOV].
        num_frames: Number of frames per camera to load (default: 4).

    Returns:
        A dictionary with the following keys:
            - image_frames: torch.Tensor of shape (N_cameras, num_frames, 3, H, W)
            - camera_indices: torch.Tensor of shape (N_cameras,)
            - ego_history_xyz: torch.Tensor of shape (1, 1, num_history_steps, 3)
            - ego_history_rot: torch.Tensor of shape (1, 1, num_history_steps, 3, 3)
            - ego_future_xyz: torch.Tensor of shape (1, 1, num_future_steps, 3)
            - ego_future_rot: torch.Tensor of shape (1, 1, num_future_steps, 3, 3)
            - relative_timestamps: torch.Tensor of shape (N_cameras, num_frames)
            - absolute_timestamps: torch.Tensor of shape (N_cameras, num_frames)
            - t0_us: The t0 timestamp used
            - clip_id: The clip ID
    """
    if avdi is None:
        avdi = physical_ai_av.PhysicalAIAVDatasetInterface()

    if camera_features is None:
        camera_features = [
            avdi.features.CAMERA.CAMERA_CROSS_LEFT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_WIDE_120FOV,
            avdi.features.CAMERA.CAMERA_CROSS_RIGHT_120FOV,
            avdi.features.CAMERA.CAMERA_FRONT_TELE_30FOV,
        ]

    camera_name_to_index = {
        "camera_cross_left_120fov": 0,
        "camera_front_wide_120fov": 1,
        "camera_cross_right_120fov": 2,
        "camera_rear_left_70fov": 3,
        "camera_rear_tele_30fov": 4,
        "camera_rear_right_70fov": 5,
        "camera_front_tele_30fov": 6,
    }

    # Load egomotion data
    egomotion = avdi.get_clip_feature(
        clip_id,
        avdi.features.LABELS.EGOMOTION,
        maybe_stream=maybe_stream,
    )

    assert t0_us > num_history_steps * time_step * 1_000_000, (
        "t0_us must be greater than the history time range"
    )

    # Compute timestamps for trajectory sampling
    # History: [..., t0-0.2s, t0-0.1s, t0] (num_history_steps points ending at t0)
    # Future: [t0+0.1s, t0+0.2s, ..., t0+6.4s] (num_future_steps points after t0)
    history_offsets_us = np.arange(
        -(num_history_steps - 1) * time_step * 1_000_000,
        time_step * 1_000_000 / 2,
        time_step * 1_000_000,
    ).astype(np.int64)
    history_timestamps = t0_us + history_offsets_us

    future_offsets_us = np.arange(
        time_step * 1_000_000,
        (num_future_steps + 0.5) * time_step * 1_000_000,
        time_step * 1_000_000,
    ).astype(np.int64)
    future_timestamps = t0_us + future_offsets_us

    # Get egomotion at history and future timestamps
    ego_history = egomotion(history_timestamps)
    ego_history_xyz = ego_history.pose.translation  # (num_history_steps, 3)
    ego_history_quat = ego_history.pose.rotation.as_quat()  # (num_history_steps, 4)

    ego_future = egomotion(future_timestamps)
    ego_future_xyz = ego_future.pose.translation  # (num_future_steps, 3)
    ego_future_quat = ego_future.pose.rotation.as_quat()  # (num_future_steps, 4)

    # Transform to local frame (relative to t0 pose)
    # The model expects trajectories in the ego frame at t0.
    # Transformation: xyz_local = R_t0^{-1} @ (xyz_world - xyz_t0)
    t0_xyz = ego_history_xyz[-1].copy()  # Position at t0
    t0_quat = ego_history_quat[-1].copy()  # Orientation at t0
    t0_rot = spt.Rotation.from_quat(t0_quat)
    t0_rot_inv = t0_rot.inv()

    # Transform history positions to local frame
    ego_history_xyz_local = t0_rot_inv.apply(ego_history_xyz - t0_xyz)

    # Transform future positions to local frame
    ego_future_xyz_local = t0_rot_inv.apply(ego_future_xyz - t0_xyz)

    # Transform rotations to local frame
    ego_history_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_history_quat)).as_matrix()
    ego_future_rot_local = (t0_rot_inv * spt.Rotation.from_quat(ego_future_quat)).as_matrix()

    # Convert to torch tensors with batch dimensions: (B=1, n_traj_group=1, T, ...)
    ego_history_xyz_tensor = (
        torch.from_numpy(ego_history_xyz_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_history_rot_tensor = (
        torch.from_numpy(ego_history_rot_local).float().unsqueeze(0).unsqueeze(0)
    )
    ego_future_xyz_tensor = torch.from_numpy(ego_future_xyz_local).float().unsqueeze(0).unsqueeze(0)
    ego_future_rot_tensor = torch.from_numpy(ego_future_rot_local).float().unsqueeze(0).unsqueeze(0)

    # Load camera images
    image_frames_list = []
    camera_indices_list = []
    timestamps_list = []

    # Image timestamps: if num_frames=4, load at [t0-0.3s, t0-0.2s, t0-0.1s, t0]
    image_timestamps = np.array(
        [t0_us - (num_frames - 1 - i) * int(time_step * 1_000_000) for i in range(num_frames)],
        dtype=np.int64,
    )

    for cam_feature in camera_features:
        camera = avdi.get_clip_feature(
            clip_id,
            cam_feature,
            maybe_stream=maybe_stream,
        )

        # frames: (num_frames, H, W, 3) uint8
        frames, frame_timestamps = camera.decode_images_from_timestamps(image_timestamps)

        # Convert to (num_frames, 3, H, W) for model input
        frames_tensor = torch.from_numpy(frames)
        frames_tensor = rearrange(frames_tensor, "t h w c -> t c h w")

        # Extract camera name from feature path
        if isinstance(cam_feature, str):
            cam_name = cam_feature.split("/")[-1] if "/" in cam_feature else cam_feature
            cam_name = cam_name.lower()
        else:
            raise ValueError(f"Unexpected camera feature type: {type(cam_feature)}")
        cam_idx = camera_name_to_index.get(cam_name, 0)

        image_frames_list.append(frames_tensor)
        camera_indices_list.append(cam_idx)
        timestamps_list.append(torch.from_numpy(frame_timestamps.astype(np.int64)))

    # Stack and sort by camera index for consistent ordering
    image_frames = torch.stack(image_frames_list, dim=0)  # (N_cameras, num_frames, 3, H, W)
    camera_indices = torch.tensor(camera_indices_list, dtype=torch.int64)  # (N_cameras,)
    all_timestamps = torch.stack(timestamps_list, dim=0)  # (N_cameras, num_frames)

    # Sort by camera index to ensure consistent ordering [0, 1, 2, 6] instead of arbitrary order
    sort_order = torch.argsort(camera_indices)
    image_frames = image_frames[sort_order]
    camera_indices = camera_indices[sort_order]
    all_timestamps = all_timestamps[sort_order]

    # Compute relative timestamps in seconds
    camera_tmin = all_timestamps.min()
    relative_timestamps = (all_timestamps - camera_tmin).float() * 1e-6  # (N_cameras, num_frames)

    return {
        "image_frames": image_frames,  # (N_cameras, num_frames, 3, H, W)
        "camera_indices": camera_indices,  # (N_cameras,)
        "ego_history_xyz": ego_history_xyz_tensor,  # (1, 1, num_history_steps, 3)
        "ego_history_rot": ego_history_rot_tensor,  # (1, 1, num_history_steps, 3, 3)
        "ego_future_xyz": ego_future_xyz_tensor,  # (1, 1, num_future_steps, 3)
        "ego_future_rot": ego_future_rot_tensor,  # (1, 1, num_future_steps, 3, 3)
        "relative_timestamps": relative_timestamps,  # (N_cameras, num_frames)
        "absolute_timestamps": all_timestamps,  # (N_cameras, num_frames)
        "t0_us": t0_us,
        "clip_id": clip_id,
    }
