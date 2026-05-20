# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test data loading from local PAI-CoC dataset for Stage 1 MLX port."""

import pytest
import torch

from alpamayo_r1.load_physical_aiavdataset import load_physical_aiavdataset


def test_load_physical_aiavdataset_with_local_dir():
    """Test that data loading works with local PAI-CoC directory."""
    clip_id = "25cd4769-5dcf-4b53-a351-bf2c5deb6124"
    local_dir = "/Volumes/MicronSSD/pai_coc"

    data = load_physical_aiavdataset(
        clip_id=clip_id,
        t0_us=5_100_000,
        local_dir=local_dir,
        maybe_stream=True,  # allow fallback streaming for partial local subset
        num_history_steps=16,
        num_future_steps=64,
        num_frames=4,
    )

    # Verify expected keys
    expected_keys = {
        "image_frames",
        "camera_indices",
        "ego_history_xyz",
        "ego_history_rot",
        "ego_future_xyz",
        "ego_future_rot",
        "relative_timestamps",
        "absolute_timestamps",
        "t0_us",
        "clip_id",
    }
    assert expected_keys.issubset(data.keys()), f"Missing keys: {expected_keys - data.keys()}"

    # Verify shapes
    assert data["image_frames"].shape[0] == 4, "Expected 4 cameras"
    assert data["image_frames"].shape[1] == 4, "Expected 4 frames per camera"
    assert data["image_frames"].ndim == 5  # (N_cameras, num_frames, 3, H, W)

    assert data["ego_history_xyz"].shape == (1, 1, 16, 3)
    assert data["ego_history_rot"].shape == (1, 1, 16, 3, 3)
    assert data["ego_future_xyz"].shape == (1, 1, 64, 3)

    assert data["clip_id"] == clip_id
    assert data["t0_us"] == 5_100_000

    # Sanity: images are uint8 or float in [0,255] range after load
    assert data["image_frames"].dtype in (torch.uint8, torch.float32, torch.float64)
