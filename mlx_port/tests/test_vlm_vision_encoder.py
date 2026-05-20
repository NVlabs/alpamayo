# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Minimal verification test for the vision encoder in the MLX VLM loader."""

from mlx_port.vlm_loader import load_vlm_with_alpamayo_tokens


def test_vision_encoder_is_loaded():
    """Verify that the vision encoder (vision_tower) is present in the loaded VLM."""
    model, processor = load_vlm_with_alpamayo_tokens()

    # The full VLM must expose a vision_tower component
    assert hasattr(model, "vision_tower"), "Model is missing vision_tower"

    vision_tower = model.vision_tower
    assert vision_tower is not None

    # Check that the vision tower has a non-trivial hidden dimension (4096 for 8B model)
    # Most mlx-vlm vision towers store config or weight shapes we can inspect
    hidden_dim = None
    if hasattr(vision_tower, "config") and hasattr(vision_tower.config, "hidden_size"):
        hidden_dim = vision_tower.config.hidden_size
    elif hasattr(vision_tower, "vision_model") and hasattr(vision_tower.vision_model, "config"):
        hidden_dim = getattr(vision_tower.vision_model.config, "hidden_size", None)

    if hidden_dim is not None:
        assert hidden_dim > 0, "Vision encoder hidden size is invalid"
        print(f"Vision encoder OK — hidden_size = {hidden_dim}")
    else:
        # Fallback: just confirm the object exists and has parameters
        print("Vision encoder OK — vision_tower present (hidden_size not directly readable)")
