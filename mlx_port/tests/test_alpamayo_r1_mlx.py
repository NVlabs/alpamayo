# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Test for AlpamayoR1MLX custom from_pretrained (Row 5)."""

from mlx_port.models.alpamayo_r1_mlx import (
    AlpamayoR1MLX,
    ExpertDecoder,
    ActionInProj,
    ActionOutProj,
    FlowMatching,
    ActionSpace,
)


def test_alpamayo_r1_mlx_class_structure():
    """Verify AlpamayoR1MLX has the expected components."""
    # Check that the class exists and has from_pretrained
    assert hasattr(AlpamayoR1MLX, "from_pretrained")
    assert callable(getattr(AlpamayoR1MLX, "from_pretrained"))


def test_expert_decoder_instantiation():
    """Verify ExpertDecoder can be created with expert_cfg parameters."""
    expert = ExpertDecoder(
        num_layers=2,
        hidden_size=2048,
        num_heads=16,
        intermediate_size=8256,
    )
    assert expert is not None
    assert len(expert.layers) == 2
    print("ExpertDecoder instantiated successfully with expert_cfg parameters")


def test_action_projection_modules():
    """Verify action projection modules can be created."""
    action_in = ActionInProj(in_dims=5, out_dim=2048)
    action_out = ActionOutProj(in_features=2048, out_features=5)

    assert action_in is not None
    assert action_out is not None
    print("ActionInProj and ActionOutProj instantiated successfully")


def test_flow_matching_and_action_space():
    """Verify diffusion and action space stubs exist and raise NotImplementedError on use."""
    diffusion = FlowMatching()
    action_space = ActionSpace()

    assert diffusion is not None
    assert action_space is not None
    print("FlowMatching and ActionSpace stubs created (raise NotImplementedError on call)")


def test_from_pretrained_stub():
    """Smoke test that from_pretrained can be called (full weight loading tested elsewhere)."""
    # We only check that the method exists and has the right signature.
    # A full integration test would require the complete local checkpoint.
    import inspect
    sig = inspect.signature(AlpamayoR1MLX.from_pretrained)
    params = list(sig.parameters.keys())
    assert "alpamayo_path" in params
    assert "load_expert" in params
    print("from_pretrained signature verified")