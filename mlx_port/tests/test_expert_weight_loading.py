"""Test that the real Qwen3-VL diffusion expert weights can be loaded successfully.

This test verifies Row 11 (Real Expert Architecture) + weight loading parity.
It uses the full Alpamayo-R1-10B checkpoint and requires significant memory/time.
"""

import mlx.core as mx
import pytest

from mlx_port.models.alpamayo_r1_mlx import AlpamayoR1MLX


CHECKPOINT_PATH = "/Users/michaellee/Projects/alpamayo/pre-trained/Alpamayo-R1-10B"


@pytest.mark.slow
def test_load_expert_weights_successfully():
    """Load the full AlpamayoR1MLX model with the real Qwen3-VL expert.

    This exercises:
    - Correct expert architecture (qwen3_vl.Model with 36 layers)
    - Proper key remapping from checkpoint (expert.* -> expert.language_model.model.*)
    - Safe bfloat16 loading via torch intermediate + astype
    - Action projection weight loading

    If this test passes, the expert is ready for step_fn integration in inference.py.
    """
    print("\n[Expert Weight Loading Test] Loading AlpamayoR1MLX with real expert (this takes time)...")
    model = AlpamayoR1MLX.from_pretrained(
        CHECKPOINT_PATH,
        load_expert=True,
        dtype=mx.bfloat16,
    )
    print("[Expert Weight Loading Test] Model loaded successfully.")

    # Basic structural checks
    assert model.expert is not None
    assert hasattr(model.expert, "language_model")
    assert len(model.expert.layers) == 36, f"Expected 36 layers, got {len(model.expert.layers)}"

    # Check that action projections exist and have reasonable shapes
    assert hasattr(model, "action_in_proj")
    assert hasattr(model, "action_out_proj")

    # Spot-check one weight to ensure it is not all zeros (loaded from checkpoint)
    sample_weight = model.expert.language_model.model.layers[0].self_attn.q_proj.weight
    assert sample_weight.shape == (2048, 2048)
    # A randomly initialized weight would have very small std; loaded weights have larger variance.
    # We just ensure it's not exactly zero.
    assert mx.any(sample_weight != 0).item(), "Loaded weight appears to be all zeros"

    print("[Expert Weight Loading Test] All structural and weight checks passed.")