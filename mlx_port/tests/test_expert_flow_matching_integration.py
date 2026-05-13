"""
Integration smoke test for Expert + FlowMatching sampler.

This test verifies that the full denoising loop can run end-to-end
(using dummy VLM KV cache) without shape or runtime errors.
"""

import pytest

import mlx.core as mx

from mlx_port.expert import Expert, create_expert_step_fn, build_expert_attention_inputs
from mlx_port.expert_loader import load_alpamayo_expert_weights
from mlx_port.flow_matching import FlowMatching


def test_expert_flow_matching_smoke():
    """
    Smoke test:
    - Load expert weights
    - Build dummy VLM cache + attention inputs
    - Create step_fn
    - Run FlowMatching for a few steps
    - Check output shape
    """
    # 1. Load expert
    state = load_alpamayo_expert_weights()
    expert = Expert(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=8256,
        num_layers=36,
    )
    expert.load_weights(list(state.items()))

    # 2. Dummy VLM prompt cache (list of (k, v) tuples)
    # For smoke test we create zero caches of the right shape
    prompt_length = 128
    num_diffusion_tokens = 8  # small for speed
    batch_size = 2

    # Create dummy past_key_values (one per layer)
    past_key_values = []
    for _ in range(36):
        k = mx.zeros((batch_size, 16, prompt_length, 128))  # (B, num_heads, T, head_dim)
        v = mx.zeros((batch_size, 16, prompt_length, 128))
        past_key_values.append((k, v))

    # 3. Build attention inputs
    position_ids, attention_mask = build_expert_attention_inputs(
        prompt_length=prompt_length,
        num_diffusion_tokens=num_diffusion_tokens,
    )

    # 4. Create step_fn using the real modules attached to the expert
    step_fn = create_expert_step_fn(
        expert=expert,
        prompt_cache=past_key_values,
        position_ids=position_ids,
        attention_mask=attention_mask,
        n_diffusion_tokens=num_diffusion_tokens,
        prefill_seq_len=prompt_length,
    )

    # 5. Run FlowMatching for a few steps (use flat action vectors)
    flow = FlowMatching(num_inference_steps=3)
    traj = flow.sample(
        batch_size=batch_size,
        step_fn=step_fn,
        x_dims=(num_diffusion_tokens * 2,),   # flat
    )

    # 6. Basic assertions
    assert traj.shape == (batch_size, num_diffusion_tokens * 2)
    assert not mx.any(mx.isnan(traj))
    assert not mx.any(mx.isinf(traj))

    print("[Integration Test] Expert + FlowMatching smoke test PASSED")
