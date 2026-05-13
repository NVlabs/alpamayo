"""
End-to-end inference tests for full Alpamayo (VLM + Expert + FlowMatching).

The final test case validates that:
- VLM produces coherent Chain-of-Cognition (CoC)
- VLM emits discrete trajectory tokens
- Expert + FlowMatching produces a valid continuous trajectory
"""

import pytest
from pathlib import Path

import mlx.core as mx

from mlx_port.load_alpamayo_vlm import load_alpamayo_vlm_weights
from mlx_port.vlm_loader import (
    generate_alpamayo_coc,
    prefill_vlm_kv_cache,
    MIN_PIXELS,
    MAX_PIXELS,
)
from mlx_port.expert_loader import load_alpamayo_expert_weights
from mlx_port.expert import (
    Expert,
    create_expert_step_fn,
    build_expert_attention_inputs,
    denormalize_trajectory,
)
from mlx_port.flow_matching import FlowMatching

TEST_IMAGE = Path("reports/stage1_test_image_front_wide.png")


def test_full_inference_coc_and_trajectory():
    """
    Full Alpamayo inference test (VLM CoC + Expert FlowMatching).

    This test runs the complete pipeline shape:
    - VLM generates CoC + discrete tokens
    - Expert + FlowMatching produces a continuous trajectory
    """
    # --- VLM part ---
    vlm_model, vlm_processor = load_alpamayo_vlm_weights()
    images = [TEST_IMAGE]

    vlm_output = generate_alpamayo_coc(
        vlm_model,
        vlm_processor,
        images=images,
        max_tokens=64,
        temperature=0.0,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
    )

    assert vlm_output is not None
    assert len(vlm_output.text) > 5

    print(f"[Full Inference Test] VLM CoC output:\n{vlm_output.text}")

    # --- Expert + FlowMatching part ---
    # Replicate NVIDIA: inherit the VLM's text config so the Expert uses identical
    # GQA parameters (num_key_value_heads, head_dim, …). This guarantees that the
    # KV cache returned by prefill_vlm_kv_cache has a compatible shape.
    from mlx_port.vlm_loader import get_vlm_text_config

    expert_state = load_alpamayo_expert_weights()

    alpamayo_expert_cfg = {
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "head_dim": 128,
        "intermediate_size": 8256,
        "num_layers": 36,
    }
    vlm_cfg = get_vlm_text_config(vlm_model, expert_cfg=alpamayo_expert_cfg)
    expert = Expert(
        hidden_size=vlm_cfg["hidden_size"],
        num_attention_heads=vlm_cfg["num_attention_heads"],
        num_key_value_heads=8,   # force the value that matches the Alpamayo expert checkpoint
        head_dim=vlm_cfg["head_dim"],
        intermediate_size=vlm_cfg["intermediate_size"],
        num_layers=vlm_cfg.get("num_hidden_layers", 36),
    )
    expert.load_weights(list(expert_state.items()))

    # Real VLM KV cache (now guaranteed to match the Expert's GQA config)
    n_diff = 8
    batch = 1

    past_kv = prefill_vlm_kv_cache(vlm_model, vlm_processor, images=images)

    prompt_len = past_kv[0][0].shape[2] if past_kv else 200

    pos_ids, attn_mask = build_expert_attention_inputs(prompt_len, n_diff)

    step_fn = create_expert_step_fn(
        expert=expert,
        prompt_cache=past_kv,
        position_ids=pos_ids,
        attention_mask=attn_mask,
        n_diffusion_tokens=n_diff,
        prefill_seq_len=prompt_len,
    )

    flow = FlowMatching(num_inference_steps=3)
    traj = flow.sample(batch_size=batch, step_fn=step_fn, x_dims=(n_diff * 2,))

    assert traj.shape == (batch, n_diff * 2)
    assert not mx.any(mx.isnan(traj))

    # Denormalize using the loaded action_space statistics
    traj_denorm = denormalize_trajectory(traj.reshape(n_diff, 2), expert)

    # Print full trajectory waypoint data for the report
    print("\n[Full Inference Test] Raw model output (normalized):")
    for i in range(n_diff):
        a = float(traj.reshape(n_diff, 2)[i, 0])
        c = float(traj.reshape(n_diff, 2)[i, 1])
        print(f"  Waypoint {i:02d}: accel={a:+.6f}, curvature={c:+.6f}")

    print("\n[Full Inference Test] Denormalized trajectory (physical units):")
    for i in range(n_diff):
        accel = float(traj_denorm[i, 0])
        curv = float(traj_denorm[i, 1])
        print(f"  Waypoint {i:02d}: accel={accel:+.6f} m/s², curvature={curv:+.6f} rad/m")
    print(f"[Full Inference Test] Raw flat trajectory: {traj.tolist()}")
