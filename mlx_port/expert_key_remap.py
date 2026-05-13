"""
Key remapping utilities for loading Alpamayo action expert weights into MLX.

Alpamayo stores expert weights under the "expert." prefix:
    expert.layers.0.self_attn.q_proj.weight
    expert.layers.0.mlp.gate_proj.weight
    ...

MLX will expect a clean hierarchy starting with "layers." or "model.layers".

This module provides a minimal remapper so the fine-tuned Alpamayo expert
can be loaded without waiting for a full conversion pipeline.
"""

from typing import Dict

import mlx.core as mx


def remap_alpamayo_expert_keys(state: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """
    Convert Alpamayo expert weight keys to a clean MLX-friendly form.

    Alpamayo stores:
        expert.layers.*          (transformer decoder layers)
        action_in_proj.*         (Fourier + MLP encoder for actions)
        action_out_proj.*        (final projection)
        action_space.*           (normalization stats)

    We strip the leading "expert." so the transformer part becomes:
        layers.*                 (ready for a custom Expert model)
    """
    remapped = {}
    for key, value in state.items():
        new_key = key

        # 1. Strip leading "expert." for the transformer backbone
        if new_key.startswith("expert."):
            new_key = new_key[7:]

        # 2. Keep action_in_proj, action_out_proj, action_space as-is
        #    (they will be handled by dedicated MLX modules later)

        remapped[new_key] = value

    return remapped


def load_alpamayo_expert_state(alpamayo_dir: str) -> Dict[str, mx.array]:
    """
    Load all non-VLM (expert + action) weights from an Alpamayo-R1 checkpoint
    and return them already remapped.
    """
    from pathlib import Path

    alpamayo_path = Path(alpamayo_dir)
    state = {}

    for shard in sorted(alpamayo_path.glob("model-*.safetensors")):
        tensors = mx.load(str(shard))
        for key, arr in tensors.items():
            if not key.startswith("vlm."):
                state[key] = arr

    return remap_alpamayo_expert_keys(state)
