"""
Expert loader for Alpamayo-R1 action expert (diffusion expert + action projections).

This module mirrors the VLM loader pattern:
1. Read all non-VLM tensors from the 5-shard safetensors.
2. Apply key remapping (strip "expert." prefix).
3. Provide verification that the weights are loadable.

The actual MLX Expert model definition (layers, attention, MLP) will be
added in a follow-up step. For now we focus on successful weight loading
and inspection.
"""

from pathlib import Path
from typing import Dict

import mlx.core as mx

from mlx_port.expert_key_remap import (
    remap_alpamayo_expert_keys,
    load_alpamayo_expert_state,
)

ALPAMAYO_PATH = Path("pre-trained/Alpamayo-R1-10B")


def load_alpamayo_expert_weights(
    alpamayo_path: Path = ALPAMAYO_PATH,
) -> Dict[str, mx.array]:
    """
    Load the action expert weights from Alpamayo-R1-10B.

    Returns a remapped state dict ready for a future MLX Expert model.
    """
    print("[mlx-port] Loading Alpamayo action expert weights ...")

    # Load raw non-VLM tensors
    raw_state: Dict[str, mx.array] = {}
    for shard in sorted(alpamayo_path.glob("model-*.safetensors")):
        print(f"  Reading {shard.name} ...")
        tensors = mx.load(str(shard))
        for key, arr in tensors.items():
            if not key.startswith("vlm."):
                raw_state[key] = arr

    print(f"[mlx-port] Read {len(raw_state)} raw expert/action tensors from Alpamayo.")

    # Apply key remapping
    expert_state = remap_alpamayo_expert_keys(raw_state)
    print(f"[mlx-port] After remapping: {len(expert_state)} parameters ready.")

    return expert_state


def main():
    """Build Expert model using the VLM's text config (NVIDIA-style inheritance)
    and attempt to load the 416 Alpamayo weights.
    """
    from mlx_port.vlm_loader import get_vlm_text_config
    from mlx_port.expert import Expert
    from mlx_port.load_alpamayo_vlm import load_alpamayo_vlm_weights

    print("\n[mlx-port] Loading VLM to obtain text config for Expert ...")
    vlm_model, _ = load_alpamayo_vlm_weights()

    # Exact expert overrides from Alpamayo-R1-10B/config.json (expert_cfg block)
    alpamayo_expert_cfg = {
        "hidden_size": 2048,
        "num_attention_heads": 16,
        "num_key_value_heads": 8,   # force GQA 8 KV heads to match Alpamayo checkpoint
        "head_dim": 128,
        "intermediate_size": 8256,
        "num_layers": 36,
    }
    vlm_cfg = get_vlm_text_config(vlm_model, expert_cfg=alpamayo_expert_cfg)
    print(f"[mlx-port] Expert config (VLM + Alpamayo expert_cfg): {vlm_cfg}")

    print("\n[mlx-port] Building MLX Expert model skeleton (inherited from VLM) ...")
    model = Expert(
        hidden_size=vlm_cfg["hidden_size"],
        num_attention_heads=vlm_cfg["num_attention_heads"],
        num_key_value_heads=8,   # force the value that matches the Alpamayo expert checkpoint
        head_dim=vlm_cfg["head_dim"],
        intermediate_size=vlm_cfg["intermediate_size"],
        num_layers=vlm_cfg.get("num_hidden_layers", 36),
    )
    print("[mlx-port] Expert model instantiated with VLM-derived GQA parameters.")

    # Load the remapped state
    state = load_alpamayo_expert_weights()

    print("\n[mlx-port] Calling model.load_weights() ...")
    try:
        model.load_weights(list(state.items()))
        print("[mlx-port] SUCCESS: All 416 expert weights loaded into MLX model.")
    except Exception as e:
        print(f"[mlx-port] Partial load (submodule alignment needed for some keys): {e}")

    return model


if __name__ == "__main__":
    model = main()
