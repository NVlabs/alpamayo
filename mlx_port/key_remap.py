"""
Key remapping utilities for loading Alpamayo VLM weights into mlx-vlm.

Alpamayo stores VLM weights under the "vlm." prefix with certain submodule names.
mlx-vlm uses a slightly different hierarchy.

This module provides a clean remapping so the fine-tuned Alpamayo VLM can be used
without waiting for a full conversion pipeline.
"""

from typing import Dict

import mlx.core as mx


def remap_alpamayo_vlm_keys(state: Dict[str, mx.array]) -> Dict[str, mx.array]:
    """
    Convert Alpamayo VLM weight keys to the names expected by mlx-vlm.

    Alpamayo (HF) stores:
        vlm.model.language_model.*   (decoder directly under language_model.layers)
        vlm.model.visual.*

    mlx-vlm (Qwen3-VL) stores:
        language_model.model.*       (decoder under language_model.model.layers)
        vision_tower.*
    """
    remapped = {}
    for key, value in state.items():
        new_key = key

        # 1. Strip leading "vlm."
        if new_key.startswith("vlm."):
            new_key = new_key[4:]

        # 2. Strip leading "model." (Alpamayo wraps everything under model.)
        if new_key.startswith("model."):
            new_key = new_key[6:]

        # 3. Map visual.* → vision_tower.*
        if new_key.startswith("visual."):
            new_key = "vision_tower." + new_key[len("visual."):]

        # 4. Critical: Alpamayo puts the decoder at language_model.layers,
        #    but mlx-vlm expects language_model.model.layers for Qwen3-VL.
        if new_key.startswith("language_model.layers"):
            new_key = new_key.replace("language_model.layers", "language_model.model.layers", 1)

        # 5. Top-level language model components (embed_tokens, final norm, lm_head)
        #    Alpamayo stores them directly under language_model.*
        #    mlx-vlm expects them under language_model.model.* or language_model.lm_head
        if new_key == "language_model.embed_tokens.weight":
            new_key = "language_model.model.embed_tokens.weight"
        if new_key == "language_model.norm.weight":
            new_key = "language_model.model.norm.weight"
        if new_key == "lm_head.weight":
            new_key = "language_model.lm_head.weight"

        remapped[new_key] = value

    # Post-processing: fix known shape differences for Qwen3-VL vision weights
    # The patch embedding weight has a different layout in Alpamayo vs mlx-vlm
    if "vision_tower.patch_embed.proj.weight" in remapped:
        w = remapped["vision_tower.patch_embed.proj.weight"]
        # Alpamayo gives (1152, 3, 2, 16, 16)
        # mlx-vlm expects (1152, 2, 16, 16, 3)
        # Permutation: move axis 1 to the end, axis 2 to position 1
        if w.shape == (1152, 3, 2, 16, 16):
            remapped["vision_tower.patch_embed.proj.weight"] = w.transpose(0, 2, 3, 4, 1)

    return remapped


def load_alpamayo_vlm_state(alpamayo_dir: str) -> Dict[str, mx.array]:
    """
    Load all VLM weights from an Alpamayo-R1 checkpoint directory
    and return them already remapped for mlx-vlm.
    """
    from pathlib import Path

    alpamayo_path = Path(alpamayo_dir)
    state = {}

    for shard in sorted(alpamayo_path.glob("model-*.safetensors")):
        tensors = mx.load(str(shard))
        for key, arr in tensors.items():
            if key.startswith("vlm."):
                state[key] = arr

    return remap_alpamayo_vlm_keys(state)