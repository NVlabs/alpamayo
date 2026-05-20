# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""VLM loader for the MLX Alpamayo port.

This module mirrors the token addition and embedding resize logic from
src/alpamayo_r1/models/base_model.py (ReasoningVLAConfig._build_processor
and ReasoningVLA.from_pretrained_submodules) but adapted for mlx-vlm.
"""

from typing import Any, Tuple

import mlx.core as mx
from mlx_vlm import load

from mlx_port.processor import LOCAL_QWEN_PROCESSOR_PATH

# --- Token definitions (copied from NVIDIA src/alpamayo_r1/models/base_model.py) ---

TRAJ_TOKEN = {
    "history": "<|traj_history|>",
    "future": "<|traj_future|>",
    "history_start": "<|traj_history_start|>",
    "future_start": "<|traj_future_start|>",
    "history_end": "<|traj_history_end|>",
    "future_end": "<|traj_future_end|>",
}

SPECIAL_TOKENS_KEYS = [
    "prompt_start",
    "prompt_end",
    "image_start",
    "image_pre_tkn",
    "image_end",
    "traj_history_start",
    "traj_history_pre_tkn",
    "traj_history_end",
    "cot_start",
    "cot_end",
    "meta_action_start",
    "meta_action_end",
    "traj_future_start",
    "traj_future_pre_tkn",
    "traj_future_end",
    "traj_history",
    "traj_future",
    "image_pad",
    "vectorized_wm",
    "vectorized_wm_start",
    "vectorized_wm_end",
    "vectorized_wm_pre_tkn",
    "route_start",
    "route_pad",
    "route_end",
    "question_start",
    "question_end",
    "answer_start",
    "answer_end",
]

SPECIAL_TOKENS = {k: "<|" + k + "|>" for k in SPECIAL_TOKENS_KEYS}

# Default Alpamayo trajectory vocab size (from NVIDIA config)
DEFAULT_TRAJ_VOCAB_SIZE = 768


def _add_alpamayo_tokens(tokenizer: Any, traj_vocab_size: int = DEFAULT_TRAJ_VOCAB_SIZE) -> None:
    """Add Alpamayo discrete trajectory tokens and special tokens to the tokenizer.

    This replicates the logic in ReasoningVLAConfig._build_processor.
    """
    # 1. Discrete trajectory tokens: <i0>, <i1>, ..., <i{traj_vocab_size-1}>
    discrete_tokens = [f"<i{v}>" for v in range(traj_vocab_size)]
    num_new = tokenizer.add_tokens(discrete_tokens)
    assert len(discrete_tokens) == num_new, "Some discrete tokens were not added"

    tokenizer.traj_token_start_idx = tokenizer.convert_tokens_to_ids("<i0>")
    tokenizer.traj_token_end_idx = tokenizer.convert_tokens_to_ids(
        f"<i{traj_vocab_size - 1}>"
    )

    # 2. Special tokens (including traj_history/future markers)
    # We add the full SPECIAL_TOKENS set to match the Alpamayo checkpoint
    special_tokens = list(SPECIAL_TOKENS.values())
    tokenizer.add_tokens(special_tokens, special_tokens=True)

    # Also ensure TRAJ_TOKEN entries are present (they overlap with SPECIAL_TOKENS)
    tokenizer.add_tokens(list(TRAJ_TOKEN.values()), special_tokens=True)

    # Build traj_token_ids mapping for convenience
    tokenizer.traj_token_ids = {
        k: tokenizer.convert_tokens_to_ids(v) for k, v in TRAJ_TOKEN.items()
    }


def _resize_embeddings(model: Any, new_vocab_size: int) -> None:
    """Resize the language model embedding matrix to accommodate new tokens.

    The new rows are initialized to small random values (Xavier-like).
    This mirrors transformers' resize_token_embeddings behavior.
    """
    embed = model.language_model.model.embed_tokens
    old_weight = embed.weight  # shape (old_vocab, hidden_size)
    old_vocab, hidden = old_weight.shape

    if new_vocab_size <= old_vocab:
        return  # nothing to do

    # Create new embedding layer
    new_weight = mx.zeros((new_vocab_size, hidden), dtype=old_weight.dtype)

    # Copy old weights
    new_weight[:old_vocab] = old_weight

    # Initialize new rows with small random values (mean 0, std 0.02)
    new_rows = mx.random.normal(
        shape=(new_vocab_size - old_vocab, hidden),
        dtype=old_weight.dtype,
    ) * 0.02
    new_weight[old_vocab:] = new_rows

    # Replace the embedding weight
    embed.weight = new_weight

    # Update config if present
    if hasattr(model, "config") and hasattr(model.config, "text_config"):
        model.config.text_config.vocab_size = new_vocab_size
    if hasattr(model, "config"):
        model.config.vocab_size = new_vocab_size


def load_vlm_with_alpamayo_tokens(
    model_path: str = LOCAL_QWEN_PROCESSOR_PATH,
    traj_vocab_size: int = DEFAULT_TRAJ_VOCAB_SIZE,
) -> Tuple[Any, Any]:
    """Load the VLM from a local checkpoint and extend it with Alpamayo tokens.

    This is the MLX equivalent of:
        ReasoningVLA.from_pretrained_submodules(...) + token addition

    Steps:
    1. Load processor + tokenizer from the local Qwen3-VL path
    2. Add discrete trajectory tokens (<i0> ... <iN>) and special tokens
    3. Load the mlx-vlm model
    4. Resize the language model embeddings to the new vocab size
    5. Return (model, processor)

    Args:
        model_path: Path to the local Qwen3-VL checkpoint directory.
        traj_vocab_size: Number of discrete trajectory tokens to add (default 768).

    Returns:
        (model, processor) where processor.tokenizer contains the new tokens
        and model has a resized embedding matrix.
    """
    # Load base processor (this also loads the tokenizer)
    from mlx_vlm import load as _mlx_load  # local import to avoid circular issues

    # We load the model first to get the processor, then modify the tokenizer
    model, processor = _mlx_load(model_path)

    tokenizer = processor.tokenizer

    # Record original vocab size before we add tokens
    original_vocab_size = len(tokenizer)

    # Add Alpamayo tokens (discrete + special)
    _add_alpamayo_tokens(tokenizer, traj_vocab_size=traj_vocab_size)

    new_vocab_size = len(tokenizer)

    # Resize embeddings in the loaded model
    if new_vocab_size > original_vocab_size:
        _resize_embeddings(model, new_vocab_size)

    # Expose useful attributes on the tokenizer for downstream code
    tokenizer.original_vocab_size = original_vocab_size
    # Note: we cannot set tokenizer.vocab_size directly (it's a read-only property),
    # so we store the effective size under a custom attribute.
    tokenizer.effective_vocab_size = new_vocab_size

    return model, processor
