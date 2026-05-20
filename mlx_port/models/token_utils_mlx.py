# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""MLX-native trajectory token utilities.

This module ports the core token fusion logic from
src/alpamayo_r1/models/base_model.py (TrajectoryFusionMixin + helpers)
so that history trajectory tokens can be fused into input_ids for the VLM.
"""

from typing import Any

import mlx.core as mx
import numpy as np


def replace_pad_token(input_ids: mx.array, new_ids: mx.array, pad_idx: int) -> mx.array:
    """Replace pad tokens in input_ids with new token values (MLX version).

    Args:
        input_ids: [B, seq_len] token ids
        new_ids: [B, n_traj_tokens] token ids to insert
        pad_idx: the pad token id to replace

    Returns:
        input_ids with pad tokens replaced by trajectory tokens.
    """
    # Convert to numpy for masked_scatter equivalent
    ids_np = np.array(input_ids)
    new_np = np.array(new_ids)

    mask = ids_np == pad_idx

    # Flatten and replace
    flat = ids_np.flatten()
    flat_mask = mask.flatten()

    # We need to replace exactly as many pads as we have new_ids
    # This assumes the number of pads matches the number of new tokens
    replacement_idx = 0
    for i in range(len(flat)):
        if flat_mask[i]:
            flat[i] = new_np.flatten()[replacement_idx]
            replacement_idx += 1
            if replacement_idx >= len(new_np.flatten()):
                break

    return mx.array(flat.reshape(ids_np.shape))


def tokenize_history_trajectory(
    tokenizer: Any,
    traj_data: dict[str, Any],
    start_idx: int = 0,
) -> mx.array:
    """Tokenize history trajectory into discrete tokens.

    This is a strict MLX port of the NVIDIA function.
    The tokenizer **must** have an `encode(hist_xyz, hist_rot, fut_xyz, fut_rot)` method
    (provided by NVIDIA's DeltaTrajectoryTokenizer / DiscreteTrajectoryTokenizer).
    No fallback is supported.

    Args:
        tokenizer: Trajectory tokenizer with .encode(hist_xyz, hist_rot, fut_xyz, fut_rot)
        traj_data: dict with "ego_history_xyz" and "ego_history_rot"
        start_idx: offset to add to token indices

    Returns:
        mx.array of shape [B, n_traj_tokens]
    """
    assert "ego_history_xyz" in traj_data
    assert traj_data["ego_history_xyz"].ndim == 4

    hist_xyz = traj_data["ego_history_xyz"]
    hist_rot = traj_data["ego_history_rot"]

    # Flatten batch and traj dimensions for the tokenizer
    B, n_traj, T, _ = hist_xyz.shape
    hist_xyz_flat = hist_xyz.reshape(B * n_traj, T, 3)
    hist_rot_flat = hist_rot.reshape(B * n_traj, T, 3, 3)

    # NVIDIA's implementation requires the tokenizer to have an .encode() method
    # (DeltaTrajectoryTokenizer or DiscreteTrajectoryTokenizer).
    # We enforce this strictly for parity.
    if not hasattr(tokenizer, "encode"):
        raise AttributeError(
            "hist_traj_tokenizer must have an 'encode(hist_xyz, hist_rot, fut_xyz, fut_rot)' method. "
            "This is provided by NVIDIA's DeltaTrajectoryTokenizer / DiscreteTrajectoryTokenizer. "
            "No fallback is supported for strict parity with the original implementation."
        )

    token_ids = tokenizer.encode(
        hist_xyz=hist_xyz_flat[:, :1],   # first position for history
        hist_rot=hist_rot_flat[:, :1],
        fut_xyz=hist_xyz_flat,           # history is passed as "future" for encoding
        fut_rot=hist_rot_flat,
    )

    # Add start offset
    token_ids = token_ids + start_idx

    # Reshape back to [B, n_traj * tokens_per_traj]
    token_ids = token_ids.reshape(B, -1)
    return mx.array(token_ids)


def fuse_traj_tokens(
    input_ids: mx.array,
    traj_data: dict[str, Any] | None,
    hist_traj_tokenizer: Any,
    hist_token_start_idx: int,
    traj_token_ids: dict[str, int],
) -> mx.array:
    """Fuse history trajectory tokens into the input_ids.

    This is the MLX equivalent of TrajectoryFusionMixin.fuse_traj_tokens.

    Args:
        input_ids: [B, seq_len] token ids (may contain image_pad tokens)
        traj_data: dict containing ego_history_xyz / ego_history_rot
        hist_traj_tokenizer: tokenizer with .encode() (Delta/DiscreteTrajectoryTokenizer)
        hist_token_start_idx: starting index for history trajectory tokens
        traj_token_ids: mapping {"history": pad_token_id, ...}

    Returns:
        input_ids with history trajectory tokens fused in place of pads.
    """
    if (
        traj_data is None
        or traj_data.get("ego_history_xyz") is None
        or traj_data.get("ego_history_rot") is None
    ):
        return input_ids

    # Tokenize history
    hist_idx = tokenize_history_trajectory(
        hist_traj_tokenizer, traj_data, hist_token_start_idx
    )

    # Replace the special history pad token with the generated indices
    pad_idx = traj_token_ids.get("history", -1)
    if pad_idx >= 0:
        input_ids = replace_pad_token(input_ids, hist_idx, pad_idx)

    return input_ids


# ------------------------------------------------------------------
# MLX-native generation helpers (Row 7)
# ------------------------------------------------------------------


class ExpertLogitsProcessor:
    """MLX port of ExpertLogitsProcessor.

    Masks out logits for the discrete trajectory tokens so that the VLM
    text generation is not polluted by trajectory-token logits.
    Accepts either a contiguous range (legacy) or an exact list of token IDs
    (robust after tokenizer.add_tokens appends missing <iN> tokens).
    """

    def __init__(self, traj_token_offset: int = None, traj_vocab_size: int = None, traj_token_ids: list[int] = None):
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size
        self.traj_token_ids = traj_token_ids or []

    def __call__(self, input_ids, scores: mx.array) -> mx.array:
        """Mask trajectory token logits to -inf (input_ids ignored, as in NVIDIA)."""
        if self.traj_token_ids:
            # Exact per-ID mask (handles non-contiguous IDs)
            for tid in self.traj_token_ids:
                if 0 <= tid < scores.shape[1]:
                    scores[:, tid] = float("-inf")
        else:
            # Fallback contiguous range
            vocab_size = scores.shape[1]
            start = max(0, self.traj_token_offset or 0)
            end = min(vocab_size, (self.traj_token_offset or 0) + (self.traj_vocab_size or 0))
            if start < end:
                scores[:, start:end] = float("-inf")
        return scores


class StopAfterEOS:
    """MLX port of StopAfterEOS.

    Stops generation after one additional token following the first
    occurrence of the eos_token_id (traj_future_start).
    """

    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id
        self.eos_found = None

    def __call__(self, input_ids: mx.array, scores: mx.array = None, **kwargs) -> bool:
        # Handle scalar integer input (used by mlx_vlm's internal fallback)
        if isinstance(input_ids, (int, np.integer)):
            input_ids = mx.array([[input_ids]])

        batch_size = input_ids.shape[0]

        if self.eos_found is None:
            self.eos_found = mx.zeros((batch_size,), dtype=mx.bool_)

        if mx.all(self.eos_found):
            return True

        last_tokens = input_ids[:, -1]
        current_has_eos = last_tokens == self.eos_token_id
        self.eos_found = self.eos_found | current_has_eos
        return False


def replace_padding_after_eos(
    token_ids: mx.array,
    eos_token_id: int,
    pad_token_id: int,
) -> mx.array:
    """MLX port of replace_padding_after_eos.

    Overwrites every token after the first EOS with pad_token_id.
    """
    result = token_ids
    for b in range(token_ids.shape[0]):
        seq = token_ids[b]
        # Find first occurrence of eos
        matches = (seq == eos_token_id).astype(mx.float32)
        if mx.any(matches):
            first = int(mx.argmax(matches))
            if first + 1 < seq.shape[0]:
                result[b, first + 1 :] = pad_token_id
    return result


# ------------------------------------------------------------------
# CoC / text extraction helpers (Row 8 support, used by inference)
# ------------------------------------------------------------------


def extract_between_special_tokens(decoded_texts: list[str], tag: str) -> list[str]:
    """Extract text between <|tag_start|> and <|tag_end|> tokens (e.g. <|cot_start|>, <|cot_end|>).

    Matches the NVIDIA implementation in src/alpamayo_r1/models/token_utils.py.
    """
    results = []
    start_token = f"<|{tag}_start|>"
    end_token = f"<|{tag}_end|>"
    for text in decoded_texts:
        if start_token in text and end_token in text:
            start_idx = text.find(start_token) + len(start_token)
            end_idx = text.find(end_token)
            results.append(text[start_idx:end_idx].strip())
        else:
            # Fallback: return whole text if markers not found (common in early rollout)
            results.append(text.strip())
    return results


def extract_text_tokens(tokenizer: Any, output_tokens: mx.array) -> dict[str, list[str]]:
    """MLX port of extract_text_tokens.

    Decodes the generated sequences and extracts 'cot', 'meta_action', 'answer'.
    Falls back to empty strings if the tokenizer does not support batch_decode
    (common in early structural tests).
    """
    if not hasattr(tokenizer, "batch_decode"):
        # Minimal tokenizer in test environment – return placeholder CoC
        return {"cot": ["(CoC extraction requires full tokenizer)"], "meta_action": [""], "answer": [""]}

    token_lists = output_tokens.tolist()
    decoded_batch = tokenizer.batch_decode(token_lists, skip_special_tokens=False)

    extract_tokens = ["cot", "meta_action", "answer"]
    extracted_text = {}
    for token in extract_tokens:
        extracted_text[token] = extract_between_special_tokens(decoded_batch, token)
    return extracted_text