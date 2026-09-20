# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``alpamayo_r1.utils.get_label_mask``.

Run:
    pytest src/alpamayo_r1/utils/test_get_label_mask.py -v
"""

from __future__ import annotations

import torch

from alpamayo_r1.utils.get_label_mask import get_assistant_mask


class _StubTokenizer:
    """Minimal tokenizer that satisfies ``convert_tokens_to_ids`` for tests."""

    def __init__(self, mapping: dict[str, int]) -> None:
        self._mapping = mapping

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._mapping[token]


def _make_tokens() -> tuple[_StubTokenizer, list[int]]:
    """Build a tokenizer and a token sequence with one assistant turn.

    Layout: [BOS, USER, content, EOS, BOS, ASSISTANT, content, content, EOS]
    The assistant turn starts at index 4 and ends at index 8.
    """
    tokenizer = _StubTokenizer(
        {
            "<|im_start|>": 100,
            "<|im_end|>": 101,
            "user": 200,
            "assistant": 201,
        }
    )
    # Indices: 0    1   2  3   4    5   6  7  8
    tokens = [100, 200, 1, 101, 100, 201, 2, 3, 101]
    return tokenizer, tokens


def test_get_assistant_mask_returns_tensor_for_tensor_input() -> None:
    tokenizer, tokens = _make_tokens()

    mask = get_assistant_mask(tokenizer, torch.tensor(tokens))

    assert isinstance(mask, torch.Tensor)
    assert mask.dtype == torch.bool
    assert mask.shape == (len(tokens),)


def test_get_assistant_mask_returns_list_for_list_input() -> None:
    """When ``tokens`` is a ``list[int]`` the function returns ``list[bool]``."""
    tokenizer, tokens = _make_tokens()

    mask = get_assistant_mask(tokenizer, tokens)

    assert isinstance(mask, list)
    assert all(isinstance(v, bool) for v in mask)
    assert len(mask) == len(tokens)


def test_get_assistant_mask_marks_only_assistant_span_inclusive_of_eos() -> None:
    """Mask must cover the assistant content + the trailing EOS, per the body's
    ``masks[start + 3 : end + 1] = True`` rule.

    For the layout above (BOS=4, ASSISTANT=5, content=6,7, EOS=8) the span
    `start + 3 = 7` through `end + 1 = 9` should be True. Index 8 (the EOS)
    is included; the user turn (indices 0-3) should be False throughout.
    """
    tokenizer, tokens = _make_tokens()

    mask = get_assistant_mask(tokenizer, torch.tensor(tokens)).tolist()

    # User turn never marked.
    assert mask[0:4] == [False, False, False, False]
    # Assistant header (BOS + role token + newline-equivalent) is skipped.
    assert mask[4:7] == [False, False, False]
    # Assistant content + EOS is marked.
    assert mask[7] is True
    assert mask[8] is True


def test_get_assistant_mask_tensor_and_list_inputs_agree() -> None:
    """Tensor and list inputs must produce identical boolean values."""
    tokenizer, tokens = _make_tokens()

    tensor_mask = get_assistant_mask(tokenizer, torch.tensor(tokens))
    list_mask = get_assistant_mask(tokenizer, tokens)

    assert tensor_mask.tolist() == list_mask
