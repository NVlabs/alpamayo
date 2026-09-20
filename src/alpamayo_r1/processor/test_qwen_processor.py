# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``alpamayo_r1.processor.qwen_processor`` helpers.

Run:
    pytest src/alpamayo_r1/processor/test_qwen_processor.py -v
"""

from __future__ import annotations

import torch

from alpamayo_r1.processor.qwen_processor import basic_collation_fn


def test_basic_collation_fn_default_unstackable_keys_is_safe() -> None:
    """Default ``unstackable_keys`` must behave as 'no extra unstackable keys'.

    Regression for the previous mutable-list default. Any caller that omits
    ``unstackable_keys`` should get a clean stack of every tensor key.
    """
    batch = [
        {"a": torch.zeros(3), "b": torch.ones(2)},
        {"a": torch.zeros(3), "b": torch.ones(2)},
    ]

    out = basic_collation_fn(batch)

    assert isinstance(out["a"], torch.Tensor)
    assert isinstance(out["b"], torch.Tensor)
    assert out["a"].shape == (2, 3)
    assert out["b"].shape == (2, 2)


def test_basic_collation_fn_explicit_none_unstackable_keys() -> None:
    """Passing ``unstackable_keys=None`` explicitly is equivalent to omitting it."""
    batch = [
        {"a": torch.zeros(3)},
        {"a": torch.zeros(3)},
    ]

    out = basic_collation_fn(batch, unstackable_keys=None)

    assert out["a"].shape == (2, 3)


def test_basic_collation_fn_marks_keys_as_unstackable() -> None:
    """Keys named in ``unstackable_keys`` must be returned as a plain list."""
    batch = [
        {"a": torch.zeros(3), "image_frames": torch.zeros(1, 2, 3)},
        {"a": torch.zeros(3), "image_frames": torch.zeros(1, 2, 3)},
    ]

    out = basic_collation_fn(batch, unstackable_keys=["image_frames"])

    assert isinstance(out["a"], torch.Tensor)
    assert out["a"].shape == (2, 3)
    assert isinstance(out["image_frames"], list)
    assert len(out["image_frames"]) == 2


def test_basic_collation_fn_default_does_not_leak_state_between_calls() -> None:
    """Repeated invocations with the default must not accumulate state.

    The original ``unstackable_keys=[]`` default was a mutable list; this test
    pins down that switching to ``None`` (or any sentinel) does not regress
    that behavior even if a future edit accidentally mutates the default.
    """
    batch = [{"a": torch.zeros(3)}, {"a": torch.zeros(3)}]

    out_first = basic_collation_fn(batch)
    out_second = basic_collation_fn(batch)

    assert out_first["a"].shape == (2, 3)
    assert out_second["a"].shape == (2, 3)


def test_basic_collation_fn_non_tensor_values_become_lists() -> None:
    """Non-tensor values must always be returned as lists, never stacked."""
    batch = [
        {"id": "a", "x": torch.zeros(2)},
        {"id": "b", "x": torch.ones(2)},
    ]

    out = basic_collation_fn(batch)

    assert out["id"] == ["a", "b"]
    assert isinstance(out["x"], torch.Tensor)
    assert out["x"].shape == (2, 2)
