# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``alpamayo_r1.helper``.

Run:
    pytest src/alpamayo_r1/test_helper.py -v
"""

from __future__ import annotations

import torch

from alpamayo_r1.helper import to_device


def test_to_device_no_dtype_preserves_int_and_float_tensors() -> None:
    """Default behavior (dtype=None) must move tensors without changing dtype."""
    payload = {
        "input_ids": torch.arange(4, dtype=torch.long),
        "image": torch.randn(2, 3, dtype=torch.float32),
    }

    moved = to_device(payload, device="cpu")

    assert moved["input_ids"].dtype == torch.long
    assert moved["image"].dtype == torch.float32


def test_to_device_dtype_preserves_integer_tensors() -> None:
    """Regression for issue #36: dtype must not be applied to int tensors."""
    payload = {
        "input_ids": torch.arange(4, dtype=torch.long),
        "attention_mask": torch.ones(4, dtype=torch.long),
        "pixel_values": torch.randn(2, 3, dtype=torch.float32),
    }

    moved = to_device(payload, device="cpu", dtype=torch.bfloat16)

    assert moved["input_ids"].dtype == torch.long
    assert moved["attention_mask"].dtype == torch.long
    assert moved["pixel_values"].dtype == torch.bfloat16


def test_to_device_dtype_preserves_bool_tensors() -> None:
    """Boolean tensors must keep their dtype even when a float dtype is requested."""
    mask = torch.tensor([True, False, True])

    moved = to_device(mask, device="cpu", dtype=torch.bfloat16)

    assert moved.dtype == torch.bool


def test_to_device_recurses_into_nested_structures() -> None:
    """Nested mappings and sequences must be walked recursively."""
    payload = {
        "tokens": {"input_ids": torch.arange(3, dtype=torch.long)},
        "frames": [torch.randn(2, dtype=torch.float32) for _ in range(2)],
    }

    moved = to_device(payload, device="cpu", dtype=torch.bfloat16)

    assert moved["tokens"]["input_ids"].dtype == torch.long
    assert all(t.dtype == torch.bfloat16 for t in moved["frames"])


def test_to_device_passes_through_non_tensor_values() -> None:
    """Strings, ints, and other non-tensor leaves must be returned unchanged."""
    payload = {"name": "ego", "id": 7, "tensor": torch.zeros(1)}

    moved = to_device(payload, device="cpu")

    assert moved["name"] == "ego"
    assert moved["id"] == 7
    assert torch.equal(moved["tensor"], torch.zeros(1))
