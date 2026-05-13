"""
Smoke tests for the Action Expert component of Alpamayo.
"""

import pytest

from mlx_port.expert_loader import load_alpamayo_expert_weights
from mlx_port.expert import Expert


def test_expert_load_weights():
    """Expert model should load all 416 Alpamayo tensors without error."""
    state = load_alpamayo_expert_weights()

    model = Expert(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        intermediate_size=8256,
        num_layers=36,
    )

    # This should not raise
    model.load_weights(list(state.items()))

    # Success criterion for smoke test: no exception and model has layers
    assert len(model.layers) == 36
