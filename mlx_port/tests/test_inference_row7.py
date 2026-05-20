"""Structural tests for Row 7 inference components."""

import mlx.core as mx

from mlx_port.models.token_utils_mlx import (
    ExpertLogitsProcessor,
    StopAfterEOS,
    replace_padding_after_eos,
)


def test_expert_logits_processor_masks_traj_tokens():
    proc = ExpertLogitsProcessor(traj_token_offset=100, traj_vocab_size=50)
    scores = mx.zeros((2, 200))
    out = proc(mx.array([[1, 2]]), scores)
    # The masked region should be -inf
    assert mx.all(out[0, 100:150] == float("-inf"))
    assert out[0, 99] == 0.0
    assert out[0, 150] == 0.0


def test_stop_after_eos_stops_one_token_after():
    stop = StopAfterEOS(eos_token_id=42)
    ids = mx.array([[1, 2, 42], [3, 4, 5]])
    stop(ids, None)
    # After first call with EOS in row 0, eos_found should be set
    assert stop.eos_found[0] == True
    assert stop.eos_found[1] == False


def test_replace_padding_after_eos():
    tokens = mx.array([[1, 2, 42, 99, 100], [3, 42, 5, 6, 7]])
    out = replace_padding_after_eos(tokens, eos_token_id=42, pad_token_id=0)
    assert out[0, 3] == 0 and out[0, 4] == 0
    assert out[1, 2] == 0 and out[1, 3] == 0 and out[1, 4] == 0