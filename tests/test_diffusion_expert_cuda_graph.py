# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU tests for exact-shape diffusion-expert CUDA graph replay."""

import gc
from threading import Event, Thread
from types import SimpleNamespace
from typing import Any
from weakref import ref

import pytest
import torch
from transformers.cache_utils import Cache, DynamicLayer
from transformers.modeling_outputs import BaseModelOutputWithPast

from alpamayo_r1.models.diffusion_expert_cuda_graph import (
    enable_diffusion_expert_cuda_graph,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")


class _TinyExpertLayer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = SimpleNamespace(head_dim=2)


class _TinyExpert(torch.nn.Module):
    """Return outputs influenced by every copied graph input."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones((), device="cuda", dtype=torch.bfloat16))
        self.config = SimpleNamespace(num_key_value_heads=1, hidden_size=4)
        self.layers = torch.nn.ModuleList([_TinyExpertLayer()])

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> BaseModelOutputWithPast:
        del input_ids, use_cache, cache_position, kwargs
        assert attention_mask is not None
        assert position_ids is not None
        assert past_key_values is not None
        assert inputs_embeds is not None
        prompt = past_key_values.layers[0].keys[:, 0, :, 0].float().mean(dim=1)
        hidden_states = (
            inputs_embeds
            + prompt[:, None, None]
            + position_ids[0, :, :, None]
            + attention_mask[:, 0, :, :1]
        ) * self.weight
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )


class _CaptureCoordinatedExpert(_TinyExpert):
    def __init__(self, capture_started: Event, other_thread_done: Event) -> None:
        super().__init__()
        self._capture_started = capture_started
        self._other_thread_done = other_thread_done

    def forward(self, *args: Any, **kwargs: Any) -> BaseModelOutputWithPast:
        if torch.cuda.is_current_stream_capturing():
            self._capture_started.set()
            if not self._other_thread_done.wait(timeout=5):
                raise TimeoutError("Concurrent CUDA allocation did not finish")
        return super().forward(*args, **kwargs)


def _prompt_cache(batch_size: int, prompt_length: int, value: float) -> Cache:
    layer = DynamicLayer()
    shape = (batch_size, 1, prompt_length, 2)
    keys = torch.full(shape, value, device="cuda", dtype=torch.bfloat16)
    values = torch.full(shape, -value, device="cuda", dtype=torch.bfloat16)
    layer.update(keys, values)
    return Cache(layers=[layer])


def _expert_inputs(
    batch_size: int,
    sequence_length: int,
    prompt_length: int,
    value: float,
    input_dtype: torch.dtype = torch.float32,
) -> dict[str, Any]:
    return {
        "inputs_embeds": torch.full(
            (batch_size, sequence_length, 4),
            value,
            device="cuda",
            dtype=input_dtype,
        ),
        "position_ids": (
            torch.arange(prompt_length, prompt_length + sequence_length, device="cuda")
            .view(1, 1, -1)
            .expand(3, batch_size, -1)
        ),
        "past_key_values": _prompt_cache(batch_size, prompt_length, value + 1),
        "attention_mask": torch.full(
            (batch_size, 1, sequence_length, prompt_length + sequence_length),
            value + 2,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        "use_cache": True,
        "is_causal": False,
    }


def test_cuda_graph_preserves_dynamic_batches_and_prompt_lengths() -> None:
    expert = _TinyExpert().eval()
    original_forward = expert.forward
    runner = enable_diffusion_expert_cuda_graph(
        expert,
        max_batch_size=4,
        max_graphs=4,
    )

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for batch_size, prompt_length, value in (
            (2, 5, 1.0),
            (2, 5, 3.0),
            (2, 6, 5.0),
            (1, 5, 7.0),
        ):
            inputs = _expert_inputs(batch_size, 3, prompt_length, value)
            expected = original_forward(**inputs)
            with runner.sampling():
                actual = expert(**inputs)
            torch.cuda.synchronize()

            assert actual.past_key_values is inputs["past_key_values"]
            torch.testing.assert_close(
                actual.last_hidden_state,
                expected.last_hidden_state,
                rtol=0,
                atol=0,
            )

        expert.weight.fill_(2)
        inputs = _expert_inputs(2, 3, 6, 9.0)
        expected = original_forward(**inputs)
        with runner.sampling():
            actual = expert(**inputs)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state)

    assert runner.stats == {
        "captures": 3,
        "replays": 5,
        "eager_fallbacks": 0,
        "graphs": 3,
    }


def test_cuda_graph_falls_back_when_signature_cache_is_full() -> None:
    expert = _TinyExpert().eval()
    original_forward = expert.forward
    runner = enable_diffusion_expert_cuda_graph(
        expert,
        max_batch_size=2,
        max_graphs=1,
    )

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        with runner.sampling():
            expert(**_expert_inputs(2, 3, 5, 1.0))
        inputs = _expert_inputs(2, 3, 6, 2.0)
        expected = original_forward(**inputs)
        with runner.sampling():
            actual = expert(**inputs)

    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state)
    assert runner.stats["graphs"] == 1
    assert runner.stats["eager_fallbacks"] == 1


def test_cuda_graph_falls_back_without_explicit_non_causal_attention() -> None:
    expert = _TinyExpert().eval()
    original_forward = expert.forward
    runner = enable_diffusion_expert_cuda_graph(expert, max_batch_size=1)
    inputs = _expert_inputs(1, 3, 5, 1.0)
    del inputs["is_causal"]

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        expected = original_forward(**inputs)
        with runner.sampling():
            actual = expert(**inputs)

    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state)
    assert runner.stats == {
        "captures": 0,
        "replays": 0,
        "eager_fallbacks": 1,
        "graphs": 0,
    }


def test_cuda_graph_rejects_batches_above_capacity() -> None:
    expert = _TinyExpert().eval()
    enable_diffusion_expert_cuda_graph(expert, max_batch_size=2)

    with torch.inference_mode(), pytest.raises(ValueError, match="exceeds configured maximum 2"):
        expert(**_expert_inputs(3, 3, 5, 1.0))


def test_cuda_graph_does_not_retain_request_prompt_cache() -> None:
    expert = _TinyExpert().eval()
    runner = enable_diffusion_expert_cuda_graph(expert, max_batch_size=1)
    inputs = _expert_inputs(1, 3, 5, 1.0)
    cache_reference = ref(inputs["past_key_values"])

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        with runner.sampling():
            output = expert(**inputs)
        torch.cuda.synchronize()

    del inputs, output
    gc.collect()
    assert cache_reference() is None


def test_cuda_graph_replays_outside_initial_inference_mode() -> None:
    expert = _TinyExpert().eval()
    original_forward = expert.forward
    runner = enable_diffusion_expert_cuda_graph(expert, max_batch_size=1)

    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        runner.sampling(),
    ):
        expert(**_expert_inputs(1, 3, 5, 1.0))

    with (
        torch.no_grad(),
        torch.autocast("cuda", dtype=torch.bfloat16),
        runner.sampling(),
    ):
        inputs = _expert_inputs(1, 3, 5, 3.0)
        expected = original_forward(**inputs)
        actual = expert(**inputs)

    torch.cuda.synchronize()
    torch.testing.assert_close(actual.last_hidden_state, expected.last_hidden_state)
    assert runner.stats == {
        "captures": 1,
        "replays": 2,
        "eager_fallbacks": 0,
        "graphs": 1,
    }


def test_cuda_graph_allows_other_thread_cuda_work_during_capture() -> None:
    capture_started = Event()
    other_thread_done = Event()
    errors: list[BaseException] = []

    def allocate_from_other_thread() -> None:
        try:
            if not capture_started.wait(timeout=5):
                raise TimeoutError("Graph capture did not start")
            torch.ones(1, device="cuda")
        except (RuntimeError, TimeoutError) as error:
            errors.append(error)
        finally:
            other_thread_done.set()

    thread = Thread(target=allocate_from_other_thread)
    thread.start()
    expert = _CaptureCoordinatedExpert(capture_started, other_thread_done).eval()
    runner = enable_diffusion_expert_cuda_graph(expert, max_batch_size=1)

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        with runner.sampling():
            output = expert(**_expert_inputs(1, 3, 5, 1.0))
        torch.cuda.synchronize()

    thread.join(timeout=5)
    assert not thread.is_alive()
    assert errors == []
    assert output.last_hidden_state.shape == (1, 3, 4)
