# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exact-shape CUDA graph replay for the diffusion action expert."""

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from threading import RLock
from typing import Any
from weakref import ReferenceType, ref

import torch
from transformers.cache_utils import Cache, CacheLayerMixin

logger = logging.getLogger(__name__)


class _ReadOnlyPromptCacheLayer(CacheLayerMixin):
    """Expose prompt K/V plus current action K/V without mutating the prompt."""

    is_compileable = True

    def __init__(self, keys: torch.Tensor, values: torch.Tensor) -> None:
        super().__init__()
        self.keys = keys
        self.values = values
        self.is_initialized = True

    def lazy_initialization(self, key_states: torch.Tensor) -> None:
        del key_states
        raise RuntimeError("Read-only prompt cache layers must be initialized")

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cache_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del cache_kwargs
        return (
            torch.cat((self.keys, key_states), dim=-2),
            torch.cat((self.values, value_states), dim=-2),
        )

    def get_mask_sizes(self, cache_position: torch.Tensor) -> tuple[int, int]:
        return self.get_seq_length() + cache_position.shape[0], 0

    def get_seq_length(self) -> int:
        return self.keys.shape[-2]

    def get_max_cache_shape(self) -> int:
        return -1

    def crop(self, max_length: int) -> None:
        if max_length < 0:
            max_length = self.get_seq_length() - abs(max_length)
        if max_length < self.get_seq_length():
            raise ValueError("Cannot crop a read-only expert prompt cache")


@dataclass
class _GraphEntry:
    """Static graph state for one exact expert input signature."""

    graph: torch.cuda.CUDAGraph
    output: Any
    inputs_embeds: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    prompt_keys: list[torch.Tensor]
    prompt_values: list[torch.Tensor]
    loaded_prompt_cache: ReferenceType[Cache] | None = None

    def replay(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_cache: Cache,
    ) -> Any:
        """Copy changing inputs into static buffers and replay the graph."""
        self.inputs_embeds.copy_(inputs_embeds)
        self.position_ids.copy_(position_ids)
        self.attention_mask.copy_(attention_mask)

        if self.loaded_prompt_cache is None or self.loaded_prompt_cache() is not prompt_cache:
            for layer_index, prompt_layer in enumerate(prompt_cache.layers):
                if prompt_layer.keys is None or prompt_layer.values is None:
                    raise ValueError("Expert prompt cache contains an uninitialized layer")
                self.prompt_keys[layer_index].copy_(prompt_layer.keys)
                self.prompt_values[layer_index].copy_(prompt_layer.values)
            self.loaded_prompt_cache = ref(prompt_cache)

        self.graph.replay()
        output = copy(self.output)
        output.past_key_values = prompt_cache
        return output


class DiffusionExpertCudaGraph:
    """Replace one expert's inference forward with bounded exact CUDA graphs."""

    def __init__(
        self,
        expert: torch.nn.Module,
        *,
        max_batch_size: int,
        max_graphs: int,
    ) -> None:
        if max_batch_size <= 0:
            raise ValueError("Diffusion expert CUDA graph maximum batch size must be positive")
        if max_graphs <= 0:
            raise ValueError("Diffusion expert CUDA graph maximum graph count must be positive")

        parameter = next(expert.parameters())
        if parameter.device.type != "cuda":
            raise ValueError("Diffusion expert CUDA graphs require a CUDA model")
        if expert.training:
            raise ValueError("Diffusion expert CUDA graphs require an eval-mode model")

        self._expert = expert
        self._original_forward = expert.forward
        self._max_batch_size = max_batch_size
        self._max_graphs = max_graphs
        self._graphs: dict[tuple[Any, ...], _GraphEntry] = {}
        self._lock = RLock()
        self._captures = 0
        self._replays = 0
        self._eager_fallbacks = 0

    @property
    def stats(self) -> dict[str, int]:
        """Return counters suitable for benchmark logging."""
        return {
            "captures": self._captures,
            "replays": self._replays,
            "eager_fallbacks": self._eager_fallbacks,
            "graphs": len(self._graphs),
        }

    @contextmanager
    def sampling(self) -> Iterator[None]:
        """Serialize one complete denoising operation over shared graph buffers."""
        with self._lock:
            for entry in self._graphs.values():
                entry.loaded_prompt_cache = None
            yield

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
    ) -> Any:
        """Replay supported inference calls and preserve eager behavior otherwise."""
        if (
            torch.is_grad_enabled()
            or self._expert.training
            or input_ids is not None
            or inputs_embeds is None
            or inputs_embeds.ndim != 3
            or attention_mask is None
            or position_ids is None
            or past_key_values is None
            or not past_key_values.layers
            or cache_position is not None
            or use_cache is False
            or kwargs.get("is_causal") is not False
            or set(kwargs) != {"is_causal"}
        ):
            self._eager_fallbacks += 1
            return self._original_forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                cache_position=cache_position,
                **kwargs,
            )

        batch_size = inputs_embeds.shape[0]
        if batch_size <= 0 or batch_size > self._max_batch_size:
            raise ValueError(
                f"Diffusion expert batch size {batch_size} exceeds configured maximum "
                f"{self._max_batch_size}"
            )

        signature = self._signature(
            inputs_embeds,
            position_ids,
            attention_mask,
            past_key_values,
        )
        with self._lock:
            entry = self._graphs.get(signature)
            if entry is None:
                if len(self._graphs) >= self._max_graphs:
                    self._eager_fallbacks += 1
                    return self._original_forward(
                        inputs_embeds=inputs_embeds,
                        position_ids=position_ids,
                        past_key_values=past_key_values,
                        attention_mask=attention_mask,
                        use_cache=use_cache,
                        **kwargs,
                    )
                entry = self._capture(
                    inputs_embeds,
                    position_ids,
                    attention_mask,
                    past_key_values,
                )
                self._graphs[signature] = entry
                self._captures += 1

            self._replays += 1
            return entry.replay(
                inputs_embeds,
                position_ids,
                attention_mask,
                past_key_values,
            )

    def _signature(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_cache: Cache,
    ) -> tuple[Any, ...]:
        def tensor_signature(tensor: torch.Tensor) -> tuple[Any, ...]:
            return tuple(tensor.shape), tensor.dtype, tensor.device

        cache_signature = []
        for prompt_layer in prompt_cache.layers:
            if prompt_layer.keys is None or prompt_layer.values is None:
                raise ValueError("Expert prompt cache contains an uninitialized layer")
            cache_signature.append(
                (
                    tensor_signature(prompt_layer.keys),
                    tensor_signature(prompt_layer.values),
                )
            )
        return (
            tensor_signature(inputs_embeds),
            tensor_signature(position_ids),
            tensor_signature(attention_mask),
            tuple(cache_signature),
        )

    def _capture(
        self,
        inputs_embeds: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_cache: Cache,
    ) -> _GraphEntry:
        parameter = next(self._expert.parameters())
        layer_count = len(self._expert.layers)
        if len(prompt_cache.layers) != layer_count:
            raise ValueError("Expert prompt cache layer count does not match the expert")
        if any(
            tensor.device != parameter.device
            for tensor in (inputs_embeds, position_ids, attention_mask)
        ):
            raise ValueError("Diffusion expert graph inputs must share the model device")

        # These buffers are updated before every replay. Allocate ordinary tensors even
        # when the first call arrives under inference mode so later no-grad callers may
        # legally copy into them as well.
        with torch.inference_mode(False):
            prompt_keys: list[torch.Tensor] = []
            prompt_values: list[torch.Tensor] = []
            for prompt_layer in prompt_cache.layers:
                if prompt_layer.keys is None or prompt_layer.values is None:
                    raise ValueError("Expert prompt cache contains an uninitialized layer")
                if (
                    prompt_layer.keys.device != parameter.device
                    or prompt_layer.values.device != parameter.device
                ):
                    raise ValueError("Diffusion expert prompt cache must share the model device")
                prompt_keys.append(torch.zeros_like(prompt_layer.keys))
                prompt_values.append(torch.zeros_like(prompt_layer.values))

            static_inputs = torch.zeros_like(inputs_embeds)
            static_position_ids = torch.zeros_like(position_ids)
            static_attention_mask = torch.zeros_like(attention_mask)

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            static_prompt_cache = Cache(
                layers=[
                    _ReadOnlyPromptCacheLayer(keys, values)
                    for keys, values in zip(prompt_keys, prompt_values, strict=True)
                ]
            )

            def static_forward() -> Any:
                return self._original_forward(
                    inputs_embeds=static_inputs,
                    position_ids=static_position_ids,
                    past_key_values=static_prompt_cache,
                    attention_mask=static_attention_mask,
                    use_cache=True,
                    is_causal=False,
                )

            current_stream = torch.cuda.current_stream(parameter.device)
            warmup_stream = torch.cuda.Stream(device=parameter.device)
            warmup_stream.wait_stream(current_stream)
            with torch.cuda.stream(warmup_stream):
                for _ in range(3):
                    static_forward()
            current_stream.wait_stream(warmup_stream)
            torch.cuda.synchronize(parameter.device)

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, capture_error_mode="thread_local"):
                static_output = static_forward()
            torch.cuda.synchronize(parameter.device)

        logger.info(
            "Captured diffusion expert CUDA graph: layers=%d batch_size=%d "
            "sequence_length=%d prompt_length=%d graph=%d/%d",
            layer_count,
            inputs_embeds.shape[0],
            inputs_embeds.shape[1],
            prompt_cache.get_seq_length(),
            len(self._graphs) + 1,
            self._max_graphs,
        )
        return _GraphEntry(
            graph=graph,
            output=static_output,
            inputs_embeds=static_inputs,
            position_ids=static_position_ids,
            attention_mask=static_attention_mask,
            prompt_keys=prompt_keys,
            prompt_values=prompt_values,
        )


def enable_diffusion_expert_cuda_graph(
    expert: torch.nn.Module,
    *,
    max_batch_size: int,
    max_graphs: int = 4,
) -> DiffusionExpertCudaGraph:
    """Install or return the bounded exact CUDA graph runner for ``expert``."""
    existing = getattr(expert, "_diffusion_expert_cuda_graph", None)
    if existing is not None:
        return existing

    runner = DiffusionExpertCudaGraph(
        expert,
        max_batch_size=max_batch_size,
        max_graphs=max_graphs,
    )
    expert.forward = runner.forward
    expert._diffusion_expert_cuda_graph = runner
    logger.info(
        "Enabled lazy diffusion expert CUDA graphs: layers=%d max_batch_size=%d max_graphs=%d",
        len(expert.layers),
        max_batch_size,
        max_graphs,
    )
    return runner
