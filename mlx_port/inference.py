"""MLX-native inference rollout for AlpamayoR1 (clean version after subclass refactor).

This module now relies on the surgical Alpamayo-specific subclasses in
`alpamayo_qwen3vl.py` (AlpamayoLanguageModel + AlpamayoModel) instead of
layering post-processing workarounds after every VLM call.

All previous rope_deltas / position_ids post-processing has been removed.
"""

from typing import Any, Dict, Tuple
import gc
import time

import mlx.core as mx
import numpy as np
from mlx_port.models.alpamayo_r1_mlx import AlpamayoR1MLX
from mlx_port.profiling import (
    is_profiling_enabled,
    StepProfiler,
    record_memory_sample,
    get_global_memory_peak,
    MemoryMonitor,
)
from mlx_lm.models.cache import KVCache
from mlx_port.models.token_utils_mlx import (
    ExpertLogitsProcessor,
    StopAfterEOS,
    replace_padding_after_eos,
    extract_text_tokens,
)
from mlx_port.models.alpamayo_qwen3vl import AlpamayoModel, AlpamayoLanguageModel


def sample_trajectories_from_data_with_vlm_rollout(
    model: AlpamayoR1MLX,
    data: Dict[str, Any],
    num_traj_samples: int = 1,
    num_traj_sets: int = 1,
    temperature: float = 1.0,
    top_p: float = 1.0,
    vlm_only: bool = False,
    return_extra: bool = False,
    **kwargs,
) -> Tuple[Any, Any, Any]:
    """Clean VLM rollout using the Alpamayo subclasses.

    The heavy post-processing that was previously required for rope_deltas
    and position_ids has been removed. The fixes now live inside
    AlpamayoLanguageModel.get_rope_index and AlpamayoModel.get_input_embeddings.
    """
    n_samples_total = num_traj_samples * num_traj_sets

    ego_history_xyz = data["ego_history_xyz"]
    ego_history_rot = data["ego_history_rot"]
    tokenized_data = data["tokenized_data"]

    input_ids = tokenized_data["input_ids"]
    if isinstance(input_ids, list):
        input_ids = mx.array(input_ids)

    image_kwargs = {}
    for k, v in tokenized_data.items():
        if k in ("pixel_values", "pixel_values_videos"):
            arr = np.asarray(v)
            if arr.ndim == 2:
                arr = arr.reshape(arr.shape[0], 2, 16, 16, 3)
                arr = np.transpose(arr, (0, 4, 1, 2, 3))
            image_kwargs[k] = mx.array(arr)
        elif k in ("image_grid_thw", "video_grid_thw"):
            image_kwargs[k] = mx.array(v, dtype=mx.int32)

    traj_data_vlm = {
        "ego_history_xyz": ego_history_xyz,
        "ego_history_rot": ego_history_rot,
    }
    input_ids = model.fuse_traj_tokens(input_ids, traj_data_vlm)

    eos_token_id = model.tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
    if eos_token_id is None:
        eos_token_id = model.tokenizer.eos_token_id

    logits_processor = ExpertLogitsProcessor(
        traj_token_offset=model.traj_token_start_idx,
        traj_vocab_size=model.traj_vocab_size,
        traj_token_ids=getattr(model, "traj_token_id_list", None),
    )

    stopping_criteria = StopAfterEOS(eos_token_id=eos_token_id)
    max_new_tokens = kwargs.get("max_generation_length") or model.tokens_per_future_traj

    vlm_profiler = StepProfiler(
        enabled=is_profiling_enabled(),
        name="VLM-Gen"
    )

    n_vlm_samples = n_samples_total

    def _run_single_vlm_generation(alpamayo_model, input_ids, image_kwargs, logits_processor,
                                    stopping_criteria, max_new_tokens, temperature, top_p):
        """Single-trajectory manual generation (now relies on fixed subclasses)."""
        generated_tokens = []
        vlm = alpamayo_model.vlm

        # Create KV cache list once before the first forward pass
        n_layers = len(vlm.language_model.model.layers)
        cache = [KVCache() for _ in range(n_layers)]

        # --- Prefill (memory peaks captured by MemoryMonitor) ---
        with MemoryMonitor(poll_interval=0.05, label="vlm_prefill"):
            outputs = vlm(
                input_ids=input_ids,
                **image_kwargs,
                cache=cache,
            )
        mx.eval(outputs.logits)
        record_memory_sample("after_vlm_prefill")

        # Decode loop – driven by StepProfiler for timing + memory delta
        decode_profiler = StepProfiler(enabled=True, name="Decode")
        for step in range(max_new_tokens):
            decode_profiler.step_start(step)

            logits = outputs.logits[:, -1, :]
            logits = logits_processor(generated_tokens, logits)

            # Compact per-step diagnostic (top-3 after processor)
            probs = mx.softmax(logits.astype(mx.float32), axis=-1)
            k = 3
            topk_vals = mx.topk(probs, k, axis=-1)
            topk_idx = mx.argsort(probs, axis=-1)[..., -k:][..., ::-1]
            topk_vals = np.asarray(topk_vals[0])
            topk_idx = np.asarray(topk_idx[0])
            decoded = []
            for tid in topk_idx:
                try:
                    tok = model.tokenizer.decode([int(tid)])
                except Exception:
                    tok = f"<{int(tid)}>"
                decoded.append(tok)
            print(f"[STEP {step+1}] top-3: " + ", ".join(f"{d}({v:.3f})" for d, v in zip(decoded, topk_vals)))

            if temperature != 1.0:
                logits = logits / temperature
            next_token = mx.random.categorical(logits)

            if step == 0:
                mx.eval(next_token)
                record_memory_sample(f"after_first_decode_eval")

            generated_tokens.append(int(next_token.item()))

            if stopping_criteria(mx.array([generated_tokens])):
                decode_profiler.step_end()
                break

            outputs = vlm(
                input_ids=next_token[None, :],
                cache=cache,
            )

            mx.eval(outputs.logits)
            decode_profiler.step_end()

        decode_profiler.summary()

        generated = mx.array([generated_tokens])
        record_memory_sample("after_vlm_generation_complete")
        return generated, cache

    if n_vlm_samples <= 1:
        generated, cache = _run_single_vlm_generation(
            model, input_ids, image_kwargs, logits_processor,
            stopping_criteria, max_new_tokens, temperature, top_p
        )
        rope_deltas = 0
    else:
        seq_list = []
        cache_list = []
        for _ in range(n_vlm_samples):
            gen, c = _run_single_vlm_generation(
                model, input_ids, image_kwargs, logits_processor,
                stopping_criteria, max_new_tokens, temperature, top_p
            )
            seq_list.append(gen)
            cache_list.append(c)
        generated = mx.concatenate(seq_list, axis=0)
        cache = cache_list[-1]
        rope_deltas = 0

    class VLMOutputs:
        def __init__(self, sequences, cache, rope_deltas):
            self.sequences = sequences
            self.cache = cache
            self.rope_deltas = rope_deltas

    vlm_outputs = VLMOutputs(sequences=generated, cache=cache, rope_deltas=rope_deltas)

    generated = replace_padding_after_eos(
        generated, eos_token_id=eos_token_id, pad_token_id=model.tokenizer.pad_token_id
    )

    if vlm_only:
        extra = extract_text_tokens(model.tokenizer, vlm_outputs.sequences)
        return None, None, extra

    if kwargs.get("return_extra", False):
        extra = extract_text_tokens(model.tokenizer, vlm_outputs.sequences)
        return None, None, extra

    return None, None, None


# ------------------------------------------------------------------
# Convenience wrapper (kept for backward compatibility with tests)
# ------------------------------------------------------------------

def run_vlm_generation(model, input_ids, image_kwargs, **gen_kwargs):
    """Thin wrapper around the clean rollout for unit tests."""
    return sample_trajectories_from_data_with_vlm_rollout(
        model,
        {"tokenized_data": {"input_ids": input_ids, **image_kwargs}},
        vlm_only=True,
        return_extra=True,
        **gen_kwargs,
    )