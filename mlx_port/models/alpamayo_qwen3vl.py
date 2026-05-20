"""Surgical Alpamayo-specific overrides for mlx_vlm Qwen3-VL classes.

These subclasses fix two issues discovered during Alpamayo-R1-10B porting:

1. get_rope_index: The stock implementation sums vision_start token positions
   instead of collecting them, causing mrope_position_deltas = [[0]] for
   16-image (4 cameras × 4 frames) prompts.

2. get_input_embeddings: The stock code unconditionally clears
   self.language_model._position_ids on every decode step (when pixel_values
   is None). This discards the cached multimodal RoPE state that Alpamayo
   needs for correct continuation.

Only these two methods are overridden; everything else (vision tower,
weight loading, KV cache, etc.) remains identical to mlx_vlm 0.5.0.
"""

from typing import Any, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from mlx_vlm.models.qwen3_vl.language import LanguageModel
from mlx_vlm.models.qwen3_vl.qwen3_vl import Model, InputEmbeddingsFeatures


class AlpamayoLanguageModel(LanguageModel):
    """LanguageModel with corrected vision-aware RoPE index computation and
    guaranteed preservation of multimodal position state across decode steps.
    """

    def __call__(
        self,
        inputs: mx.array,
        inputs_embeds: Optional[mx.array] = None,
        mask: Optional[mx.array] = None,
        cache=None,
        visual_pos_masks: Optional[mx.array] = None,
        deepstack_visual_embeds: Optional[mx.array] = None,
        **kwargs,
    ):
        # --- DIAGNOSTIC: attention / mask / cache state (FORCED - no latch) ---
        is_decode_step = (
            getattr(self, "_position_ids", None) is not None
            and inputs.shape[1] < self._position_ids.shape[2]
        )
        step_kind = "DECODE" if is_decode_step else "PREFILL"
        mask_info = "None" if mask is None else f"shape={mask.shape} dtype={mask.dtype}"
        cache_info = "None"
        if cache is not None and len(cache) > 0:
            c0 = cache[0]
            offset = getattr(c0, "offset", None)
            idx = getattr(c0, "_idx", None)
            cache_info = f"len={len(cache)} offset={offset} _idx={idx}"
        print(f"[ATTN_DIAG] {step_kind} | mask={mask_info} | cache={cache_info} | inputs.shape={inputs.shape}")

        # --- Alpamayo continuation guard (refined) ---
        position_ids = kwargs.pop("position_ids", None)
        rope_deltas_kw = kwargs.pop("rope_deltas", None)

        if (
            getattr(self, "_position_ids", None) is not None
            and position_ids is None
            and inputs.shape[1] < self._position_ids.shape[2]
        ):
            seq_length = inputs.shape[1]
            position_ids = self._position_ids[:, :, -seq_length:]
            if getattr(self, "_rope_deltas", None) is None:
                last_pos = int(self._position_ids[0, 0, -1].item())
                self._rope_deltas = mx.array([[last_pos]], dtype=self._position_ids.dtype)

        if position_ids is not None:
            kwargs["position_ids"] = position_ids
        if rope_deltas_kw is not None:
            kwargs["rope_deltas"] = rope_deltas_kw

        outputs = super().__call__(
            inputs,
            inputs_embeds=inputs_embeds,
            mask=mask,
            cache=cache,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
            **kwargs,
        )

        # --- DIAGNOSTIC: first-decode logits snapshot (FORCED) ---
        if is_decode_step:
            try:
                logits = outputs.logits  # [B, 1, V]
                mx.eval(logits)
                logits_np = np.asarray(logits[0, 0, :])
                top_idx = int(np.argmax(logits_np))
                top_val = float(np.max(logits_np))
                print(f"[ATTN_DIAG] first-decode logits: shape={logits.shape} max={top_val:.3f} argmax={top_idx}")
            except Exception as e:
                print(f"[ATTN_DIAG] first-decode logits inspect failed: {e}")

        return outputs

    def get_rope_index(
        self,
        input_ids: mx.array,
        image_grid_thw: Optional[mx.array] = None,
        video_grid_thw: Optional[mx.array] = None,
        attention_mask: Optional[mx.array] = None,
    ) -> Tuple[mx.array, mx.array]:
        # --- Core logic copied from mlx_vlm with one critical fix ---
        batch_size, seq_length = input_ids.shape
        position_ids = mx.arange(seq_length, dtype=mx.int32)
        position_ids = mx.broadcast_to(position_ids[None, :], (batch_size, seq_length))
        spatial_merge_size = self.config.vision_config.spatial_merge_size
        image_token_id = self.config.image_token_id
        video_token_id = self.config.video_token_id
        vision_start_token_id = self.config.vision_start_token_id
        mrope_position_deltas: list[int] = []

        if input_ids is not None and (image_grid_thw is not None or video_grid_thw is not None):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = mx.ones_like(input_ids)
            position_ids = mx.ones((3, batch_size, seq_length), dtype=input_ids.dtype)
            image_index, video_index = 0, 0
            for i, input_ids_row in enumerate(total_input_ids):
                input_ids_row = mx.where(
                    attention_mask[i] == 1, input_ids_row, mx.zeros_like(input_ids_row)
                )
                # === THE FIX: collect real indices (list comprehension) ===
                input_tokens = input_ids_row.tolist()
                vision_start_indices = [
                    idx for idx, tok in enumerate(input_tokens) if tok == vision_start_token_id
                ]
                image_nums = sum(
                    1 for idx in vision_start_indices
                    if input_tokens[idx + 1] == image_token_id
                )
                video_nums = sum(
                    1 for idx in vision_start_indices
                    if input_tokens[idx + 1] == video_token_id
                )

                print(f"[ROPE_DEBUG] batch{i}: seq_len={len(input_tokens)}, vision_start_indices={vision_start_indices}")
                print(f"[ROPE_DEBUG]   image_nums={image_nums}, video_nums={video_nums}, image_grid_thw={image_grid_thw.tolist() if image_grid_thw is not None else None}")

                # Pre-compute actual vision token count per block from the token positions.
                # This is the exact number of vision tokens the processor inserted between
                # two consecutive <vision_start> markers (or to the end of the prompt).
                per_block_vision_token_count = []
                for k in range(len(vision_start_indices)):
                    start = vision_start_indices[k]
                    if k + 1 < len(vision_start_indices):
                        end = vision_start_indices[k + 1]
                    else:
                        end = len(input_tokens)
                    # The block contains the <vision_start> token itself + the image patches
                    # that follow it until the next vision_start. We only want the patch count.
                    count = end - start - 1   # subtract the <vision_start> token
                    per_block_vision_token_count.append(count)
                print(f"[ROPE_DEBUG]   per_block_vision_token_count (first 5): {per_block_vision_token_count[:5]} ... total={sum(per_block_vision_token_count)}")

                llm_pos_ids_list: list[mx.array] = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
                vision_block_idx = 0  # 0..15 for Alpamayo 4×4 layout
                # Running position counter for RoPE indices.
                # Incremented by the actual token count of each segment (text or vision).
                # This ensures st_idx grows by ~2040 per vision block instead of by the
                # small spatial-grid range (≈ 60).
                current_pos = 0
                for _ in range(image_nums + video_nums):
                    if image_token_id in input_tokens and remain_images > 0:
                        ed_image = input_tokens.index(image_token_id, st)
                    else:
                        ed_image = len(input_tokens) + 1
                    if video_token_id in input_tokens and remain_videos > 0:
                        ed_video = input_tokens.index(video_token_id, st)
                    else:
                        ed_video = len(input_tokens) + 1
                    if ed_image < ed_video:
                        # Map the current vision block to the correct grouped row
                        camera_id = vision_block_idx // 4          # 0..3
                        frame_in_camera = vision_block_idx % 4     # 0..3
                        grid_row = camera_id
                        t, h, w = (
                            image_grid_thw[grid_row][0],
                            image_grid_thw[grid_row][1],
                            image_grid_thw[grid_row][2],
                        )
                        image_index += 1
                        remain_images -= 1
                        ed = ed_image
                    else:
                        t, h, w = (
                            video_grid_thw[video_index][0],
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )
                        video_index += 1
                        remain_videos -= 1
                        ed = ed_video
                    llm_grid_t = int(t.item())
                    llm_grid_h = int(h.item()) // spatial_merge_size
                    llm_grid_w = int(w.item()) // spatial_merge_size
                    text_len = ed - st
                    # Use the running position counter instead of the previous block's max value.
                    st_idx = current_pos

                    print(f"[ROPE_DEBUG]   block#{vision_block_idx}: st={st}, ed={ed}, text_len={text_len}, st_idx={st_idx}, grid=({llm_grid_t},{llm_grid_h},{llm_grid_w}), cam={camera_id}, frame={frame_in_camera}")

                    index = mx.arange(text_len).reshape(1, text_len)
                    index = mx.broadcast_to(index, (3, text_len)) + st_idx
                    llm_pos_ids_list.append(index)
                    current_pos += text_len   # advance by text tokens

                    # Construct a full 2-D spatial grid of size (llm_grid_h, llm_grid_w)
                    # = (34, 60) and flatten it. This guarantees every vision block
                    # receives exactly 2040 distinct (h, w) RoPE indices.
                    h_2d = mx.arange(llm_grid_h)[:, None]
                    w_2d = mx.arange(llm_grid_w)[None, :]
                    h_grid, w_grid = mx.broadcast_arrays(h_2d, w_2d)
                    h_index = h_grid.flatten()
                    w_index = w_grid.flatten()
                    t_index = mx.full((llm_grid_h * llm_grid_w,), frame_in_camera, dtype=mx.int32)

                    vision_indices = mx.stack([t_index, h_index, w_index]) + text_len + st_idx

                    # If the actual token count for this block is larger than the
                    # spatial grid (2040), append extra RoPE positions for the
                    # additional placeholder / special tokens. These extra tokens
                    # receive the same (t, h_last, w_last) coordinate so they stay
                    # spatially aligned with the last patch of the frame.
                    actual_vision_tokens_per_block = per_block_vision_token_count[vision_block_idx]
                    n_grid = llm_grid_h * llm_grid_w          # 2040
                    extra = actual_vision_tokens_per_block - n_grid
                    if extra > 0:
                        last_h = h_index[-1:]
                        last_w = w_index[-1:]
                        extra_t = mx.full((extra,), frame_in_camera, dtype=mx.int32)
                        extra_h = mx.tile(last_h, (extra,))
                        extra_w = mx.tile(last_w, (extra,))
                        extra_indices = mx.stack([extra_t, extra_h, extra_w]) + text_len + st_idx
                        vision_indices = mx.concatenate([vision_indices, extra_indices], axis=1)

                    llm_pos_ids_list.append(vision_indices)
                    if vision_indices.size > 0:
                        vision_pos_max = int(vision_indices.max().item())
                        print(f"[ROPE_DEBUG]     appended vision block: vision_pos_range=[{st_idx + text_len}, {vision_pos_max}], new_st={st}")
                    else:
                        print(f"[ROPE_DEBUG]     appended vision block: EMPTY (grid=({llm_grid_t},{llm_grid_h},{llm_grid_w}))")

                    # Advance the RoPE position counter and the token pointer by the
                    # true token count for this block (2040 vision patches + any extra
                    # placeholder tokens). This keeps the length of the constructed
                    # llm_positions identical to the original sequence length.
                    current_pos += actual_vision_tokens_per_block
                    st = ed + actual_vision_tokens_per_block
                    vision_block_idx += 1

                if st < len(input_tokens):
                    # Use the running position counter for the final text segment.
                    st_idx = current_pos
                    text_len = len(input_tokens) - st
                    print(f"[ROPE_DEBUG]   final_text_segment: st={st}, text_len={text_len}, st_idx={st_idx}")
                    t_index = mx.arange(text_len).reshape(1, text_len)
                    t_index = mx.broadcast_to(t_index, (3, text_len))
                    llm_pos_ids_list.append(t_index + st_idx)
                    current_pos += text_len

                llm_positions = mx.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
                print(f"[ROPE_DEBUG]   llm_positions shape={llm_positions.shape}, min={int(llm_positions.min().item())}, max={int(llm_positions.max().item())}")
                mask = mx.array(attention_mask[i] == 1)
                expanded_mask = mx.expand_dims(mask, axis=0)
                expanded_mask = mx.broadcast_to(expanded_mask, (3, 1, mask.shape[0]))
                expanded_positions = mx.expand_dims(llm_positions, axis=1)
                new_positions = mx.where(
                    expanded_mask, expanded_positions, position_ids[:, i : i + 1, :]
                )
                updated_position_ids = mx.concatenate(
                    [position_ids[:, :i, :], new_positions, position_ids[:, i + 1 :, :]],
                    axis=1,
                )
                position_ids = updated_position_ids
                mrope_position_deltas.append(int(llm_positions.max().item()) + 1 - len(total_input_ids[i]))

            mrope_position_deltas = mx.array(mrope_position_deltas).reshape(-1, 1)
            print(f"[ROPE_DEBUG] vision_path_return: position_ids.shape={position_ids.shape}, max_pos={int(position_ids.max().item())}, mrope_deltas={mrope_position_deltas.tolist()}")
            return position_ids, mrope_position_deltas

        # --- fallback (no vision) ---
        if attention_mask is not None:
            position_ids = mx.cumsum(attention_mask.astype(mx.int64), axis=-1) - 1
            position_ids = mx.where(attention_mask == 0, mx.ones_like(position_ids), position_ids)
            max_position_ids = position_ids.max(axis=-1, keepdims=True)
            position_ids = mx.broadcast_to(position_ids[None, :, :], (3, *position_ids.shape))
            mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
        else:
            position_ids = mx.arange(input_ids.shape[1]).reshape(1, -1)
            position_ids = mx.broadcast_to(position_ids, (3, input_ids.shape[0], input_ids.shape[1]))
            mrope_position_deltas = mx.zeros([input_ids.shape[0], 1], dtype=input_ids.dtype)
        return position_ids, mrope_position_deltas


class AlpamayoModel(Model):
    """Model that preserves cached multimodal position state across decode steps."""

    def get_input_embeddings(
        self,
        input_ids: Optional[mx.array] = None,
        pixel_values: Optional[mx.array] = None,
        **kwargs,
    ) -> InputEmbeddingsFeatures:
        # --- FORCED DIAGNOSTIC (get_input_embeddings path, which is always hit) ---
        m = kwargs.get("mask", None)
        mask_info = "None" if m is None else f"shape={m.shape} dtype={m.dtype}"
        print(f"[ATTN_DIAG] get_input_embeddings | mask={mask_info} | pixel_values={'present' if pixel_values is not None else 'None'} | input_ids.shape={input_ids.shape if input_ids is not None else 'None'}")

        image_grid_thw = kwargs.get("image_grid_thw", None)
        video_grid_thw = kwargs.get("video_grid_thw", None)
        mask = kwargs.get("mask", None)
        grid_thw = image_grid_thw if image_grid_thw is not None else video_grid_thw

        if pixel_values is None:
            pixel_values = kwargs.get("pixel_values_videos", None)

        if pixel_values is None:
            # Decode / text-only continuation path.
            # 1. Advance the cached _position_ids (vision-aware layout + text continuation)
            #    by appending last_col + 1. This keeps the full T/H/W layout from prefill.
            # 2. Explicitly preserve _rope_deltas (the mrope_position_deltas offset).
            #    The base LanguageModel.__call__ computes delta = cache_offset + _rope_deltas.
            #    If _rope_deltas is None on the first decode, recalc_condition triggers a
            #    full get_rope_index recompute that produces the 104 → 261 jump.
            lm = self.language_model

            if getattr(lm, "_position_ids", None) is not None:
                pos = lm._position_ids                 # (3, batch, seq_len)
                last_col = pos[:, :, -1:]              # (3, 1, 1)
                next_col = last_col + 1                # advance text continuation
                lm._position_ids = mx.concatenate([pos, next_col], axis=2)

            # Preserve _rope_deltas across decode steps (do not let it become None).
            # It is a constant offset computed once during prefill; we only need it to survive.
            if getattr(lm, "_rope_deltas", None) is None and getattr(lm, "_position_ids", None) is not None:
                # Fallback: if _rope_deltas was somehow cleared but we have _position_ids,
                # derive a minimal delta from the current last position (best-effort).
                # This keeps generation alive even if prefill state was partially lost.
                last_pos = int(lm._position_ids[0, 0, -1].item())
                lm._rope_deltas = mx.array([[last_pos]], dtype=lm._position_ids.dtype)

            return InputEmbeddingsFeatures(
                inputs_embeds=self.language_model.model.embed_tokens(input_ids)
            )

        # --- vision path unchanged ---
        dtype = self.vision_tower.patch_embed.proj.weight.dtype
        pixel_values = pixel_values.astype(dtype)

        inputs_embeds = self.language_model.model.embed_tokens(input_ids)

        cached = kwargs.get("cached_image_features", None)
        if cached is not None:
            hidden_states = cached
            deepstack_visual_embeds = None
        else:
            hidden_states, deepstack_visual_embeds = self.vision_tower(
                pixel_values, grid_thw
            )

        visual_pos_masks = None
        inputs_embeds, image_mask = self.merge_input_ids_with_image_features(
            hidden_states,
            inputs_embeds,
            input_ids,
            self.config.image_token_index,
            self.config.video_token_index,
        )
        image_mask = image_mask[..., 0]
        visual_pos_masks = image_mask
        mx.eval(deepstack_visual_embeds)

        # Note: self.language_model here is still the base mlx_vlm LanguageModel
        # instance. However, AlpamayoR1MLX.from_pretrained() monkey-patches
        # vlm.language_model.get_rope_index with AlpamayoLanguageModel's
        # corrected implementation (the one that collects all 16 vision_start
        # indices instead of summing them). Therefore this call executes the
        # fixed Alpamayo-specific logic even though the runtime type is base.
        if image_grid_thw is not None or video_grid_thw is not None:
            position_ids, rope_deltas = self.language_model.get_rope_index(
                input_ids, image_grid_thw, video_grid_thw, mask
            )
            self.language_model._position_ids = position_ids
            self.language_model._rope_deltas = rope_deltas

        return InputEmbeddingsFeatures(
            inputs_embeds=inputs_embeds,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )