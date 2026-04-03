# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""Configuration classes for Wrapped Reasoning VLA models for RL training."""

from typing import Any
from alpamayo_r1.models.base_model import ReasoningVLAConfig


class RLWrapperReasoningVLAConfig(ReasoningVLAConfig):
    """Configuration for the RLWrapperReasoningVLA model."""

    # NOTE:
    # `transformers.PretrainedConfig` may call `self.__class__()` internally when serializing
    # configs with `use_diff=True` (e.g. `save_pretrained()`, `to_json_string()`), to obtain
    # the class "default" config for diffing.
    #
    # Our `__init__` triggers `_initialize_vlm_config()` which can be heavy and may require
    # accessing HF configs/processors (sometimes network / cached artifacts). Setting this
    # flag tells transformers to *not* instantiate a fresh default config during serialization,
    # avoiding unintended side-effects and failures when saving checkpoints/logging.
    has_no_defaults_at_init = True

    def __init__(
        self,
        hist_traj_embed_cfg: dict[str, Any] | None = None,
        include_camera_ids: bool = False,
        include_frame_nums: bool = False,
        padding_side: str = "left",
        loss_weights: dict[str, float] = {"future_traj": 1.0, "others": 1.0},
        **kwargs: Any,
    ) -> None:
        """Initialize the model configuration.

        Args:
            hist_traj_embed_cfg: Configuration for the history trajectory embedding, if using a
                separate embedding method.
            include_camera_ids: Whether to include camera IDs as text before images.
            include_frame_nums: Whether to include frame numbers as text before images.
            padding_side: Padding side for tokenization ("left" or "right").
            loss_weights: Per-component loss weights with keys "future_traj" and "others".
            **kwargs: Passed through to ``ReasoningVLAConfig`` (e.g. ``traj_vocab_size``,
                ``tokens_per_history_traj``, ``model_dtype``, etc.).
        """
        super().__init__(**kwargs)

        # Core configuration
        self.hist_traj_embed_cfg = hist_traj_embed_cfg
        self.include_camera_ids = include_camera_ids
        self.include_frame_nums = include_frame_nums
        self.loss_weights = loss_weights
        self.padding_side = padding_side
        # Initialize VLM-specific configurations
        self._initialize_vlm_config()

    def _patch_llm_config_vocab(self, cfg: Any) -> Any:
        """Ensure the LLM config's vocab_size matches the expanded alpamayo vocab.

        The upstream VLM config reports its own text_config.vocab_size, which is
        smaller than the alpamayo model's expanded vocab (due to added trajectory
        and special tokens).  Downstream consumers -- vLLM TP sharding, Cosmos-RL
        weight mapper, rollout -- all read vocab_size from this config and must
        see the expanded value to avoid shape mismatches.
        """
        vocab = getattr(self, "vocab_size", None)
        if vocab is not None:
            if getattr(cfg, "text_config", None) is not None:
                cfg.text_config.vocab_size = vocab
            cfg.vocab_size = vocab
        return cfg

    def get_llm_config(self) -> Any:
        """Return the underlying language model config for the VLM.

        This mirrors composite configs used in Cosmos that provide a get_llm_config()
        helper for downstream components (e.g., vLLM overrides) that need LLM fields
        like num_attention_heads, num_key_value_heads, hidden_size, etc.
        """
        # Cache to avoid repeated HF config loads (vLLM may call this multiple
        # times during initialization).
        cached: Any = getattr(self, "_cached_llm_config", None)
        if cached is not None:
            # NOTE: Some checkpoints accidentally serialize `_cached_llm_config`
            # into `config.json` as a plain dict. Downstream components (Cosmos
            # weight mapper, vLLM overrides) expect a `PretrainedConfig` object
            # with attributes like `text_config.num_key_value_heads`.
            # If we see a dict here, reconstruct a proper config object and
            # re-cache it.
            if isinstance(cached, dict):
                from transformers import AutoConfig

                model_type = cached.get("model_type", None)
                rebuilt = None
                # Best effort: rebuild from the dict itself.
                if model_type is not None:
                    try:
                        rebuilt = AutoConfig.for_model(model_type, **cached)
                    except Exception:
                        rebuilt = None
                # Fallback: reload from the backing VLM path.
                if rebuilt is None:
                    rebuilt = AutoConfig.from_pretrained(
                        self.vlm_name_or_path,
                        trust_remote_code=True,
                    )
                setattr(self, "_cached_llm_config", rebuilt)
                return self._patch_llm_config_vocab(rebuilt)
            return self._patch_llm_config_vocab(cached)

        from transformers import AutoConfig

        base_cfg = AutoConfig.from_pretrained(
            self.vlm_name_or_path,
            trust_remote_code=True,
        )
        setattr(self, "_cached_llm_config", base_cfg)
        return self._patch_llm_config_vocab(base_cfg)

    def to_dict(self) -> dict[str, Any]:
        """Serialize config without transient caches.

        `_cached_llm_config` can be very large and, if saved, will be loaded back
        as a plain dict (breaking attribute-based consumers). Exclude it from
        serialization to keep checkpoints stable and minimal.
        """
        d: dict[str, Any] = super().to_dict()
        d.pop("_cached_llm_config", None)
        return d

    def get_text_config(self) -> Any:
        """Return the underlying *text* config.

        vLLM uses `config.get_text_config()` to populate `hf_text_config` and
        derive critical fields like `hidden_size`. For our composite config, the
        text config lives under the backing VLM config.
        """
        llm_cfg = self.get_llm_config()
        # Many multimodal configs (e.g., Qwen3-VL) expose a `get_text_config()`
        # that returns the nested text config (with hidden_size, heads, ...).
        if hasattr(llm_cfg, "get_text_config"):
            return llm_cfg.get_text_config()
        return llm_cfg
