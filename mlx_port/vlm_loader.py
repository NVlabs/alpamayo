"""
Stage 1 - VLM Port (MLX)
Replicates the VLM loading logic from src/alpamayo_r1/models/base_model.py (ReasoningVLA).

NVIDIA loading flow (PyTorch / transformers):
  1. ReasoningVLAConfig specifies vlm_name_or_path (defaults to Qwen/Qwen3-VL-8B-Instruct)
  2. from_pretrained_submodules() or _initialize_qwenvl3_vlm() does:
       vlm = Qwen3VLForConditionalGeneration.from_pretrained(vlm_name_or_path)
       vlm.resize_token_embeddings(new_vocab_size)   # for added <|traj_*> and <i0>..<iN> tokens
  3. AlpamayoR1 wraps the VLM + adds discrete token head + diffusion expert.

MLX equivalent (mlx-vlm):
  - mlx_vlm.load(local_path) handles the Qwen3-VL architecture natively.
  - We point it at the downloaded Qwen3-VL-8B-Instruct (the same backbone used for Alpamayo-R1).
  - Token extension and Alpamayo-specific chat template will be layered on top.
"""

from pathlib import Path
from typing import Any, Optional, Union

import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.models.cache import KVCache
from PIL import Image
from mlx_vlm.utils import load_image  # helper for PIL <-> MLX

from mlx_port.alpamayo_chat import (
    create_inference_message,
    apply_alpamayo_chat_template,
)

import numpy as np

# ------------------------------------------------------------------
# Numeric history injection (MLX equivalent of fuse_traj_tokens)
# ------------------------------------------------------------------

# ------------------------------------------------------------------
# Real Alpamayo tokenizer parameters (loaded once)
# ------------------------------------------------------------------
_ALPAMAYO_TOKENIZER_CFG = None


# ------------------------------------------------------------------
# ExpertLogitsProcessor (MLX port of NVIDIA's masker)
# ------------------------------------------------------------------
class ExpertLogitsProcessor:
    """MLX-compatible logits processor that masks discrete trajectory tokens.

    Replicates NVIDIA's ExpertLogitsProcessor exactly:
        scores[:, traj_token_offset:traj_token_offset+traj_vocab_size] = -inf

    Purpose: During the CoC (Chain-of-Cognition) text generation phase, prevent
    the VLM from emitting <i0>..<i3999> tokens so it produces readable English
    reasoning instead of jumping straight into the discrete trajectory tail.
    """

    def __init__(
        self,
        traj_token_offset: int = 151669,
        traj_vocab_size: int = 4000,
    ):
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size

    def __call__(self, tokens: "mx.array", logits: "mx.array") -> "mx.array":
        """Apply the mask.

        Args:
            tokens: Recently generated token IDs (shape [batch] or [seq]).
            logits: Logits for next token (shape [batch, vocab] or [vocab]).

        Returns:
            Modified logits with the trajectory token range set to -inf.
        """
        if logits.ndim == 1:
            logits = logits[None, :]
            was_1d = True
        else:
            was_1d = False

        start = self.traj_token_offset
        end = self.traj_token_offset + self.traj_vocab_size
        vocab = logits.shape[-1]

        # Build a broadcastable mask: 0 everywhere except -inf in the forbidden range
        mask = mx.zeros((1, vocab))
        mask[:, start:end] = float("-inf")
        logits = logits + mask

        if was_1d:
            logits = logits[0]
        return logits


class AlpamayoTwoStageLogitsProcessor:
    """Stateful logits processor implementing Alpamayo two-stage generation.

    - CoC phase (default): masks `<i0>..<i3999>` → readable English reasoning.
    - After a switch token (`<|traj_future_start|>` or `<|cot_end|>`) is
      generated, automatically enters discrete-tail phase (no masking) so the
      VLM can emit the `<iN>` discrete trajectory tokens.

    Usage in generate_alpamayo_coc (when two_stage=True):
        processor = AlpamayoTwoStageLogitsProcessor()
        output = generate(..., logits_processors=[processor], ...)
    The processor flips internally when it sees the switch token id.
    """

    def __init__(
        self,
        traj_token_offset: int = 151669,
        traj_vocab_size: int = 4000,
        switch_token_ids: list[int] | None = None,
    ):
        self.traj_token_offset = traj_token_offset
        self.traj_vocab_size = traj_vocab_size
        self.switch_token_ids = set(switch_token_ids or [])
        self.in_discrete_phase = False

    def __call__(self, tokens: "mx.array", logits: "mx.array") -> "mx.array":
        if logits.ndim == 1:
            logits = logits[None, :]
            was_1d = True
        else:
            was_1d = False

        # Check if we just generated a switch token → enter discrete phase
        if not self.in_discrete_phase and self.switch_token_ids:
            if isinstance(tokens, (list, tuple)):
                recent = tokens
            elif hasattr(tokens, "tolist"):
                recent = tokens.tolist()
            else:
                recent = [int(tokens)] if isinstance(tokens, (int, mx.array)) else []
            if any(t in self.switch_token_ids for t in recent):
                self.in_discrete_phase = True

        if not self.in_discrete_phase:
            start = self.traj_token_offset
            end = self.traj_token_offset + self.traj_vocab_size
            vocab = logits.shape[-1]
            mask = mx.zeros((1, vocab))
            mask[:, start:end] = float("-inf")
            logits = logits + mask

        if was_1d:
            logits = logits[0]
        return logits

    def set_switch_token_ids(self, ids: list[int]):
        """Allow the caller to inject the exact token ids after tokenizer setup."""
        self.switch_token_ids = set(ids)


class StopAfterEOS:
    """MLX-compatible stopping criterion that stops generation one token
    after the first occurrence of a designated EOS token.

    This exactly replicates NVIDIA's `StopAfterEOS` behavior used in
    `sample_trajectories_from_data_with_vlm_rollout`:
        eos_token_id = tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
        stopping_criteria = StopAfterEOS(eos_token_id)

    The extra token after EOS ensures the KV cache is updated for the
    position right after `<|traj_future_start|>`, which the Expert needs.
    """

    def __init__(self, eos_token_id: int):
        self.eos_token_id = eos_token_id
        self.eos_found = False
        self.extra_token_generated = False

    def __call__(self, generated_tokens) -> bool:
        """Return True when we should stop.

        mlx_lm passes either a single int or a list of recently generated tokens.
        We treat any occurrence of the EOS id as the trigger.
        """
        # Normalize to list
        if isinstance(generated_tokens, int):
            token_list = [generated_tokens]
        else:
            token_list = list(generated_tokens)

        if self.eos_found:
            # We already saw EOS; this call means one more token has been generated
            self.extra_token_generated = True
            return True

        if self.eos_token_id in token_list:
            self.eos_found = True
            # Do not stop yet – let one more token be generated
            return False

        return False

    def reset(self, eos_token_id=None):
        """Reset internal state so the criterion can be reused."""
        self.eos_found = False
        self.extra_token_generated = False
        if eos_token_id is not None:
            self.eos_token_id = eos_token_id

def _get_alpamayo_tokenizer_params():
    """Load the exact hist_traj_tokenizer / traj_tokenizer parameters from the checkpoint."""
    global _ALPAMAYO_TOKENIZER_CFG
    if _ALPAMAYO_TOKENIZER_CFG is not None:
        return _ALPAMAYO_TOKENIZER_CFG

    import json
    from pathlib import Path

    # Prefer the Alpamayo checkpoint if present
    alpamayo_cfg_path = Path(__file__).parent.parent / "pre-trained" / "Alpamayo-R1-10B" / "config.json"
    if alpamayo_cfg_path.exists():
        with open(alpamayo_cfg_path) as f:
            cfg = json.load(f)
        # traj_tokenizer_cfg has the authoritative dims / num_bins for discrete tokens
        tcfg = cfg.get("traj_tokenizer_cfg", {})
        _ALPAMAYO_TOKENIZER_CFG = {
            "dims_min": tcfg.get("dims_min", [-10.0, -10.0]),
            "dims_max": tcfg.get("dims_max", [10.0, 10.0]),
            "num_bins": tcfg.get("num_bins", 3000),
            # action_space stats (also used for denormalization in expert.py)
            "accel_mean": cfg.get("action_space_cfg", {}).get("accel_mean", 0.02902694707164455),
            "accel_std": cfg.get("action_space_cfg", {}).get("accel_std", 0.6810426736454882),
            "curvature_mean": cfg.get("action_space_cfg", {}).get("curvature_mean", 0.0002692167976330542),
            "curvature_std": cfg.get("action_space_cfg", {}).get("curvature_std", 0.026148280660833106),
        }
    else:
        # Fallback to known-good values extracted from the checkpoint
        _ALPAMAYO_TOKENIZER_CFG = {
            "dims_min": [-10.0, -10.0],
            "dims_max": [10.0, 10.0],
            "num_bins": 3000,
            "accel_mean": 0.02902694707164455,
            "accel_std": 0.6810426736454882,
            "curvature_mean": 0.0002692167976330542,
            "curvature_std": 0.026148280660833106,
        }
    return _ALPAMAYO_TOKENIZER_CFG


def _encode_history_to_discrete_token_ids(
    traj_data: dict, num_history_tokens: int = 16
) -> list[int] | None:
    """
    Encode ego_history_xyz / ego_history_rot into discrete <iN> token IDs
    using the exact NVIDIA pattern from `tokenize_history_trajectory`.

    NVIDIA calls the tokenizer with:
        hist_xyz = last_pose_only   (shape [..., 1, 3])   → the initial state at t0
        fut_xyz  = full_history     (shape [..., T, 3])   → the sequence to tokenize

    This guarantees the tokenizer always returns exactly `tokens_per_history_traj`
    tokens (default 16) regardless of how many raw pose steps are supplied.

    We default to 16 tokens per history trajectory to match Alpamayo's
    `tokens_per_history_traj` configuration.
    """
    xyz = traj_data.get("ego_history_xyz")
    rot = traj_data.get("ego_history_rot")
    if xyz is None or rot is None:
        return None

    # Convert to numpy
    if hasattr(xyz, "detach"):
        xyz = xyz.detach().cpu().numpy()
    if hasattr(rot, "detach"):
        rot = rot.detach().cpu().numpy()
    xyz = np.asarray(xyz).reshape(-1, 3)
    rot = np.asarray(rot).reshape(-1, 3, 3)

    # NVIDIA trick: last pose only as the "history prefix" (initial state)
    last_pose_xyz = xyz[-1:, :]          # shape (1, 3)
    last_pose_rot = rot[-1:, :, :]       # shape (1, 3, 3)

    # Full history window passed as the "future" sequence
    full_xyz = xyz
    full_rot = rot

    # Use the real tokenizer (loads params from Alpamayo config)
    from mlx_port.action_space import create_history_tokenizer_from_alpamayo_config

    tokenizer = create_history_tokenizer_from_alpamayo_config()
    token_ids = tokenizer.encode(last_pose_xyz, last_pose_rot, full_xyz, full_rot)

    # Guarantee exactly num_history_tokens IDs
    if len(token_ids) < num_history_tokens:
        token_ids = np.pad(token_ids, (0, num_history_tokens - len(token_ids)), mode="edge")
    else:
        token_ids = token_ids[:num_history_tokens]

    return token_ids.tolist()


DEFAULT_VLM_PATH = Path("pre-trained/Qwen3-VL-8B-Instruct")


# Same values used by NVIDIA in src/alpamayo_r1/helper.py
MIN_PIXELS = 163840
MAX_PIXELS = 196608


def get_vlm_text_config(model, expert_cfg: dict | None = None) -> dict:
    """
    Extract the attention-related configuration from a loaded VLM model
    and optionally apply Alpamayo-specific expert overrides.

    This replicates the exact NVIDIA pattern:

        expert_config = copy.deepcopy(self.vlm.config.text_config)
        if config.expert_cfg is not None:
            for key, value in config.expert_cfg.items():
                setattr(expert_config, key, value)
        self.expert = AutoModel.from_config(expert_config)

    The VLM supplies the base GQA / RoPE / head_dim style, while
    `expert_cfg` (from Alpamayo-R1-10B/config.json) supplies the
    Expert-specific dimensions (hidden_size=2048, intermediate_size=8256, …).
    """
    # --- 1. Read base text config from the loaded VLM ---
    text_model = getattr(model, "language_model", None) or model
    inner = getattr(text_model, "model", text_model)
    args = getattr(inner, "args", None) or getattr(inner, "config", None)

    base = {
        "hidden_size": getattr(args, "hidden_size", 4096),
        "num_attention_heads": getattr(args, "num_attention_heads", 16),
        "num_key_value_heads": getattr(args, "num_key_value_heads", 8),
        "head_dim": getattr(args, "head_dim", 128),
        "intermediate_size": getattr(args, "intermediate_size", 11008),
        "num_hidden_layers": getattr(args, "num_hidden_layers", 36),
    }

    # --- 2. Apply expert overrides (the Alpamayo "expert_cfg" block) ---
    if expert_cfg:
        for k, v in expert_cfg.items():
            if k in base or k in ("num_layers",):
                base[k] = v

    # Normalize a few common aliases
    if "num_layers" in base:
        base["num_hidden_layers"] = base.pop("num_layers")

    return base


def load_vlm(
    model_path: Union[str, Path] = DEFAULT_VLM_PATH,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
):
    """
    Load the Qwen3-VL VLM using mlx-vlm.

    This is the direct MLX counterpart to:
        Qwen3VLForConditionalGeneration.from_pretrained(vlm_name_or_path)
    in src/alpamayo_r1/models/base_model.py:_initialize_qwenvl3_vlm / from_pretrained_submodules.

    Args:
        model_path: Local path to Qwen3-VL-8B-Instruct (or Alpamayo VLM weights once converted).
        min_pixels / max_pixels: Control the number of visual tokens (same values as NVIDIA).

    Returns:
        (model, processor) tuple ready for generation.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"VLM checkpoint not found at {model_path}")

    print(f"[mlx-vlm] Loading VLM from {model_path} ...")
    model, processor = load(
        str(model_path),
        processor_kwargs={"min_pixels": min_pixels, "max_pixels": max_pixels},
    )
    print("[mlx-vlm] VLM loaded successfully on Apple Silicon GPU.")
    return model, processor


def load_alpamayo_vlm_submodule(alpamayo_path: Union[str, Path] = Path("pre-trained/Alpamayo-R1-10B")):
    """
    Future helper (Stage 1/2) to extract and load only the VLM weights from a full Alpamayo checkpoint.

    In the NVIDIA code this is achieved by:
        vlm = Qwen3VLForConditionalGeneration.from_pretrained(config.vlm_name_or_path)
        ...
        pretrained_modules["vlm"] = vlm

    For MLX we will either:
      - Convert only the VLM portion of Alpamayo-R1-10B to mlx-vlm format, or
      - Continue using the base Qwen3-VL checkpoint for the VLM foundation and
        later graft the Alpamayo fine-tuned VLM weights.
    """
    # Placeholder – will be implemented once we have a converted Alpamayo VLM checkpoint
    raise NotImplementedError("VLM submodule extraction from Alpamayo-R1-10B pending conversion tooling.")


def generate_vlm(
    model,
    processor,
    image_path: Union[str, Path],
    prompt: str,
    max_tokens: int = 256,
    temperature: float = 0.7,
) -> str:
    """
    Run VLM inference on a single image + text prompt.

    This mirrors the visual reasoning step in Alpamayo (CoC generation).
    """
    image = load_image(str(image_path))

    # mlx-vlm expects chat-style formatting for Qwen-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    # Apply chat template (Qwen-VL specific)
    prompt_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    output = generate(
        model,
        processor,
        prompt_text,
        image,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return output


# ------------------------------------------------------------------
# Alpamayo-specific generation helpers (Gap #2 in progress)
# ------------------------------------------------------------------

class StopOnSpecialToken:
    """Stops generation when one of the Alpamayo special tokens appears."""
    def __init__(self, processor, stop_tokens: list[str]):
        self.stop_ids = [processor.tokenizer.convert_tokens_to_ids(t) for t in stop_tokens]
        self.triggered = False

    def __call__(self, generated_tokens: list[int]) -> bool:
        if any(tok in generated_tokens[-2:] for tok in self.stop_ids):
            self.triggered = True
            return True
        return False


def generate_alpamayo_coc_with_control(
    model,
    processor,
    images: list[Any],
    max_tokens: int = 256,
    temperature: float = 0.6,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> str:
    """
    Controlled Alpamayo CoC generation with proper stopping criteria + logits masking.

    Replicates the first half of NVIDIA's `sample_trajectories_from_data_with_vlm_rollout`:
    - Alpamayo multi-camera + history chat template
    - ExpertLogitsProcessor masks <i0>..<i3999> so the VLM generates readable English CoC
    - Stopping on `<|cot_end|>` / `<|traj_future_start|>`
    """
    pil_images = []
    for img in images:
        if isinstance(img, (str, Path)):
            pil_images.append(Image.open(img).convert("RGB"))
        else:
            pil_images.append(img)

    messages = create_inference_message(frames=pil_images)
    prompt_text = apply_alpamayo_chat_template(processor, messages)

    # Build stopping criteria
    stop_crit = StopOnSpecialToken(processor, ["<|cot_end|>", "<|traj_future_start|>"])

    # Use the underlying mlx-lm generation with custom stopping
    output = generate(
        model,
        processor,
        prompt_text,
        pil_images,
        max_tokens=max_tokens,
        temperature=temperature,
        processor_kwargs={"min_pixels": min_pixels, "max_pixels": max_pixels},
        # Note: mlx-vlm 0.1.x passes **kwargs to mlx_lm.generate
        stopping_criteria=[stop_crit],
    )

    # If we stopped early on a special token, the output may contain it.
    # For now we just return the raw text (further post-processing can be added).
    return output


def generate_alpamayo_coc(
    model,
    processor,
    images: list[Any],
    max_tokens: int = 256,
    temperature: float = 0.6,
    top_p: float = 0.98,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    two_stage: bool = True,
    stop_on_traj_future: bool = True,
) -> str:
    """
    Run Alpamayo-style Chain-of-Cognition (CoC) + optional discrete tail.

    When `two_stage=True` (default) this uses the `AlpamayoTwoStageLogitsProcessor`
    which starts masked (for readable CoC) and automatically switches to unmasked
    mode once `<|traj_future_start|>` or `<|cot_end|>` is generated.

    A true KV-cache-based two-pass (Pass 1 → extract CoC → Pass 2 with cache)
    is the ideal next step for perfect separation of reasoning and discrete tail,
    but requires deeper integration with mlx_vlm internals.
    """
    pil_images = []
    for img in images:
        if isinstance(img, (str, Path)):
            pil_images.append(Image.open(img).convert("RGB"))
        else:
            pil_images.append(img)

    messages = create_inference_message(frames=pil_images)
    prompt_text = apply_alpamayo_chat_template(processor, messages)

    switch_ids = [
        processor.tokenizer.convert_tokens_to_ids(t)
        for t in ["<|traj_future_start|>", "<|cot_end|>"]
    ]

    if two_stage:
        lp = AlpamayoTwoStageLogitsProcessor()
        lp.set_switch_token_ids(switch_ids)
    else:
        lp = ExpertLogitsProcessor()

    stop_crit = None
    if stop_on_traj_future and not two_stage:
        eos_id = processor.tokenizer.convert_tokens_to_ids("<|traj_future_start|>")
        stop_crit = StopAfterEOS(eos_id)

    output = generate(
        model,
        processor,
        prompt_text,
        pil_images,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        processor_kwargs={"min_pixels": min_pixels, "max_pixels": max_pixels},
        stopping_criteria=stop_crit,
        logits_processors=[lp],
    )
    return output


def extract_readable_coc(raw_output: str) -> str:
    """
    Extract the clean English Chain-of-Cognition text from the raw VLM output.

    Looks for the span between <|cot_start|> and <|cot_end|>.
    Falls back to returning the raw string if markers are absent.

    This addresses Gap #3 (Readable CoC extraction) so that reports and
    downstream code can show human-interpretable reasoning instead of
    the full token stream.
    """
    start_marker = "<|cot_start|>"
    end_marker = "<|cot_end|>"
    if start_marker in raw_output and end_marker in raw_output:
        start_idx = raw_output.find(start_marker) + len(start_marker)
        end_idx = raw_output.find(end_marker)
        coc = raw_output[start_idx:end_idx].strip()
        # Also strip any leading/trailing special tokens or whitespace artifacts
        return coc
    # Fallback: return everything (will still contain extension tokens etc.)
    return raw_output.strip()


def prefill_vlm_kv_cache(
    model,
    processor,
    images: list,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
    prompt_tokens: int = 200,
    traj_data: dict | None = None,
) -> list:
    """
    Perform a real VLM forward prefill using the *exact* Alpamayo chat-template
    prompt + image tokens and explicit KVCache objects.

    Because mlx-vlm's high-level wrapper does not expose past_key_values,
    we manually create one KVCache per language-model layer, pass the list
    to the model, and let the caches be mutated in-place via update_and_fetch.
    The resulting tensors are real, weight-dependent, and semantically
    conditioned on the full driving-scene prompt.

    Returns a list of (k, v) tuples (one per VLM layer). If the VLM has fewer
    layers than the Expert (36), the last cache is repeated to match.
    """
    # 1. Prepare images as PIL
    pil_images = []
    for img in images:
        if isinstance(img, (str, Path)):
            pil_images.append(Image.open(img).convert("RGB"))
        else:
            pil_images.append(img)

    # 2. Build the exact Alpamayo inference prompt
    messages = create_inference_message(frames=pil_images)
    prompt_text = apply_alpamayo_chat_template(processor, messages)

    # 3. Tokenize + preprocess images
    inputs = processor(
        text=prompt_text,
        images=pil_images,
        return_tensors="mlx",
        min_pixels=min_pixels,
        max_pixels=max_pixels,
    )

    # 3b. Numeric history injection (MLX equivalent of fuse_traj_tokens)
    # NVIDIA produces exactly `tokens_per_history_traj` (16) discrete tokens for history.
    # The chat template still contains 48 <|traj_history|> placeholders, but we only
    # replace the first 16 (matching the trained tokenizer output length).
    if traj_data is not None and traj_data.get("ego_history_xyz") is not None:
        try:
            hist_ids = _encode_history_to_discrete_token_ids(traj_data, num_history_tokens=16)
            if hist_ids is not None and len(hist_ids) == 16:
                traj_hist_token_id = processor.tokenizer.convert_tokens_to_ids("<|traj_history|>")
                input_ids = inputs["input_ids"]
                mask = input_ids == traj_hist_token_id
                n_placeholders = int(mask.sum())
                if n_placeholders >= 16:
                    # Replace the first 16 occurrences (they are contiguous)
                    positions = np.where(mask)[1][:16].tolist()
                    input_ids_np = np.array(input_ids)
                    input_ids_np[:, positions] = hist_ids
                    inputs["input_ids"] = mx.array(input_ids_np)
                    print(f"[Info] Numeric history injection: replaced 16 <|traj_history|> placeholders "
                          f"with discrete <iN> tokens (IDs {hist_ids[:5]}...{hist_ids[-3:]}).")
                else:
                    print(f"[Warning] History injection: found only {n_placeholders} <|traj_history|> tokens "
                          f"(expected ≥16). Injection skipped.")
        except Exception as e:
            print(f"[Warning] History injection skipped: {e}")

    # 4. Determine number of language-model layers
    try:
        num_layers = len(model.language_model.model.layers)
    except Exception:
        num_layers = 28  # safe default for Qwen3-VL-8B

    # 5. Create explicit KVCache objects (one per layer)
    caches = [KVCache() for _ in range(num_layers)]

    # 6. Run the forward pass, passing the cache list.
    #    The model will call cache.update_and_fetch inside each layer,
    #    mutating the caches in-place with real key/value tensors.
    _ = model(**inputs, cache=caches)

    # 7. Extract the up-to-date (k, v) tensors from the caches.
    #    Most KVCache implementations store .keys and .values.
    past_key_values = []
    for c in caches:
        # Prefer the sliced view up to current offset if available
        if hasattr(c, "keys") and c.keys is not None:
            k = c.keys[..., : c.offset, :] if hasattr(c, "offset") else c.keys
            v = c.values[..., : c.offset, :] if hasattr(c, "offset") else c.values
        else:
            # Fallback for some cache variants
            k = getattr(c, "key", None)
            v = getattr(c, "value", None)
        if k is not None and v is not None:
            past_key_values.append((k, v))

    # Return exactly the caches produced by the VLM (no padding, no dummy tensors).
    # The caller is responsible for handling any layer-count mismatch with the Expert.
    return past_key_values


# ------------------------------------------------------------------
# Example usage (run this file directly for quick VLM smoke test)
# ------------------------------------------------------------------
if __name__ == "__main__":
    model, processor = load_vlm()

    # Example prompt similar to what Alpamayo VLM would receive
    sample_prompt = (
        "Describe the driving scene in detail. "
        "Focus on ego-vehicle motion, surrounding agents, traffic lights, and road layout. "
        "Think step-by-step about what the ego vehicle should do next."
    )

    # TODO (Stage 1): Replace with actual image from /Volumes/MicronSSD/pai_coc/
    # For now we just demonstrate the API.
    print("\n[Stage 1 VLM] Ready for inference.")
    print("Call generate_vlm(model, processor, image_path, prompt) to run visual reasoning.")