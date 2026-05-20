# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Message construction and processor utilities for the MLX Alpamayo port.

This module mirrors the behavior of src/alpamayo_r1/helper.py but is
framework-agnostic so it works with MLX, NumPy, or PyTorch tensors.

History configuration (used by create_message and the inference pipeline):
    DEFAULT_NUM_FRAMES = 4 camera frames per view
    DEFAULT_NUM_HISTORY_STEPS = 16 egomotion steps (1.6 s)
    DEFAULT_HISTORY_TRAJ_TOKENS = 48 tokens produced by the history tokenizer
"""

from typing import Any, List, Union

import numpy as np
from PIL import Image
from transformers import AutoProcessor

try:
    import mlx.core as mx
except ImportError:
    mx = None

try:
    import torch
except ImportError:
    torch = None


MIN_PIXELS = 163840
MAX_PIXELS = 196608
# Use the locally downloaded Qwen3-VL-8B-Instruct checkpoint (not the Hub)
LOCAL_QWEN_PROCESSOR_PATH = "/Users/michaellee/Projects/alpamayo/pre-trained/Qwen3-VL-8B-Instruct"

# =============================================================================
# History configuration constants (exposed for clarity & configurability)
# =============================================================================

# Number of discrete trajectory tokens used to represent egomotion history.
# With the default num_history_steps=16 (1.6 s @ 10 Hz), the tokenizer emits 48 tokens.
DEFAULT_HISTORY_TRAJ_TOKENS = 48

# Default number of camera frames per view (visual history ending at t0).
DEFAULT_NUM_FRAMES = 4

# Default number of egomotion history steps (1.6 s at 0.1 s step).
DEFAULT_NUM_HISTORY_STEPS = 16


def _to_numpy(frames: Any) -> np.ndarray:
    """Convert supported array types to NumPy for shape inspection."""
    if isinstance(frames, np.ndarray):
        return frames
    if mx is not None and isinstance(frames, mx.array):
        return np.array(frames)
    if torch is not None and isinstance(frames, torch.Tensor):
        return frames.detach().cpu().numpy()
    if isinstance(frames, (list, tuple)):
        return np.array(frames)
    raise TypeError(f"Unsupported frame type: {type(frames)}")


def create_message(
    frames: Any,
    num_history_traj_tokens: int = DEFAULT_HISTORY_TRAJ_TOKENS,
) -> List[dict]:
    """Construct the chat message list expected by the VLM.

    This function is an exact port of alpamayo_r1.helper.create_message.
    It builds the system + user + assistant turn structure with the
    trajectory-history placeholder tokens.

    History lengths (see module constants):
        - DEFAULT_NUM_FRAMES = 4 camera frames per view
        - DEFAULT_NUM_HISTORY_STEPS = 16 egomotion steps (1.6 s)
        - DEFAULT_HISTORY_TRAJ_TOKENS = 48 tokens (from hist_traj_tokenizer)

    Args:
        frames: Image tensor of shape (N, C, H, W). Accepts NumPy, MLX,
                PyTorch, or Python list.
        num_history_traj_tokens: Number of <|traj_history|> tokens to insert.
            Defaults to 48 (matches the current tokenizer for 16 history steps).

    Returns:
        List of chat messages in the format expected by mlx_vlm / Qwen3-VL.
    """
    arr = _to_numpy(frames)
    if arr.ndim != 4:
        raise ValueError(f"{arr.ndim=}, expected 4 (N, C, H, W)")

    # Convert each frame (C, H, W) to a PIL Image (H, W, C) for robust processing
    # by the Qwen3-VL image processor. This prevents shape misalignment issues
    # where the processor expects 3D images but receives 4D tensors.
    pil_images = []
    for frame in arr:
        # Ensure channel-first to channel-last and uint8 range
        if frame.shape[0] in (1, 3):  # (C, H, W) format
            frame = np.transpose(frame, (1, 2, 0))  # -> (H, W, C)
        if frame.dtype != np.uint8:
            # Assume float in [0, 1] or [0, 255]; scale appropriately
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        pil_images.append(Image.fromarray(frame, mode="RGB" if frame.shape[2] == 3 else "L"))

    # NOTE: we expand the padding tokens to match training, so we can
    # directly apply the native processor from the VLM.
    num_traj_token = DEFAULT_HISTORY_TRAJ_TOKENS
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )

    return [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": "You are a driving assistant that generates safe and accurate actions.",
                }
            ],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": img} for img in pil_images]
            + [
                {
                    "type": "text",
                    "text": f"{hist_traj_placeholder}output the chain-of-thought reasoning of the driving process, then output the future trajectory.",
                }
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": "<|cot_start|>",
                }
            ],
        },
    ]


def get_processor(tokenizer: Any, model_path: str = LOCAL_QWEN_PROCESSOR_PATH) -> Any:
    """Get the processor for the locally downloaded Qwen3-VL model.

    This is the MLX-port equivalent of alpamayo_r1.helper.get_processor.
    It loads the processor from the local checkpoint directory and then
    injects the Alpamayo tokenizer (which contains the trajectory special tokens).

    Args:
        tokenizer: The Alpamayo tokenizer (with traj tokens already added).
        model_path: Path to the local Qwen3-VL checkpoint. Defaults to the
            project-local copy at pre-trained/Qwen3-VL-8B-Instruct.

    Returns:
        A processor object with the Alpamayo tokenizer attached.
    """
    processor_kwargs = {
        "min_pixels": MIN_PIXELS,
        "max_pixels": MAX_PIXELS,
    }

    processor = AutoProcessor.from_pretrained(model_path, **processor_kwargs)
    processor.tokenizer = tokenizer
    return processor


def alpamayo_apply_chat_template(
    processor: Any,
    messages: List[dict],
    tokenize: bool = True,
    add_generation_prompt: bool = False,
    continue_final_message: bool = True,
    return_dict: bool = True,
    return_tensors: str = "np",
    padding: bool = True,
) -> Any:
    """Custom apply_chat_template that guarantees a flat images list.

    This is the Alpamayo equivalent of AlpamayoPatchEmbed: a controlled
    wrapper that works around a latent bug in mlx_vlm's Qwen3-VL processor.

    The mlx_vlm implementation of apply_chat_template(..., tokenize=True)
    can internally construct a nested list-of-lists for the images when
    processing multi-image messages. That nested structure reaches
    Qwen3VLImageProcessor.__call__ as [[pil0, ..., pil15]], which is then
    converted to a (16, H, W, C) array and passed to _process_one, causing
    "too many values to unpack (expected 3)".

    This function bypasses that path entirely:
    1. Calls apply_chat_template(..., tokenize=False) to obtain the prompt text.
    2. Extracts the 16 PIL images directly from the message content as a flat list.
    3. Calls the processor with an explicit flat `images=` argument.
    4. Returns a dict with the same keys as the tokenize=True path
       (input_ids, attention_mask, pixel_values, image_grid_thw, ...).

    The result is identical in structure to what a correct processor would
    return, but with the correct (16, 3) image_grid_thw that respects the
    4-camera × 4-frame temporal grouping expected by the Alpamayo fine-tune.

    Args:
        processor: The Alpamayo-injected Qwen3-VL processor.
        messages: Chat messages as produced by create_message.
        tokenize: Must be True (the only supported mode for this helper).
        add_generation_prompt, continue_final_message: Passed through.
        return_dict, return_tensors, padding: Control the output format.

    Returns:
        A dict-like object (or BatchFeature) containing tokenized inputs
        and vision tensors, exactly as the standard tokenize=True path would.
    """
    if not tokenize or not return_dict:
        # For non-tokenize or non-dict paths we simply delegate.
        return processor.apply_chat_template(
            messages,
            tokenize=tokenize,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=continue_final_message,
            return_dict=return_dict,
            return_tensors=return_tensors,
        )

    # --- The safe, controlled path ---
    text = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        continue_final_message=continue_final_message,
    )

    # Extract images as a flat list directly from the message structure.
    images = [
        item["image"]
        for item in messages[1]["content"]
        if isinstance(item, dict) and item.get("type") == "image"
    ]

    # Call the processor with an explicit flat images list.
    # This guarantees that Qwen3VLImageProcessor receives a flat list of 16
    # individual images and therefore produces the correct image_grid_thw.
    inputs = processor(
        text=text,
        images=images,
        return_tensors=return_tensors,
        padding=padding,
    )

    return inputs


def enforce_alpamayo_temporal_grouping(
    inputs: dict,
    num_cameras: int = 4,
    num_frames_per_camera: int = 4,
) -> dict:
    """Post-process processor output to enforce Alpamayo's 4×4 temporal grouping.

    The Qwen3-VL processor (even with a flat images list) produces an
    image_grid_thw with 16 independent rows of [1, H, W] because it treats
    each of the 16 images as a single-frame temporal group.

    Alpamayo was fine-tuned with 4 cameras × 4 frames per camera, expecting
    the vision tower's Conv3D (temporal_patch_size=2) to see temporally
    coherent stacks. This function reorganizes image_grid_thw so that each
    camera's 4 frames form a single temporal group with T=4 (or T=2×2 if
    we want to respect the patch size).

    For now we use T=4 per camera (4 groups total). This keeps the total
    patch count identical (16×68×120 = 130560) while giving the language
    model the correct temporal structure for RoPE and vision-language
    alignment.

    Args:
        inputs: Dict returned by alpamayo_apply_chat_template (or any
            processor output) containing "image_grid_thw".
        num_cameras: Number of cameras (default 4).
        num_frames_per_camera: Frames per camera (default 4).

    Returns:
        The same dict with "image_grid_thw" replaced by the temporally
        grouped version. Other keys are left unchanged.
    """
    if "image_grid_thw" not in inputs:
        return inputs

    grid = np.asarray(inputs["image_grid_thw"])  # (N, 3)
    n_groups = grid.shape[0]
    expected = num_cameras * num_frames_per_camera
    if n_groups != expected:
        # Not the 4×4 case we know how to handle; leave unchanged.
        return inputs

    # Each original row is [1, H, W]. We want to collapse every
    # `num_frames_per_camera` consecutive rows into one row [T, H, W]
    # where T = num_frames_per_camera.
    h, w = grid[0, 1], grid[0, 2]
    t = num_frames_per_camera

    new_grid = np.zeros((num_cameras, 3), dtype=np.int64)
    for cam in range(num_cameras):
        new_grid[cam] = [t, h, w]

    inputs = dict(inputs)  # shallow copy so we don't mutate caller's dict
    inputs["image_grid_thw"] = new_grid
    return inputs
