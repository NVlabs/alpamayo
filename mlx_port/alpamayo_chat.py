"""
Alpamayo Chat Template Adaptation for MLX VLM (Stage 1)

Replicates the conversation template logic from:
- src/alpamayo_r1/chat_template/conversation.py (build_conversation, construct_*)
- src/alpamayo_r1/helper.py (create_message for inference)

Key Alpamayo elements:
- System prompt: "You are a driving assistant..."
- User: Multi-camera images (with optional labels) + <|traj_history_*> placeholders + CoC/future prompt
- Assistant: Starts with <|cot_start|> for generation mode

This produces messages that can be fed to processor.apply_chat_template()
exactly as mlx-vlm / Qwen3-VL expects, preserving all special tokens.
"""

from typing import Any

# Alpamayo special tokens (subset used in inference chat template)
TRAJ_HISTORY_START = "<|traj_history_start|>"
TRAJ_HISTORY = "<|traj_history|>"
TRAJ_HISTORY_END = "<|traj_history_end|>"
COT_START = "<|cot_start|>"

SYSTEM_PROMPT = "You are a driving assistant that generates safe and accurate actions."


def create_inference_message(
    frames: list[Any],
    num_history_tokens: int = 48,
    prompt_text: str | None = None,
) -> list[dict[str, Any]]:
    """
    Build the 3-turn conversation used for VLM rollout / inference.

    Mirrors helper.create_message() but returns a clean list of dicts
    compatible with mlx-vlm's apply_chat_template.

    Args:
        frames: list of image arrays / PIL images (one per camera or view)
        num_history_tokens: number of <|traj_history|> padding tokens
        prompt_text: optional override for the final user text prompt

    Returns:
        messages list ready for processor.apply_chat_template(messages, ...)
    """
    if prompt_text is None:
        prompt_text = (
            "output the chain-of-thought reasoning of the driving process, "
            "then output the future trajectory."
        )

    hist_placeholder = (
        f"{TRAJ_HISTORY_START}"
        + (TRAJ_HISTORY * num_history_tokens)
        + f"{TRAJ_HISTORY_END}"
    )

    user_content = [{"type": "image", "image": frame} for frame in frames]
    user_content.append({"type": "text", "text": f"{hist_placeholder}{prompt_text}"})

    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": SYSTEM_PROMPT}],
        },
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": COT_START}],
        },
    ]


def apply_alpamayo_chat_template(processor, messages: list[dict], **kwargs) -> str:
    """
    Convenience wrapper around processor.apply_chat_template that
    ensures Alpamayo generation settings (continue_final_message=True for CoC start).
    """
    return processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        continue_final_message=True,
        **kwargs,
    )