"""
Strict detokenizer for Alpamayo MLX port.

Replaces the default SPMStreamingDetokenizer so that any token ID outside
the expected vocabulary causes an immediate hard exit.

This is intentional: an out-of-range token during generation means the
tokenizer extension + embedding resize has not fully propagated into the
mlx-vlm generation pipeline, and we want to surface that immediately.
"""

import sys
from typing import List

from mlx_vlm.tokenizer_utils import SPMStreamingDetokenizer


class StrictSPMDetokenizer(SPMStreamingDetokenizer):
    """SPM detokenizer that exits the process if a generated token ID
    is out of range for the known vocabulary.

    Used during development to verify that Alpamayo's extended vocabulary
    (155697 tokens) is correctly recognized by the entire generation stack.
    """

    def __init__(self, tokenizer, trim_space=True, model_vocab_size: int | None = None):
        self.trim_space = trim_space
        vocab_size = model_vocab_size or len(tokenizer)
        self.tokenmap = [None] * vocab_size
        self.is_byte_token = [False] * vocab_size
        self.byte_value = [0] * vocab_size

        for value, tokenid in tokenizer.vocab.items():
            if 0 <= tokenid < vocab_size:
                self.tokenmap[tokenid] = value
                if value.startswith("<0x") and len(value) >= 6 and value[5] == ">":
                    self.is_byte_token[tokenid] = True
                    self.byte_value[tokenid] = int(value[3:5], 16)

        self.reset()

    def add_token(self, token: int, skip_special_token_ids: List[int] = []):
        if token in skip_special_token_ids:
            return
        if token >= len(self.tokenmap) or self.tokenmap[token] is None:
            print(
                f"\n[mlx-port][FATAL] Token ID {token} is outside the detokenizer's "
                f"tokenmap (size={len(self.tokenmap)})."
            )
            print(
                "[mlx-port][FATAL] This means the tokenizer extension / padding step "
                "did not take effect inside the mlx-vlm generation pipeline."
            )
            print("[mlx-port][FATAL] Exiting immediately as requested.")
            sys.exit(1)
        super().add_token(token, skip_special_token_ids)
