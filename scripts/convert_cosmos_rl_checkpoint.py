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

r"""Convert a Cosmos-RL policy checkpoint into a HuggingFace-compatible checkpoint directory.

Cosmos-RL policy checkpoints (e.g. ``.../checkpoints/step_460/policy/``) are saved as
per-rank PyTorch files (``model_rank_<r>.pth``) containing DTensor shards. They are
meant for Cosmos-RL resume and cannot be directly loaded via
``ReasoningVLA.from_pretrained()``.

This script:
  1. Merges the DTensor shards across ranks on CPU.
  2. Copies non-weight files (config.json, tokenizer, processor) from a base HF
     checkpoint directory (the training-ready checkpoint produced by
     ``convert_release_config_to_training.py``).
  3. Writes sharded ``*.safetensors`` + ``model.safetensors.index.json``.

The resulting directory can be loaded with ``ReasoningVLA.from_pretrained()``.

Usage::

    python scripts/convert_cosmos_rl_checkpoint.py \
      --cosmos-policy-ckpt "/path/to/checkpoints/step_<N>/policy" \
      --base-hf-ckpt "$ALPAMAYO_MODEL_DIR" \
      --output-dir "/path/to/exported/step_<N>_hf"

Then load the model::

    from rl.models.reasoning_vla.base_model import RLWrapperReasoningVLA
    model = RLWrapperReasoningVLA.from_pretrained("/path/to/exported/step_<N>_hf")
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch.distributed.tensor import DTensor, Replicate, Shard

MODEL_KEY_PREFIX = "reasoning_vla."


def _parse_size_to_bytes(s: str) -> int:
    """Parse human-readable sizes like ``'4GB'``, ``'500MB'`` into bytes."""
    s = str(s).strip()
    if not s:
        raise ValueError("Empty size string")
    m = re.fullmatch(r"(?i)\s*(\d+(?:\.\d+)?)\s*([kmgt]?b)?\s*", s)
    if m is None:
        raise ValueError(f"Invalid size: {s!r} (expected e.g. '4GB', '500MB', '1234')")
    value = float(m.group(1))
    unit = (m.group(2) or "B").upper()
    scale = {"B": 1, "KB": 1024, "MB": 1024**2, "GB": 1024**3, "TB": 1024**4}[unit]
    out = int(value * scale)
    if out <= 0:
        raise ValueError(f"Size must be > 0, got {s!r}")
    return out


def _rank_from_filename(name: str) -> int:
    """Extract rank index from a ``model_rank_<r>.pth`` filename."""
    m = re.search(r"model_rank_(\d+)\.pth$", name)
    if m is None:
        raise ValueError(f"Not a model_rank file: {name!r}")
    return int(m.group(1))


# ---------------------------------------------------------------------------
# File filtering – separate weight files from config/tokenizer/processor
# ---------------------------------------------------------------------------

_WEIGHT_FILENAMES = {
    "pytorch_model.bin",
    "pytorch_model.bin.index.json",
    "model.safetensors",
    "model.safetensors.index.json",
}


def _is_weight_file(name: str) -> bool:
    """Return True if *name* is a model weight or weight-index file."""
    if name.endswith(".safetensors"):
        return True
    if name in _WEIGHT_FILENAMES:
        return True
    if name.startswith("model-") and (name.endswith(".bin") or name.endswith(".safetensors")):
        return True
    if name.endswith(".index.json") and ("safetensors" in name or "pytorch_model" in name):
        return True
    return False


def _copy_non_weight_files(*, src_dir: Path, dst_dir: Path, overwrite: bool) -> list[str]:
    """Copy config, tokenizer, and processor files from *src_dir* to *dst_dir*."""
    if not src_dir.exists():
        raise FileNotFoundError(f"base_hf_ckpt not found: {str(src_dir)!r}")
    if not (src_dir / "config.json").exists():
        raise FileNotFoundError(f"base_hf_ckpt is missing config.json: {str(src_dir)!r}")
    copied: list[str] = []
    for p in sorted(src_dir.iterdir()):
        if not p.is_file():
            continue
        if _is_weight_file(p.name):
            continue
        dst = dst_dir / p.name
        if dst.exists() and not overwrite:
            continue
        shutil.copy2(p, dst)
        copied.append(p.name)
    return copied


# ---------------------------------------------------------------------------
# DTensor merging
# ---------------------------------------------------------------------------


def _ordered_mesh_ranks(v: DTensor, available_ranks: list[int]) -> list[int]:
    """Best-effort: order shard ranks using DTensor device_mesh ordering."""
    try:
        mesh = v.device_mesh.mesh
        mesh_ranks = [int(x) for x in mesh.flatten().tolist()]
    except Exception:
        mesh_ranks = list(available_ranks)
    ordered = [r for r in mesh_ranks if r in set(available_ranks)]
    if len(ordered) != len(available_ranks):
        ordered = list(available_ranks)
    return ordered


def _assemble_full_tensor(
    *,
    key: str,
    rank_state_dicts: dict[int, dict[str, Any]],
    ranks: list[int],
) -> torch.Tensor:
    """Merge per-rank DTensor shards for a single key into one full tensor."""
    v0 = rank_state_dicts[ranks[0]][key]

    if isinstance(v0, DTensor):
        placements = v0.placements
        if len(placements) != 1:
            raise NotImplementedError(
                f"Unsupported placements for {key}: {placements} (expected 1D mesh)"
            )
        p0 = placements[0]
        if isinstance(p0, Replicate):
            t = v0.to_local()
            assert isinstance(t, torch.Tensor)
            return t
        if not isinstance(p0, Shard):
            raise NotImplementedError(
                f"Unsupported placement for {key}: {placements} (expected Shard/Replicate)"
            )
        dim = int(p0.dim)
        ordered_ranks = _ordered_mesh_ranks(v0, ranks)
        locals_t: list[torch.Tensor] = []
        for r in ordered_ranks:
            vr = rank_state_dicts[r][key]
            if not isinstance(vr, DTensor):
                raise TypeError(f"Rank {r} value type mismatch for {key}: {type(vr).__name__}")
            locals_t.append(vr.to_local())
        return torch.cat(locals_t, dim=dim)

    if isinstance(v0, torch.Tensor):
        return v0

    raise TypeError(f"Unsupported value type for {key}: {type(v0).__name__}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="Convert a Cosmos-RL policy checkpoint to a HuggingFace checkpoint.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--cosmos-policy-ckpt",
        required=True,
        type=str,
        help="Path to Cosmos policy ckpt dir (e.g. .../checkpoints/step_460/policy).",
    )
    parser.add_argument(
        "--base-hf-ckpt",
        required=True,
        type=str,
        help=(
            "Path to the training-ready HF checkpoint dir (output of "
            "convert_release_config_to_training.py).  Config, tokenizer, and "
            "processor files are copied from here."
        ),
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=str,
        help="Output directory for the exported HF checkpoint.",
    )
    parser.add_argument(
        "--max-shard-size",
        default="4GB",
        type=str,
        help="Maximum safetensors shard size (e.g. '4GB', '500MB').",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output_dir.",
    )
    args = parser.parse_args()

    cosmos_ckpt_dir = Path(args.cosmos_policy_ckpt)
    base_hf_dir = Path(args.base_hf_ckpt)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_bytes = _parse_size_to_bytes(args.max_shard_size)

    # -- 1. Copy non-weight files from the base HF checkpoint --
    copied = _copy_non_weight_files(src_dir=base_hf_dir, dst_dir=out_dir, overwrite=args.overwrite)
    print(f"Copied {len(copied)} config/tokenizer/processor files from {base_hf_dir}")

    # -- 2. Load per-rank model shards --
    rank_paths = sorted(
        cosmos_ckpt_dir.glob("model_rank_*.pth"),
        key=lambda p: _rank_from_filename(p.name),
    )
    if not rank_paths:
        raise FileNotFoundError(f"No model_rank_*.pth found under {str(cosmos_ckpt_dir)!r}")

    print(f"Loading {len(rank_paths)} rank shards from {cosmos_ckpt_dir} ...")
    rank_state_dicts: dict[int, dict[str, Any]] = {}
    for p in rank_paths:
        r = _rank_from_filename(p.name)
        rank_state_dicts[r] = torch.load(p, map_location="cpu", weights_only=False)
    ranks = sorted(rank_state_dicts.keys())
    print(f"  ranks: {ranks}")

    # -- 3. Validate key consistency across ranks --
    keys0 = list(rank_state_dicts[ranks[0]].keys())
    key_set0 = set(keys0)
    for r in ranks[1:]:
        ks = set(rank_state_dicts[r].keys())
        if ks != key_set0:
            missing = sorted(list(key_set0 - ks))[:10]
            extra = sorted(list(ks - key_set0))[:10]
            raise RuntimeError(
                f"Rank key mismatch: rank0={len(key_set0)} rank{r}={len(ks)} "
                f"missing(sample)={missing} extra(sample)={extra}"
            )

    keys = sorted(keys0)
    print(f"  total keys: {len(keys)}")

    # -- 4. Merge shards and write safetensors --
    weight_map: dict[str, str] = {}
    total_size_bytes = 0
    shard_idx = 1
    chunk: dict[str, torch.Tensor] = {}
    chunk_bytes = 0

    def _shard_name(idx: int) -> str:
        return f"model-{idx:05d}.safetensors"

    def flush() -> None:
        nonlocal shard_idx, chunk, chunk_bytes
        if not chunk:
            return
        name = _shard_name(shard_idx)
        save_file(chunk, str(out_dir / name))
        print(f"  wrote {name} ({len(chunk)} tensors, {chunk_bytes / 1024**3:.2f} GB)")
        shard_idx += 1
        chunk = {}
        chunk_bytes = 0

    for k in keys:
        full = _assemble_full_tensor(key=k, rank_state_dicts=rank_state_dicts, ranks=ranks)
        if not isinstance(full, torch.Tensor):
            raise TypeError(f"Expected Tensor for {k}, got {type(full).__name__}")
        full = full.contiguous()

        out_key = k
        if out_key.startswith(MODEL_KEY_PREFIX):
            out_key = out_key[len(MODEL_KEY_PREFIX) :]

        nbytes = int(full.numel() * full.element_size())
        if chunk and (chunk_bytes + nbytes > max_bytes):
            flush()

        chunk[out_key] = full
        chunk_bytes += nbytes
        total_size_bytes += nbytes
        weight_map[out_key] = _shard_name(shard_idx)

    flush()

    # -- 5. Write index --
    index = {
        "metadata": {"total_size": int(total_size_bytes)},
        "weight_map": weight_map,
    }
    index_path = out_dir / "model.safetensors.index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, sort_keys=True)

    # -- 6. Provenance metadata --
    num_shards = shard_idx - 1
    prov = {
        "cosmos_policy_ckpt": str(cosmos_ckpt_dir),
        "base_hf_ckpt": str(base_hf_dir),
        "ranks": ranks,
        "num_keys": len(keys),
        "max_shard_size_bytes": int(max_bytes),
        "num_shards": int(num_shards),
        "total_size_bytes": int(total_size_bytes),
    }
    with open(out_dir / "cosmos_export_provenance.json", "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, sort_keys=True)

    print(
        f"\nDone. output_dir={out_dir}\n"
        f"  shards={num_shards}  total_size={total_size_bytes / 1024**3:.2f} GB"
    )


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
