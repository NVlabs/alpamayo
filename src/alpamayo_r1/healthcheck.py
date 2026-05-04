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

"""Smoke test for an alpamayo_r1 install.

Run as::

    python -m alpamayo_r1.healthcheck

Each check is independent and prints PASS / FAIL / SKIP with a reason.
The process exit code is the number of failed checks (0 means all green;
non-zero is the count of failures so CI can inspect it without parsing
stdout). Skipped checks (e.g. no GPU on a laptop) are not counted as
failures.

Why this exists: the most-asked support questions on the repo boil down
to "did I install this right?" -- CUDA visible? flash-attn compiled?
HuggingFace authenticated? PAI dataset readable? -- and the answer is
not obvious until inference fails 30 seconds into a long run. This
script answers all of those in a few seconds without loading the model.
"""

from __future__ import annotations

import argparse
import importlib
import os
import platform
import sys
from dataclasses import dataclass
from typing import Callable, Literal

REQUIRED_PYTHON = (3, 12)
REQUIRED_TRANSFORMERS = "4.57.1"
REQUIRED_TORCH = "2.8"

Status = Literal["pass", "fail", "skip"]


@dataclass(frozen=True)
class CheckResult:
    """Outcome of a single health check."""

    name: str
    status: Status
    detail: str


def _ok(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status="pass", detail=detail)


def _fail(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status="fail", detail=detail)


def _skip(name: str, detail: str) -> CheckResult:
    return CheckResult(name=name, status="skip", detail=detail)


def check_python_version() -> CheckResult:
    """The repo declares ``requires-python = "==3.12.*"`` in pyproject.toml."""
    actual = sys.version_info
    expected_major, expected_minor = REQUIRED_PYTHON
    actual_str = f"{actual.major}.{actual.minor}.{actual.micro}"
    if (actual.major, actual.minor) == REQUIRED_PYTHON:
        return _ok("python", f"{actual_str} on {platform.platform()}")
    return _fail(
        "python",
        f"got {actual_str}, but pyproject.toml requires "
        f"=={expected_major}.{expected_minor}.*",
    )


def check_torch() -> CheckResult:
    """Torch import + reported version."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001 - we want to surface anything
        return _fail("torch", f"import failed: {exc!r}")
    detail = f"torch=={torch.__version__}"
    if not torch.__version__.startswith(REQUIRED_TORCH):
        detail += (
            f" (note: pyproject.toml pins torch=={REQUIRED_TORCH}.0; "
            "training/inference may behave differently)"
        )
    return _ok("torch", detail)


def check_cuda() -> CheckResult:
    """CUDA availability + device count + visible memory on each device."""
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return _fail("cuda", f"torch import failed: {exc!r}")
    if not torch.cuda.is_available():
        return _skip(
            "cuda",
            "torch.cuda.is_available() is False -- no GPU visible. "
            "Inference will fail; only run checks here.",
        )
    n = torch.cuda.device_count()
    devices = []
    for i in range(n):
        try:
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / (1024**3)
            devices.append(f"{i}:{props.name} ({mem_gb:.1f} GiB)")
        except Exception as exc:  # noqa: BLE001
            devices.append(f"{i}:<error: {exc!r}>")
    cuda_ver = getattr(torch.version, "cuda", None) or "?"
    return _ok("cuda", f"CUDA {cuda_ver}, {n} device(s): " + "; ".join(devices))


def check_transformers() -> CheckResult:
    """The repo pins ``transformers==4.57.1``; mismatches often break inference."""
    try:
        import transformers
    except Exception as exc:  # noqa: BLE001
        return _fail("transformers", f"import failed: {exc!r}")
    actual = transformers.__version__
    if actual == REQUIRED_TRANSFORMERS:
        return _ok("transformers", f"{actual} (matches pinned version)")
    return _fail(
        "transformers",
        f"got {actual}, pyproject.toml pins {REQUIRED_TRANSFORMERS}",
    )


def check_flash_attn() -> CheckResult:
    """flash-attn is the default attention backend; SDPA fallback is documented."""
    try:
        flash_attn = importlib.import_module("flash_attn")
    except Exception as exc:  # noqa: BLE001
        return _skip(
            "flash_attn",
            f"flash_attn not importable ({exc!r}). "
            "Inference still works via attn_implementation='sdpa' (see README).",
        )
    version = getattr(flash_attn, "__version__", "<unknown>")
    return _ok("flash_attn", f"flash_attn=={version}")


def check_physical_ai_av() -> CheckResult:
    """The dataset loader depends on the ``physical_ai_av`` package."""
    try:
        physical_ai_av = importlib.import_module("physical_ai_av")
    except Exception as exc:  # noqa: BLE001
        return _fail(
            "physical_ai_av",
            f"import failed: {exc!r}. "
            "Run `uv sync --active` to install it from PyPI.",
        )
    version = getattr(physical_ai_av, "__version__", "<unknown>")
    return _ok("physical_ai_av", f"version={version}")


def check_huggingface_auth() -> CheckResult:
    """Both the model and dataset are gated; without auth they 401 silently late."""
    try:
        from huggingface_hub import HfApi
    except Exception as exc:  # noqa: BLE001
        return _fail("hf_auth", f"huggingface_hub import failed: {exc!r}")
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    try:
        api = HfApi()
        info = api.whoami(token=token) if token else api.whoami()
    except Exception as exc:  # noqa: BLE001
        return _fail(
            "hf_auth",
            f"HuggingFace auth failed ({exc!r}). "
            "Run `hf auth login` and accept the gated terms for "
            "nvidia/Alpamayo-R1-10B and nvidia/PhysicalAI-Autonomous-Vehicles.",
        )
    user = info.get("name", "<unknown>") if isinstance(info, dict) else "<unknown>"
    return _ok("hf_auth", f"authenticated as {user!r}")


def check_alpamayo_r1_import() -> CheckResult:
    """The package itself must be importable."""
    try:
        importlib.import_module("alpamayo_r1")
    except Exception as exc:  # noqa: BLE001
        return _fail(
            "alpamayo_r1",
            f"import failed: {exc!r}. "
            "Did you `uv sync --active` from the repo root?",
        )
    return _ok("alpamayo_r1", "package importable")


# Ordered list. ``check_alpamayo_r1_import`` runs last so that earlier
# checks (which only touch upstream deps) still surface even when the
# package itself is broken.
ALL_CHECKS: tuple[Callable[[], CheckResult], ...] = (
    check_python_version,
    check_torch,
    check_cuda,
    check_transformers,
    check_flash_attn,
    check_physical_ai_av,
    check_huggingface_auth,
    check_alpamayo_r1_import,
)


def _format_line(r: CheckResult) -> str:
    glyph = {"pass": "[ OK ]", "fail": "[FAIL]", "skip": "[SKIP]"}[r.status]
    return f"{glyph} {r.name:<16} {r.detail}"


def run(checks: tuple[Callable[[], CheckResult], ...] | None = None) -> list[CheckResult]:
    """Run every check in order. Each check is isolated; one failure does not stop the rest.

    ``checks`` defaults to the module-level ``ALL_CHECKS`` tuple, looked up at
    call time so tests / callers can monkey-patch the module attribute and
    have ``run()`` (and therefore ``main()``) pick up the override.
    """
    if checks is None:
        checks = ALL_CHECKS
    results: list[CheckResult] = []
    for check in checks:
        try:
            results.append(check())
        except Exception as exc:  # noqa: BLE001 - never let a buggy check abort the script
            results.append(_fail(check.__name__, f"check raised unexpectedly: {exc!r}"))
    return results


def main(argv: list[str] | None = None) -> int:
    """Run all checks and print a summary. Returns the number of failures."""
    parser = argparse.ArgumentParser(
        description="Smoke test the alpamayo_r1 install.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quiet",
        "-q",
        action="store_true",
        help="Only print failures (and the final summary).",
    )
    args = parser.parse_args(argv)

    results = run()
    failures = sum(1 for r in results if r.status == "fail")
    skips = sum(1 for r in results if r.status == "skip")
    passes = len(results) - failures - skips

    for r in results:
        if args.quiet and r.status == "pass":
            continue
        print(_format_line(r))

    print(
        f"\nSummary: {passes} passed, {failures} failed, {skips} skipped "
        f"({len(results)} total)."
    )
    return failures


if __name__ == "__main__":
    sys.exit(main())
