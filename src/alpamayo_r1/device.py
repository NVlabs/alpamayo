"""Device selection shared by the inference entry points (XPU, then CUDA, then CPU).

An explicit device always wins over detection and is never swapped for another backend.
"""

import contextlib
import importlib.util
import logging

import torch

logger = logging.getLogger(__name__)

_ACCELERATORS = ("xpu", "cuda")
_warned_attn_fallback = False


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this torch build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def default_device_type() -> str:
    """Best available backend: ``xpu``, then ``cuda``, then ``cpu``."""
    for device_type in _ACCELERATORS:
        if is_available(device_type):
            return device_type
    return "cpu"


def resolve_device(device: str | torch.device | None = None) -> torch.device:
    """Return ``device`` verbatim, or the detected default (index 0) when it is ``None``."""
    if device is None or device == "":
        device_type = default_device_type()
        return torch.device("cpu") if device_type == "cpu" else torch.device(device_type, 0)
    return torch.device(device)


def manual_seed_all(seed: int, device_type: str | None = None) -> None:
    """Seed the RNGs of the given (or detected) accelerator backend; no-op on CPU."""
    device_type = device_type or default_device_type()
    if device_type == "xpu" and is_available("xpu"):
        torch.xpu.manual_seed_all(seed)
    elif device_type == "cuda" and is_available("cuda"):
        torch.cuda.manual_seed_all(seed)


def resolve_attn_implementation(requested: str) -> str:
    """Fall back from flash attention to SDPA when it cannot run on this host."""
    global _warned_attn_fallback
    if requested != "flash_attention_2":
        return requested
    if is_available("cuda") and importlib.util.find_spec("flash_attn") is not None:
        return requested
    if not _warned_attn_fallback:
        logger.warning("flash_attention_2 is not usable on this host, using sdpa instead.")
        _warned_attn_fallback = True
    return "sdpa"


@contextlib.contextmanager
def autocast_disabled():
    """Disable autocast on the CPU and every accelerator backend present (also a decorator)."""
    with contextlib.ExitStack() as stack:
        for device_type in ("cpu", *_ACCELERATORS):
            if is_available(device_type):
                stack.enter_context(torch.autocast(device_type, enabled=False))
        yield
