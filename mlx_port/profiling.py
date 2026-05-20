"""Lightweight profiling utilities for MLX Alpamayo inference.

Usage:
    PROFILE=1 python -m pytest mlx_port/tests/test_end_to_end_inference.py -q

When PROFILE=1 is set, the VLM generation loop and diffusion sampling loop
will print per-step memory and timing information.
"""

import ctypes
import os
import threading
import time
from contextlib import contextmanager
from typing import Optional

import mlx.core as mx

# ------------------------------------------------------------------
# Global peak memory tracker (captures spikes during long operations)
# ------------------------------------------------------------------
_global_peak_resident = 0
_global_peak_compressed = 0
_global_peak_total = 0
_global_peak_metal = 0


def record_memory_sample(label: str = "") -> dict:
    """
    Record a memory sample and update the global high-water mark.

    This should be called at strategic points inside long-running operations
    (e.g., right after a large VLM forward pass, after mx.eval, etc.) so that
    transient spikes are captured even if memory later declines.

    Returns the current memory dict for convenience.
    """
    global _global_peak_resident, _global_peak_compressed, _global_peak_total, _global_peak_metal

    mem_info = _get_macos_memory()
    resident = mem_info.get("resident", 0)
    compressed = mem_info.get("compressed", 0)
    total = mem_info.get("total", resident + compressed)

    try:
        metal = mx.metal.get_active_memory()
    except Exception:
        metal = 0

    # Update global peaks
    if resident > _global_peak_resident:
        _global_peak_resident = resident
    if compressed > _global_peak_compressed:
        _global_peak_compressed = compressed
    if total > _global_peak_total:
        _global_peak_total = total
    if metal > _global_peak_metal:
        _global_peak_metal = metal

    if label:
        print(
            f"[MEM_SAMPLE] {label}: "
            f"resident={resident/1e9:.2f}GB  "
            f"compressed={compressed/1e9:.2f}GB  "
            f"total={total/1e9:.2f}GB  "
            f"metal={metal/1e9:.2f}GB"
        )

    return mem_info


def get_global_memory_peak() -> dict:
    """Return the highest memory values observed since the process started."""
    return {
        "resident": _global_peak_resident,
        "compressed": _global_peak_compressed,
        "total": _global_peak_total,
        "metal": _global_peak_metal,
    }


def reset_global_memory_peak() -> None:
    """Reset the global peak counters (useful between test runs)."""
    global _global_peak_resident, _global_peak_compressed, _global_peak_total, _global_peak_metal
    _global_peak_resident = 0
    _global_peak_compressed = 0
    _global_peak_total = 0
    _global_peak_metal = 0


# ------------------------------------------------------------------
# Background memory monitor (captures peaks during long operations)
# ------------------------------------------------------------------
class MemoryMonitor:
    """
    Context manager that starts a background thread polling memory at high
    frequency while the enclosed operation runs.

    This is useful for capturing transient spikes inside long-running calls
    (e.g., a large VLM forward pass) where memory may peak and then drop
    before the next explicit sample.
    """

    def __init__(self, poll_interval: float = 0.05, label: str = ""):
        self.poll_interval = poll_interval
        self.label = label
        self._stop_event = threading.Event()
        self._thread = None

    def __enter__(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        # One final sample after the operation to catch any late peak
        record_memory_sample(f"{self.label}_final" if self.label else "monitor_final")

    def _monitor_loop(self):
        while not self._stop_event.is_set():
            record_memory_sample(self.label)
            time.sleep(self.poll_interval)


# ------------------------------------------------------------------
# macOS Mach task memory reporting (matches Activity Monitor "Memory")
# ------------------------------------------------------------------
def _get_macos_memory() -> dict:
    """
    Return current process memory statistics that closely match
    Activity Monitor's "Memory" column on macOS.

    Uses the Mach `task_info(TASK_VM_INFO_REV1)` API.
    Keys returned:
        resident   - resident (dirty) pages in bytes
        compressed - compressed pages in bytes
        total      - resident + compressed (what Activity Monitor shows)
    """
    if os.uname().sysname != "Darwin":
        return {"resident": 0, "compressed": 0, "total": 0}

    try:
        libc = ctypes.CDLL("/usr/lib/libSystem.dylib", use_errno=True)

        mach_port_t = ctypes.c_uint32
        mach_msg_type_number_t = ctypes.c_uint32
        TASK_VM_INFO_REV1 = 22

        class task_vm_info_data_t(ctypes.Structure):
            _fields_ = [
                ("virtual_size", ctypes.c_uint64),
                ("region_count", ctypes.c_int32),
                ("page_size", ctypes.c_int32),
                ("resident_size", ctypes.c_uint64),
                ("resident_size_peak", ctypes.c_uint64),
                ("device", ctypes.c_uint64),
                ("device_peak", ctypes.c_uint64),
                ("internal", ctypes.c_uint64),
                ("internal_peak", ctypes.c_uint64),
                ("external", ctypes.c_uint64),
                ("external_peak", ctypes.c_uint64),
                ("reusable", ctypes.c_uint64),
                ("reusable_peak", ctypes.c_uint64),
                ("purgeable_volatile_pmap", ctypes.c_uint64),
                ("purgeable_volatile_resident", ctypes.c_uint64),
                ("purgeable_volatile_wired", ctypes.c_uint64),
                ("compressed", ctypes.c_uint64),
                ("compressed_peak", ctypes.c_uint64),
                ("compressed_lifetime", ctypes.c_uint64),
            ]

        mach_task_self = libc.mach_task_self
        mach_task_self.restype = mach_port_t

        task_info = libc.task_info
        task_info.argtypes = [
            mach_port_t,
            ctypes.c_int,
            ctypes.POINTER(task_vm_info_data_t),
            ctypes.POINTER(mach_msg_type_number_t),
        ]
        task_info.restype = ctypes.c_int

        task = mach_task_self()
        info = task_vm_info_data_t()
        count = mach_msg_type_number_t(ctypes.sizeof(info) // 4)

        kr = task_info(task, TASK_VM_INFO_REV1, ctypes.byref(info), ctypes.byref(count))
        if kr != 0:
            return {"resident": 0, "compressed": 0, "total": 0}

        resident = info.resident_size
        compressed = info.compressed
        return {
            "resident": resident,
            "compressed": compressed,
            "total": resident + compressed,
        }
    except Exception:
        return {"resident": 0, "compressed": 0, "total": 0}


def is_profiling_enabled() -> bool:
    """Return True if profiling is enabled via the PROFILE environment variable."""
    return os.environ.get("PROFILE", "0").lower() in ("1", "true", "yes", "on")


class StepProfiler:
    """Simple per-step profiler for memory and timing.

    Usage inside a loop:

        profiler = StepProfiler(enabled=is_profiling_enabled(), name="VLM")
        for step in range(max_steps):
            profiler.step_start(step)
            ... do work ...
            profiler.step_end()
        profiler.summary()
    """

    def __init__(self, enabled: bool = True, name: str = "Profiler"):
        self.enabled = enabled
        self.name = name
        self.steps = 0
        self.start_time = 0.0
        self.peak_memory = 0          # Metal active memory
        self.peak_rss = 0             # system RSS (bytes)
        self._baseline_mem = None     # Metal memory at end of previous step
        self._baseline_rss = None     # RSS at end of previous step
        self.times = []

    def _get_rss(self) -> int:
        """Return the memory figure that most closely matches Activity Monitor.

        On macOS we use the Mach task_info API and return (resident + compressed).
        On other platforms we fall back to 0.
        """
        mem = _get_macos_memory()
        return mem.get("total", 0)

    def step_start(self, step: int):
        if not self.enabled:
            return
        self.start_time = time.perf_counter()
        if self._baseline_mem is None:
            self._baseline_mem = mx.metal.get_active_memory()
            self._baseline_rss = self._get_rss()

    def step_end(self):
        if not self.enabled:
            return
        elapsed = time.perf_counter() - self.start_time
        current_mem = mx.metal.get_active_memory()
        current_rss = self._get_rss()
        delta_mem = current_mem - self._baseline_mem
        delta_rss = current_rss - self._baseline_rss
        self.times.append(elapsed)
        self.peak_memory = max(self.peak_memory, current_mem)
        self.peak_rss = max(self.peak_rss, current_rss)
        self.steps += 1

        if self.steps <= 5 or self.steps % 8 == 0:
            mem_info = _get_macos_memory()
            print(
                f"[{self.name}] step={self.steps:3d}  "
                f"time={elapsed*1000:6.1f}ms  "
                f"mem={current_mem/1e9:5.2f}GB  "
                f"delta_mem={delta_mem/1e9:+6.2f}GB  "
                f"rss={current_rss/1e9:5.2f}GB (AM total)  "
                f"resident={mem_info.get('resident', 0)/1e9:.2f}GB  "
                f"compressed={mem_info.get('compressed', 0)/1e9:.2f}GB"
            )

        self._baseline_mem = current_mem
        self._baseline_rss = current_rss

    def summary(self):
        if not self.enabled or self.steps == 0:
            return
        avg_time = sum(self.times) / len(self.times) * 1000
        print(
            f"[{self.name}] Summary: {self.steps} steps, "
            f"avg_time={avg_time:.1f}ms, "
            f"peak_mem={self.peak_memory/1e9:.2f}GB, "
            f"peak_rss={self.peak_rss/1e9:.2f}GB (Activity Monitor total)"
        )


@contextmanager
def profile_section(name: str, enabled: Optional[bool] = None):
    """Context manager for timing and memory a whole section."""
    if enabled is None:
        enabled = is_profiling_enabled()
    if not enabled:
        yield
        return

    start = time.perf_counter()
    mem_before = mx.metal.get_active_memory()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        mem_after = mx.metal.get_active_memory()
        print(
            f"[{name}] time={elapsed*1000:.1f}ms  "
            f"mem_before={mem_before/1e9:.2f}GB  "
            f"mem_after={mem_after/1e9:.2f}GB"
        )
