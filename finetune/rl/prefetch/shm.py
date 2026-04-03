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

"""This module exists purely to keep `data_prefetch.py` readable:
- server packs objects into (meta shm + extra shm blocks for large payloads)
- client reconstructs objects and manages shm handle lifetimes for true 0-copy views
"""

from __future__ import annotations

import pickle
import threading
import weakref
from multiprocessing import resource_tracker
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import numpy as np
import torch

# Marker used in pickled metadata objects to represent "stored in shm as-is".
SHM_TAG = "__alpamayo_shm__"


def torch_dtype_from_str(s: str):
    # torch.dtype stringifies like "torch.float32"
    if s.startswith("torch."):
        return getattr(torch, s.split(".", 1)[1], None)
    return None


def contains_cuda_tensor(x: Any) -> bool:
    seen: set[int] = set()

    def walk(v: Any) -> bool:
        oid = id(v)
        if oid in seen:
            return False
        seen.add(oid)
        if isinstance(v, torch.Tensor):
            return bool(v.is_cuda)
        if isinstance(v, dict):
            return any(walk(t) for t in v.values())
        if isinstance(v, (list, tuple)):
            return any(walk(t) for t in v)
        return False

    return walk(x)


def alloc_shm_bytes(*, nbytes: int) -> SharedMemory:
    if nbytes <= 0:
        raise ValueError("nbytes must be > 0")
    shm = SharedMemory(create=True, size=int(nbytes))
    try:
        resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
    except (KeyError, ValueError):
        pass
    return shm


def pack_obj(obj: Any) -> tuple[Any, list[str]]:
    """Return (meta_obj, extra_shm_names)."""
    extra: list[str] = []
    _INLINE_THRESHOLD = 1 << 20  # 1 MiB

    if isinstance(obj, torch.Tensor):
        if bool(obj.is_cuda):
            raise RuntimeError("Server fetched CUDA tensors; not supported for shm")
        t = obj.detach().contiguous().cpu()
        nbytes = int(t.numel() * t.element_size())
        if nbytes == 0 or nbytes < _INLINE_THRESHOLD:
            return t, extra
        shm = alloc_shm_bytes(nbytes=nbytes)
        try:
            shm.buf[:nbytes] = t.numpy().tobytes()
        finally:
            shm.close()
        nm = str(shm.name)
        extra.append(nm)
        return (
            {
                SHM_TAG: "torch_tensor",
                "dtype": str(t.dtype),
                "shape": tuple(int(x) for x in t.shape),
                "nbytes": nbytes,
                "shm_name": nm,
            },
            extra,
        )

    if isinstance(obj, np.ndarray):
        arr = np.ascontiguousarray(obj)
        nbytes = int(arr.nbytes)
        if nbytes < _INLINE_THRESHOLD:
            return arr, extra
        shm = alloc_shm_bytes(nbytes=nbytes)
        try:
            shm.buf[:nbytes] = arr.view(np.uint8).tobytes()
        finally:
            shm.close()
        nm = str(shm.name)
        extra.append(nm)
        return (
            {
                SHM_TAG: "numpy_array",
                "dtype": str(arr.dtype),
                "shape": tuple(int(x) for x in arr.shape),
                "nbytes": nbytes,
                "shm_name": nm,
            },
            extra,
        )

    if isinstance(obj, (bytes, bytearray)) and len(obj) > 0:
        b = bytes(obj)
        nbytes = int(len(b))
        if nbytes < _INLINE_THRESHOLD:
            return b, extra
        shm = alloc_shm_bytes(nbytes=nbytes)
        try:
            shm.buf[:nbytes] = b
        finally:
            shm.close()
        nm = str(shm.name)
        extra.append(nm)
        return ({SHM_TAG: "bytes", "nbytes": nbytes, "shm_name": nm}, extra)

    if isinstance(obj, dict):
        out: dict[Any, Any] = {}
        for k, v in obj.items():
            mv, ex = pack_obj(v)
            out[k] = mv
            extra.extend(ex)
        return out, extra
    if isinstance(obj, list):
        out_l: list[Any] = []
        for v in obj:
            mv, ex = pack_obj(v)
            out_l.append(mv)
            extra.extend(ex)
        return out_l, extra
    if isinstance(obj, tuple):
        out_t: list[Any] = []
        for v in obj:
            mv, ex = pack_obj(v)
            out_t.append(mv)
            extra.extend(ex)
        return tuple(out_t), extra

    return obj, extra


def shm_put(obj: Any) -> tuple[str, int, tuple[str, ...]]:
    """Store obj into shm as: pickled meta + extra shm blocks for large tensors/arrays/bytes."""
    meta_obj, extra = pack_obj(obj)
    b = pickle.dumps(meta_obj, protocol=pickle.HIGHEST_PROTOCOL)
    shm = SharedMemory(create=True, size=len(b))
    try:
        resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
    except (KeyError, ValueError):
        pass
    shm.buf[: len(b)] = b
    shm.close()
    return str(shm.name), int(len(b)), tuple(extra)


def shm_unlink_quiet(name: str) -> None:
    try:
        shm = SharedMemory(name=str(name), create=False)
    except FileNotFoundError:
        return
    try:
        shm.unlink()
    except FileNotFoundError:
        pass
    finally:
        shm.close()


def read_shm_bytes(*, shm_name: str, nbytes: int) -> bytes:
    shm = SharedMemory(name=str(shm_name), create=False)
    try:
        resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
    except (KeyError, ValueError):
        pass
    try:
        return bytes(shm.buf[: int(nbytes)])
    finally:
        shm.close()


# ---- Client-side shm handle registry for true 0-copy views ----
_CLIENT_SHM_LOCK = threading.Lock()
_CLIENT_SHM_HANDLES: dict[str, tuple[SharedMemory, int]] = {}


def client_shm_acquire(shm_name: str) -> SharedMemory:
    nm = str(shm_name)
    with _CLIENT_SHM_LOCK:
        for k, (h, rc0) in list(_CLIENT_SHM_HANDLES.items()):
            if int(rc0) != 0:
                continue
            try:
                h.close()
            except BufferError:
                continue
            except Exception:
                continue
            else:
                _CLIENT_SHM_HANDLES.pop(k, None)
        ent = _CLIENT_SHM_HANDLES.get(nm)
        if ent is None:
            shm = SharedMemory(name=nm, create=False)
            try:
                resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
            except (KeyError, ValueError):
                pass
            _CLIENT_SHM_HANDLES[nm] = (shm, 1)
            return shm
        shm, rc = ent
        _CLIENT_SHM_HANDLES[nm] = (shm, int(rc) + 1)
        return shm


def client_shm_release(shm_name: str) -> None:
    nm = str(shm_name)
    with _CLIENT_SHM_LOCK:
        ent = _CLIENT_SHM_HANDLES.get(nm)
        if ent is None:
            return
        shm, rc = ent
        rc2 = int(rc) - 1
        if rc2 > 0:
            _CLIENT_SHM_HANDLES[nm] = (shm, rc2)
            return
        _CLIENT_SHM_HANDLES[nm] = (shm, 0)
        try:
            shm.close()
        except BufferError:
            return
        except Exception:
            return
        else:
            _CLIENT_SHM_HANDLES.pop(nm, None)


def client_shm_close_all() -> None:
    with _CLIENT_SHM_LOCK:
        for nm, (shm, _rc) in list(_CLIENT_SHM_HANDLES.items()):
            try:
                shm.close()
            except BufferError:
                _CLIENT_SHM_HANDLES[nm] = (shm, 0)
            except Exception:
                _CLIENT_SHM_HANDLES[nm] = (shm, 0)
            else:
                _CLIENT_SHM_HANDLES.pop(nm, None)


def client_shm_view(*, shm_name: str, nbytes: int) -> memoryview:
    shm = client_shm_acquire(str(shm_name))
    return shm.buf[: int(nbytes)]


def shm_unpack_client(obj: Any) -> Any:
    if isinstance(obj, dict) and SHM_TAG in obj:
        kind = obj.get(SHM_TAG)
        if kind == "torch_tensor":
            dtype = torch_dtype_from_str(str(obj.get("dtype", "")))
            if dtype is None:
                raise ValueError(f"Unknown torch dtype in shm meta: {obj.get('dtype')!r}")
            shape = tuple(int(x) for x in obj.get("shape", ()))
            nbytes = int(obj.get("nbytes", 0))
            shm_name = str(obj.get("shm_name", ""))
            buf = client_shm_view(shm_name=shm_name, nbytes=nbytes)
            elem_sz = int(torch.empty((), dtype=dtype).element_size())
            if elem_sz <= 0 or (int(nbytes) % int(elem_sz)) != 0:
                raise ValueError(f"Invalid tensor nbytes for dtype: nbytes={nbytes} dtype={dtype}")
            numel = int(nbytes) // int(elem_sz)
            t = torch.frombuffer(buf, dtype=dtype, count=int(numel))
            out = t.reshape(shape)
            try:
                weakref.finalize(out, client_shm_release, str(shm_name))
            except TypeError:
                return out
            return out
        if kind == "numpy_array":
            dtype = np.dtype(str(obj.get("dtype", "uint8")))
            shape = tuple(int(x) for x in obj.get("shape", ()))
            nbytes = int(obj.get("nbytes", 0))
            shm_name = str(obj.get("shm_name", ""))
            buf = client_shm_view(shm_name=shm_name, nbytes=nbytes)
            arr_u8 = np.frombuffer(buf, dtype=np.uint8, count=int(nbytes))
            arr = arr_u8.view(dtype).reshape(shape)
            try:
                weakref.finalize(arr, client_shm_release, str(shm_name))
            except TypeError:
                return arr
            return arr
        if kind == "bytes":
            nbytes = int(obj.get("nbytes", 0))
            shm_name = str(obj.get("shm_name", ""))
            return read_shm_bytes(shm_name=shm_name, nbytes=nbytes)
        return obj
    if isinstance(obj, dict):
        return {k: shm_unpack_client(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [shm_unpack_client(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(shm_unpack_client(v) for v in obj)
    return obj
