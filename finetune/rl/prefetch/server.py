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


"""Node-level shared-memory prefetch.

Design goals:
- One node = one server process (per split) does dataset fetch + heavy preprocessing.
- Policy/Rollout processes read results from shared memory (no per-rank waste).
- Keep the code minimal and fail loudly (no broad try/except).

Note: some training loops reuse logical indices across epochs (e.g. 0..N-1, then wrap to 0).
This implementation treats logical indices as a ring of size `len(dataset)` and can prefetch
"the next" sample for each active shard residue class.
"""

from __future__ import annotations

import atexit
import copy
import fcntl
import hashlib
import os
import pickle
import queue
import signal
import socket
import struct
import sys
import threading
import time
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from multiprocessing import get_context, resource_tracker
from multiprocessing.shared_memory import SharedMemory
from typing import Any

import torch

# Keep the shm packing/unpacking code out of this file to improve readability.
from rl.prefetch.shm import (
    client_shm_close_all as _client_shm_close_all,
    contains_cuda_tensor as _contains_cuda_tensor,
    shm_put,
    shm_unlink_quiet as _shm_unlink_quiet,
    shm_unpack_client as _shm_unpack_client,
)

# Keep rollout extras (same as packer_prefetch).
ROLLOUT_KEEP_KEYS: tuple[str, ...] = (
    "ego_future_xyz",
    "ego_future_rot",
    "ego_history_xyz",
    "ego_history_rot",
    "cot",
    "meta_action_strings",
)

# shm marker is imported from `prefetch_shm` as `_SHM_TAG`.

# -----------------------------------------------------------------------------
# Config plumbing (Cosmos-RL passes `Config.custom`)
# -----------------------------------------------------------------------------
_alpamayo_custom_cfg: dict[str, Any] = {}
# Prefetch scheduling parameters are ALWAYS inferred from the main Cosmos config.
# We intentionally do NOT read them from `custom.alpamayo.prefetch.*` to avoid
# accidental misconfiguration and to keep the contract simple.
_PREFETCH_DISPATCH_BASE_POLICY: int = 0
_PREFETCH_DISPATCH_BASE_ROLLOUT: int = 0
_PREFETCH_LOOKAHEAD_PER_MOD_POLICY: int = 1
_PREFETCH_LOOKAHEAD_PER_MOD_ROLLOUT: int = 1


def _is_controller_role() -> bool:
    """Return True when the current process is the Cosmos controller (not policy/rollout)."""
    role = str(os.environ.get("COSMOS_ROLE", "")).strip().lower()
    return ("controller" in role) and ("policy" not in role) and ("rollout" not in role)


def set_custom_cfg(config: Any) -> None:
    """Populate module-level custom config from Cosmos-RL `Config.custom`.

    Expected shape:
    custom:
    alpamayo:
      idx_rng_enable: bool
      idx_rng_seed: int
      idx_mapper_identity: bool
      prefetch:
        capacity: int
        num_workers: int             # optional (default: 1)
        socket_dir: str               # optional
        log_level: str                # optional (default: "debug")
        # one of off/error/warn/info/debug/trace
        log_request_details: bool     # optional (default: True)
    """
    global _alpamayo_custom_cfg
    custom = getattr(config, "custom", None)
    if custom is None:
        _alpamayo_custom_cfg = {}
        return
    if hasattr(custom, "model_dump"):
        custom = custom.model_dump()
    if not isinstance(custom, Mapping):
        _alpamayo_custom_cfg = {}
        return
    alp = custom.get("alpamayo", {})
    _alpamayo_custom_cfg = dict(alp) if isinstance(alp, Mapping) else {}

    # Prefetch config (server is shared across roles for a given split/socket):
    # - Store BOTH policy/rollout dispatch bases and lookaheads so the server can schedule per-role.
    pre = _alpamayo_custom_cfg.get("prefetch")
    if not isinstance(pre, dict):
        pre = {}
        _alpamayo_custom_cfg["prefetch"] = pre

    if int(pre.get("capacity") or 0) > 0:
        global _PREFETCH_DISPATCH_BASE_POLICY
        global _PREFETCH_DISPATCH_BASE_ROLLOUT
        global _PREFETCH_LOOKAHEAD_PER_MOD_POLICY
        global _PREFETCH_LOOKAHEAD_PER_MOD_ROLLOUT

        pol = getattr(
            getattr(getattr(config, "policy", None), "parallelism", None),
            "n_init_replicas",
            None,
        )
        rol = getattr(
            getattr(getattr(config, "rollout", None), "parallelism", None),
            "n_init_replicas",
            None,
        )
        if pol is None or rol is None:
            raise RuntimeError(
                "Prefetch enabled but cannot infer dispatch bases; expected "
                "policy.parallelism.n_init_replicas and "
                "rollout.parallelism.n_init_replicas in TOML."
            )
        _PREFETCH_DISPATCH_BASE_POLICY = int(pol)
        _PREFETCH_DISPATCH_BASE_ROLLOUT = int(rol)

        rb_any = getattr(getattr(config, "rollout", None), "batch_size", None)
        train_any = getattr(config, "train", None)
        bpr_any = getattr(train_any, "train_batch_per_replica", None)
        _PREFETCH_LOOKAHEAD_PER_MOD_ROLLOUT = int(rb_any) if rb_any is not None else 1
        _PREFETCH_LOOKAHEAD_PER_MOD_POLICY = int(bpr_any) if bpr_any is not None else 1

    # If global state has already been initialized, apply prefetch wrapping immediately.
    # This keeps the "wrap in init_once" behavior while also handling the common ordering
    # where Cosmos calls Dataset.setup() (and thus sets TOML) after state.init_once().
    try:
        from rl import (
            state as alp_state,  # local import to avoid cycles at import time
        )
    except ImportError:
        return
    if not bool(getattr(alp_state, "is_initialized", lambda: False)()):
        return
    # Controller should never start or use node-prefetch. Controller is a FastAPI webserver and
    # should not spawn multiprocessing fork workers (unsafe post-CUDA / multi-threaded).
    if _is_controller_role():
        return
    dls = alp_state.get_dataloaders()
    splits = list(dls.keys()) if isinstance(dls, Mapping) else ["train", "val"]
    for s in splits:
        alp_state.maybe_enable_node_prefetch(split=str(s))


def _alpamayo_cfg_get(path: str, default: Any) -> Any:
    """Look up a dot-separated key in ``_alpamayo_custom_cfg``, returning *default* on miss."""
    cur: Any = _alpamayo_custom_cfg
    for part in path.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


# -----------------------------------------------------------------------------
# Logging (stderr)
# -----------------------------------------------------------------------------
_LOG_LEVELS: dict[str, int] = {
    "off": 100,
    "error": 40,
    "warn": 30,
    "warning": 30,
    "info": 20,
    "debug": 10,
    "trace": 5,
}


def _log_enabled(level: str) -> bool:
    """Check whether *level* is at or above the configured log threshold."""
    cfg = str(_alpamayo_cfg_get("prefetch.log_level", "debug") or "debug").lower()
    return int(_LOG_LEVELS.get(str(level).lower(), 20)) >= 0 and int(
        _LOG_LEVELS.get(cfg, 20)
    ) <= int(_LOG_LEVELS.get(str(level).lower(), 20))


def _fmt_log_val(v: Any) -> str:
    """Format a value for log output, falling back gracefully on repr failures."""
    try:
        s = repr(v)
    except Exception:
        # Expected: user objects can raise from __repr__.
        s = f"<unrepr:{type(v).__name__}>"
    return s


def _prefetch_log(level: str, msg: str, /, **fields: Any) -> None:
    """Emit a structured prefetch log line to stderr if *level* is enabled."""
    if not _log_enabled(level):
        return
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    base = f"[AlpamayoPrefetch][{str(level).lower()}] {ts} pid={os.getpid()} {msg}"
    if fields:
        extras = " ".join(f"{k}={_fmt_log_val(v)}" for k, v in fields.items())
        base = base + " " + extras
    print(base, file=sys.stderr, flush=True)


# -----------------------------------------------------------------------------
# Deterministic (seed,n) -> idx mapping (stateless)
# -----------------------------------------------------------------------------
def _alpamayo_rand_idx(n: int, *, dataset_size: int) -> int:
    """Deterministically map logical index *n* to a dataset index via BLAKE2b hashing."""
    if int(dataset_size) <= 0:
        raise ValueError("dataset_size must be > 0")
    if not bool(_alpamayo_cfg_get("idx_rng_enable", False)):
        return int(n)
    seed_any = _alpamayo_cfg_get("idx_rng_seed", None)
    if seed_any is None:
        raise ValueError("idx_rng_enable=true but idx_rng_seed is not set")
    seed = int(seed_any)

    h = hashlib.blake2b(digest_size=8)
    h.update(seed.to_bytes(8, "little", signed=True))
    h.update(int(n).to_bytes(8, "little", signed=True))
    u64 = int.from_bytes(h.digest(), "little", signed=False)
    return int(u64 % int(dataset_size))


def _alpamayo_map_idx(n: int, *, dataset_size: int) -> int:
    """Map logical index *n* using the configured strategy (identity or random hash)."""
    if bool(_alpamayo_cfg_get("idx_mapper_identity", False)):
        return int(n)
    return _alpamayo_rand_idx(int(n), dataset_size=int(dataset_size))


def _copy_for_role(x: Any) -> Any:
    """Deep-copy containers but keep torch tensors (avoid expensive clones)."""
    if isinstance(x, torch.Tensor):
        return x
    if isinstance(x, dict):
        return {k: _copy_for_role(v) for k, v in x.items()}
    if isinstance(x, list):
        return [_copy_for_role(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_copy_for_role(v) for v in x)
    return copy.deepcopy(x)


def _materialize_local_for_role(*, dataset: Any, mapped_idx: int, role: str) -> Any:
    """Local (in-process) materialization path used when the node server returns a miss.

    This mirrors the server's role semantics but does not use shared memory.
    """
    raw = dataset.__getitem__(int(mapped_idx))
    if role == "raw":
        return raw
    from rl.models.reasoning_vla.data_packer import RVLADataPacker

    packer = RVLADataPacker()
    if role == "policy":
        pol = _copy_for_role(raw)
        if not isinstance(pol, dict):
            raise TypeError(f"Expected dict sample for policy, got {type(pol)}")
        return packer._prepare_policy_sample(pol)  # type: ignore[attr-defined]
    if role == "rollout":
        ro_raw = _copy_for_role(raw)
        if not isinstance(ro_raw, dict):
            raise TypeError(f"Expected dict sample for rollout, got {type(ro_raw)}")
        prompt = packer._sample_to_rollout_prompt(ro_raw)  # type: ignore[attr-defined]
        if not isinstance(prompt, dict):
            raise TypeError(f"Expected rollout prompt dict, got {type(prompt)}")
        for k in ROLLOUT_KEEP_KEYS:
            if k in ro_raw:
                prompt[k] = ro_raw[k]
        return prompt
    raise ValueError(f"Unknown role: {role!r}")


# -----------------------------------------------------------------------------
# Node server protocol (Unix socket + shared memory segments)
# -----------------------------------------------------------------------------


def _socket_path(*, server_key: str) -> str:
    """Build the Unix-domain socket path for this node's prefetch server."""
    sock_dir = str(_alpamayo_cfg_get("prefetch.socket_dir", "/tmp"))
    os.makedirs(sock_dir, exist_ok=True)
    host = os.environ.get("HOSTNAME", "host")
    job = os.environ.get("SLURM_JOB_ID") or os.environ.get("JOB_ID") or "job"
    safe_key = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(server_key))
    return os.path.join(sock_dir, f"alpamayo_prefetch_{job}_{host}_{safe_key}.sock")


def _send(sock: socket.socket, obj: Any) -> None:
    """Pickle *obj* and send it over *sock* with a length-prefix header."""
    b = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    sock.sendall(struct.pack("!Q", len(b)))
    sock.sendall(b)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    """Read exactly *n* bytes from *sock*, raising EOFError on premature close."""
    out = bytearray()
    while len(out) < n:
        chunk = sock.recv(n - len(out))
        if not chunk:
            raise EOFError("socket closed")
        out.extend(chunk)
    return bytes(out)


def _recv(sock: socket.socket) -> Any:
    """Receive a length-prefixed pickled object from *sock*."""
    (ln,) = struct.unpack("!Q", _recv_exact(sock, 8))
    return pickle.loads(_recv_exact(sock, int(ln)))


@dataclass
class _Entry:
    """A cached prefetch entry holding shared-memory references per role."""

    n: int
    mapped_idx: int
    # role -> (meta_shm_name, meta_size_bytes, extra_shm_names)
    # - meta shm contains pickled lightweight object graph (dict/list/scalars + shm refs)
    # - large tensors/arrays/bytes are stored in extra shm blocks "as is"
    by_role: dict[str, tuple[str, int, tuple[str, ...]]]


@dataclass
class _WorkerResult:
    """Result message returned by a background prefetch worker process."""

    ok: bool
    n: int
    mapped_idx: int
    by_role: dict[str, tuple[str, int, tuple[str, ...]]] | None = None
    error: str | None = None


def _server_main(*, dataset: Any, socket_path: str, server_key: str) -> None:
    """Run the node-level prefetch server loop (one per node per split)."""
    from rl.models.reasoning_vla.data_packer import RVLADataPacker

    ds_len = len(dataset)
    if ds_len <= 0:
        raise ValueError("dataset is empty")

    capacity = int(_alpamayo_cfg_get("prefetch.capacity", 8) or 0)
    if capacity <= 0:
        raise ValueError("prefetch.capacity must be > 0 to run server")
    num_workers = int(_alpamayo_cfg_get("prefetch.num_workers", 1) or 1)

    _prefetch_log("info", "server_starting", socket=socket_path, server_key=str(server_key))

    log_request_details = bool(_alpamayo_cfg_get("prefetch.log_request_details", True))
    packer = RVLADataPacker()

    cache: OrderedDict[int, _Entry] = OrderedDict()
    # Background prefetch workers: keep "next-per-mod" warm without blocking hit-serving.
    cache_lock = threading.Lock()
    inflight: set[int] = set()
    # Sharded (mod/base) dispatch + lookahead are ROLE-SPECIFIC, even though the server is shared.
    # The role is provided per-request (op=get, role=...).
    base_by_role = {
        "policy": int(_PREFETCH_DISPATCH_BASE_POLICY),
        "rollout": int(_PREFETCH_DISPATCH_BASE_ROLLOUT),
        "raw": int(_PREFETCH_DISPATCH_BASE_ROLLOUT),
    }
    lookahead_by_role = {
        "policy": int(_PREFETCH_LOOKAHEAD_PER_MOD_POLICY),
        "rollout": int(_PREFETCH_LOOKAHEAD_PER_MOD_ROLLOUT),
        "raw": int(_PREFETCH_LOOKAHEAD_PER_MOD_ROLLOUT),
    }
    if base_by_role["policy"] <= 0 or base_by_role["rollout"] <= 0:
        raise RuntimeError(
            "Prefetch enabled but dispatch bases are not configured. Expected TOML keys "
            "policy.parallelism.n_init_replicas and rollout.parallelism.n_init_replicas."
        )
    for k, v in list(lookahead_by_role.items()):
        lookahead_by_role[k] = int(max(int(v), 1))

    # Track active residues and shard-local k per role.
    active_mods_by_role: dict[str, set[int]] = {"policy": set(), "rollout": set(), "raw": set()}
    last_k_by_role_mod: dict[str, dict[int, int]] = {"policy": {}, "rollout": {}, "raw": {}}

    mp = get_context("fork")
    task_qs = [mp.Queue(maxsize=2048) for _ in range(int(num_workers))]
    result_q = mp.Queue(maxsize=4096)
    worker_ps: list[Any] = []

    _prefetch_log(
        "info",
        "server_init",
        socket=socket_path,
        dataset_len=int(ds_len),
        capacity=int(capacity),
        num_workers=int(num_workers),
        idx_rng_enable=bool(_alpamayo_cfg_get("idx_rng_enable", False)),
        idx_mapper_identity=bool(_alpamayo_cfg_get("idx_mapper_identity", False)),
        cosmos_role=os.environ.get("COSMOS_ROLE", ""),
    )

    def mapped_idx_for(n: int) -> int:
        return int(_alpamayo_map_idx(int(n), dataset_size=ds_len) % ds_len)

    def _schedule_unlink_entry(ent: _Entry) -> None:
        # Unlink meta + all extra shm blocks for all roles in this cache entry.
        for _role, (nm, _sz, extra) in ent.by_role.items():
            _shm_unlink_quiet(str(nm))
            for x in extra:
                _shm_unlink_quiet(str(x))

    def _materialize_one(n: int) -> _Entry:
        """Materialize a single n into cache if missing (raw+policy+rollout)."""
        key = int(n)

        mapped = mapped_idx_for(key)
        if log_request_details:
            _prefetch_log(
                "debug", "server_cache_miss", n=key, mapped_idx=int(mapped), cache_size=len(cache)
            )
        t_fetch0 = time.monotonic()
        raw = dataset.__getitem__(mapped)
        if log_request_details:
            _prefetch_log(
                "debug",
                "server_dataset_fetched",
                n=key,
                mapped_idx=int(mapped),
                dt_ms=(time.monotonic() - t_fetch0) * 1000.0,
            )
        if _contains_cuda_tensor(raw):
            raise RuntimeError("Server fetched CUDA tensors; not supported for shm")

        by_role: dict[str, tuple[str, int, tuple[str, ...]]] = {}

        def _put_role(role: str, obj: Any) -> None:
            t0 = time.monotonic()
            by_role[role] = shm_put(obj)
            if log_request_details:
                _prefetch_log(
                    "debug",
                    "server_put_role",
                    n=key,
                    role=str(role),
                    dt_ms=(time.monotonic() - t0) * 1000.0,
                    shm_name=by_role[role][0],
                    size=int(by_role[role][1]),
                )

        # raw
        _put_role("raw", raw)

        # policy
        pol = _copy_for_role(raw)
        if not isinstance(pol, dict):
            raise TypeError(f"Expected dict sample for policy, got {type(pol)}")
        pol2 = packer._prepare_policy_sample(pol)  # type: ignore[attr-defined]
        _put_role("policy", pol2)

        # rollout
        ro_raw = _copy_for_role(raw)
        if not isinstance(ro_raw, dict):
            raise TypeError(f"Expected dict sample for rollout, got {type(ro_raw)}")
        prompt = packer._sample_to_rollout_prompt(ro_raw)  # type: ignore[attr-defined]
        if not isinstance(prompt, dict):
            raise TypeError(f"Expected rollout prompt dict, got {type(prompt)}")
        for k2 in ROLLOUT_KEEP_KEYS:
            if k2 in ro_raw:
                prompt[k2] = ro_raw[k2]
        _put_role("rollout", prompt)

        return _Entry(n=key, mapped_idx=int(mapped), by_role=by_role)

    def _effective_capacity() -> int:
        # Keep behavior compatible with old code: cap cannot exceed ds_len.
        return int(min(int(capacity), int(ds_len)))

    def _shard_len(*, mod: int, base: int) -> int:
        # Number of n in [0..ds_len-1] with n % base == mod.
        m = int(mod)
        b = int(base)
        if b <= 0:
            return int(ds_len)
        if m < 0 or m >= int(b):
            return 0
        if m >= int(ds_len):
            return 0
        return int(((int(ds_len) - 1 - m) // int(b)) + 1)

    def _next_ns_for_mod(*, mod: int, base: int, last_k: int, count: int) -> list[int]:
        m = int(mod)
        b = int(base)
        L = int(_shard_len(mod=m, base=b))
        if b <= 0 or L <= 0:
            return []
        # No point prefetching more than the shard length; also avoids wrap-around duplicates.
        count2 = int(min(int(max(1, count)), int(L)))
        k0 = int(last_k)
        out: list[int] = []
        for i in range(1, int(count2) + 1):
            k1 = (k0 + i) % int(L)
            n2 = int(m) + int(k1) * int(b)
            if 0 <= int(n2) < int(ds_len):
                out.append(int(n2))
        return out

    def _evict_to_capacity() -> None:
        # Evict LRU entries when cache exceeds capacity.
        cap_eff = int(_effective_capacity())
        if cap_eff <= 0:
            return
        while len(cache) > cap_eff:
            _k, ent = cache.popitem(last=False)
            _prefetch_log(
                "debug",
                "server_cache_evict_lru",
                n=int(ent.n),
                mapped_idx=int(ent.mapped_idx),
                cache_size=len(cache),
                capacity=int(cap_eff),
            )
            _schedule_unlink_entry(ent)

    def _enqueue_prefetch_ns(ns: set[int]) -> None:
        if not ns:
            return
        to_submit: list[int] = []
        with cache_lock:
            for n2 in sorted(int(x) for x in ns):
                ent = cache.get(int(n2))
                complete = bool(
                    ent is not None and all(r in ent.by_role for r in ("raw", "policy", "rollout"))
                )
                if complete:
                    continue
                if int(n2) in inflight:
                    continue
                inflight.add(int(n2))
                to_submit.append(int(n2))

        enq_ok = 0
        failed: list[int] = []
        for n2 in to_submit:
            try:
                # Distribute evenly even when n follows arithmetic progressions
                # (e.g. sharded n=mod+k*base).
                wid = (int(n2) * 2654435761) % int(num_workers)
                task_qs[int(wid)].put_nowait(int(n2))
                enq_ok += 1
            except queue.Full:
                failed.append(int(n2))

        if failed:
            with cache_lock:
                for n2 in failed:
                    inflight.discard(int(n2))

        if log_request_details and enq_ok:
            _prefetch_log(
                "debug",
                "server_prefetch_enqueued",
                enqueued=int(enq_ok),
            )

    def ensure_present_and_prefetch_next(
        n: int, *, role_key: str
    ) -> tuple[_Entry | None, bool, int]:
        """Serve req_n fast if cached.

        Prefetch ONLY the next sample per active residue class.

        Simplified behavior:
        - Track residue class mod = n % dispatch_mod_base_* for requests seen by this server.
        - For each active mod, prefetch the next `lookahead_per_mod` n in that residue (with wrap).
        - Also enqueue the current req_n if missing, so other replicas can hit it later.
        - Cache is capped by LRU eviction (no sliding window).
        """
        req_n = int(n) % int(ds_len)
        if int(_effective_capacity()) <= 0:
            raise ValueError("capacity must be > 0")

        rk = role_key if role_key in ("policy", "rollout", "raw") else "raw"
        base = int(base_by_role.get(rk, 0) or 0)
        if base <= 0:
            base = int(base_by_role["rollout"])
        mod = int(req_n) % int(base) if base > 0 else 0
        with cache_lock:
            active_mods_by_role[rk].add(int(mod))
            if base > 0:
                # shard-local k where n = mod + k*base
                last_k_by_role_mod[rk][int(mod)] = int((int(req_n) - int(mod)) // int(base))
            _evict_to_capacity()
            pre_ent = cache.get(int(req_n))
            if pre_ent is not None:
                # LRU: mark as recently used.
                cache.move_to_end(int(req_n))
            was_hit = bool(
                pre_ent is not None
                and all(r in pre_ent.by_role for r in ("raw", "policy", "rollout"))
            )

        mapped = mapped_idx_for(req_n)
        ent: _Entry | None = pre_ent if was_hit else None  # type: ignore[assignment]

        # Prefetch: current req_n (if missing) + next per-role lookahead for each active residue.
        desired: set[int] = {int(req_n)}
        for rk2, mods in list(active_mods_by_role.items()):
            b2 = int(base_by_role.get(rk2, 0) or 0)
            if b2 <= 0:
                continue
            la2 = int(lookahead_by_role.get(rk2, 1) or 1)
            lk_map = last_k_by_role_mod.get(rk2, {})
            for m in list(mods):
                last_k = int(lk_map.get(int(m), -1))
                for nn in _next_ns_for_mod(mod=int(m), base=int(b2), last_k=last_k, count=int(la2)):
                    desired.add(int(nn))
        _enqueue_prefetch_ns(desired)

        return ent, bool(was_hit), int(mapped)

    def _worker_loop(*, wid: int) -> None:
        # Let the main process handle Ctrl-C / termination.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        while True:
            n = task_qs[int(wid)].get()
            if n is None:
                return
            try:
                ent = _materialize_one(int(n))
                result_q.put(
                    _WorkerResult(
                        ok=True,
                        n=int(ent.n),
                        mapped_idx=int(ent.mapped_idx),
                        by_role=ent.by_role,
                    )
                )
            except Exception as e:
                result_q.put(
                    _WorkerResult(
                        ok=False,
                        n=int(n),
                        mapped_idx=int(mapped_idx_for(int(n))),
                        by_role=None,
                        error=repr(e),
                    )
                )

    for wid in range(int(num_workers)):
        p = mp.Process(target=_worker_loop, kwargs={"wid": int(wid)}, daemon=True)
        p.start()
        worker_ps.append(p)

    def _check_and_restart_workers() -> None:
        for wid in range(len(worker_ps)):
            p = worker_ps[wid]
            if p.is_alive():
                continue
            exitcode = getattr(p, "exitcode", None)
            _prefetch_log(
                "warn",
                "server_worker_died",
                wid=int(wid),
                pid=getattr(p, "pid", None),
                exitcode=exitcode,
            )
            stale = 0
            old_q = task_qs[int(wid)]
            while True:
                try:
                    item = old_q.get_nowait()
                except queue.Empty:
                    break
                if item is not None:
                    with cache_lock:
                        inflight.discard(int(item))
                    stale += 1
            new_q = mp.Queue(maxsize=2048)
            task_qs[int(wid)] = new_q
            new_p = mp.Process(target=_worker_loop, kwargs={"wid": int(wid)}, daemon=True)
            new_p.start()
            worker_ps[wid] = new_p
            _prefetch_log(
                "info",
                "server_worker_restarted",
                wid=int(wid),
                new_pid=getattr(new_p, "pid", None),
                stale_cleared=int(stale),
            )

    def _drain_worker_results(max_items: int = 256) -> None:
        got = 0
        while got < int(max_items):
            try:
                res = result_q.get_nowait()
            except queue.Empty:
                break
            got += 1
            if not isinstance(res, _WorkerResult):
                continue
            with cache_lock:
                inflight.discard(int(res.n))
                if res.ok and res.by_role is not None:
                    cache[int(res.n)] = _Entry(
                        n=int(res.n), mapped_idx=int(res.mapped_idx), by_role=res.by_role
                    )
                    cache.move_to_end(int(res.n))
                    _evict_to_capacity()
                else:
                    _prefetch_log(
                        "error", "server_prefetch_worker_error", n=int(res.n), err=str(res.error)
                    )

    # Ensure socket file is not clobbered if a live server exists.
    if os.path.exists(socket_path):
        probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        probe.settimeout(0.1)
        try:
            probe.connect(socket_path)
            _prefetch_log("info", "server_already_running", socket=socket_path)
            return
        except OSError:
            _prefetch_log("warn", "server_socket_stale_unlink", socket=socket_path)
            os.unlink(socket_path)
        finally:
            probe.close()

    srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    srv.bind(socket_path)
    srv.listen(128)
    srv.settimeout(0.5)
    _prefetch_log("info", "server_listening", socket=socket_path)

    try:
        while True:
            _drain_worker_results()
            _check_and_restart_workers()

            try:
                conn, _ = srv.accept()
            except TimeoutError:
                continue

            with conn:
                try:
                    req = _recv(conn)
                except EOFError:
                    continue

                if not isinstance(req, dict) or "op" not in req:
                    _send(conn, {"ok": False, "error": "bad_request"})
                    continue

                op = str(req["op"])
                if op == "shutdown":
                    _prefetch_log("info", "server_shutdown_req")
                    _send(conn, {"ok": True})
                    return
                if op == "clear_cache":
                    _prefetch_log(
                        "info",
                        "server_clear_cache_req",
                        cache_size=len(cache),
                    )
                    # IMPORTANT: also retire shm segments; otherwise we leak /dev/shm.
                    for ent in list(cache.values()):
                        _schedule_unlink_entry(ent)
                    cache.clear()
                    _send(conn, {"ok": True})
                    continue
                if op != "get":
                    _send(conn, {"ok": False, "error": f"unknown_op:{op}"})
                    continue

                n = int(req.get("n", 0))
                role = str(req.get("role", "raw") or "raw")
                role_key = role if role in ("policy", "rollout", "raw") else "raw"
                t0 = time.monotonic()
                ent, was_hit, mapped_idx = ensure_present_and_prefetch_next(
                    n, role_key=str(role_key)
                )
                if not was_hit:
                    dt_ms = (time.monotonic() - t0) * 1000.0
                    if log_request_details:
                        _prefetch_log(
                            "debug",
                            "server_get_miss",
                            n=int(n) % int(ds_len),
                            mapped_idx=int(mapped_idx),
                            role=role,
                            hit=False,
                            dt_ms=float(dt_ms),
                            cache_size=len(cache),
                        )
                    _send(
                        conn,
                        {
                            "ok": True,
                            "n": int(n) % int(ds_len),
                            "mapped_idx": int(mapped_idx),
                            "role": role,
                            "hit": False,
                            "miss": True,
                        },
                    )
                    continue
                assert ent is not None
                if role not in ent.by_role:
                    _send(conn, {"ok": False, "error": f"role_missing:{role}"})
                    continue

                dt_ms = (time.monotonic() - t0) * 1000.0
                if log_request_details:
                    _prefetch_log(
                        "debug",
                        "server_get_ok",
                        n=int(ent.n),
                        mapped_idx=int(ent.mapped_idx),
                        role=role,
                        hit=bool(was_hit),
                        dt_ms=float(dt_ms),
                        cache_size=len(cache),
                    )

                nm, sz, _extra = ent.by_role[role]
                try:
                    _send(
                        conn,
                        {
                            "ok": True,
                            "n": int(ent.n),
                            "mapped_idx": int(ent.mapped_idx),
                            "role": role,
                            "shm_name": nm,
                            "size": int(sz),
                            "hit": bool(was_hit),
                        },
                    )
                except (BrokenPipeError, ConnectionResetError):
                    continue
                # (no-op) keep serving requests
    finally:
        # Stop worker processes.
        for q in task_qs:
            try:
                q.put_nowait(None)
            except queue.Full:
                # Expected: bounded queue can be full; block to deliver shutdown sentinel.
                q.put(None)
        for p in worker_ps:
            p.join(timeout=1.0)
        for p in worker_ps:
            if p.is_alive():
                p.terminate()
        # Cleanup shm blocks on server exit.
        for ent in list(cache.values()):
            for _role, (nm, _sz, extra) in ent.by_role.items():
                for x in (nm, *tuple(extra)):
                    _shm_unlink_quiet(str(x))
        srv.close()
        if os.path.exists(socket_path):
            try:
                os.unlink(socket_path)
            except FileNotFoundError:
                # Expected: another process may have cleaned up concurrently.
                pass


class _NodeClient:
    """Client that talks to the per-node prefetch server over a Unix socket."""

    def __init__(self, *, dataset: Any, server_key: str):
        self._dataset = dataset
        self._server_key = str(server_key)
        self._socket_path = _socket_path(server_key=self._server_key)
        self._ctx = get_context("fork")
        self._owner_pid = int(os.getpid())
        self._proc = None
        self._cleanup_stale_sockets()

    def _cleanup_stale_sockets(self) -> None:
        """Remove stale socket/lock/shm files from previous runs on this node.

        After a slurm requeue/resume, old socket files and /dev/shm semaphores
        from the previous run may linger. If the prefetch server that owned
        them is dead, they prevent the new server from starting cleanly and
        can exhaust /dev/shm on repeated retries.
        """
        sock_path = self._socket_path
        lock_path = sock_path + ".lock"
        for path in (sock_path, lock_path):
            if not os.path.exists(path):
                continue
            probe = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            probe.settimeout(0.1)
            try:
                probe.connect(sock_path)
                probe.close()
                return  # server is alive, don't clean up
            except OSError:
                probe.close()
            try:
                os.unlink(path)
                _prefetch_log("info", "client_cleaned_stale_socket", path=path)
            except FileNotFoundError:
                pass

    def _connect(self, timeout_s: float) -> socket.socket | None:
        """Open a Unix socket connection to the server, returning None on failure."""
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(timeout_s)
        try:
            s.connect(self._socket_path)
            s.settimeout(None)
            return s
        except OSError:
            s.close()
            return None

    def _ensure_server(self) -> None:
        """Start the server process if it is not already running."""
        _prefetch_log(
            "debug",
            "client_ensure_server_begin",
            socket=self._socket_path,
            server_key=self._server_key,
        )
        if self._connect(0.05) is not None:
            _prefetch_log(
                "debug",
                "client_server_already_up",
                socket=self._socket_path,
                server_key=self._server_key,
            )
            return

        lock_path = self._socket_path + ".lock"
        fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o644)
        try:
            _prefetch_log("debug", "client_lock_acquire", lock_path=lock_path)
            fcntl.flock(fd, fcntl.LOCK_EX)
            _prefetch_log("debug", "client_lock_acquired", lock_path=lock_path)

            # If someone else created the socket, wait a bit before spawning a competitor.
            if os.path.exists(self._socket_path):
                _prefetch_log("debug", "client_socket_exists_wait", socket=self._socket_path)
                deadline = time.monotonic() + 2.0
                while time.monotonic() < deadline:
                    s = self._connect(0.2)
                    if s is not None:
                        s.close()
                        _prefetch_log("debug", "client_socket_connect_ok", socket=self._socket_path)
                        return
                    time.sleep(0.05)

            _prefetch_log(
                "info",
                "client_spawn_server",
                socket=self._socket_path,
                server_key=self._server_key,
            )
            p = self._ctx.Process(
                target=_server_main,
                name="alpamayo-node-prefetch-server",
                kwargs={
                    "dataset": self._dataset,
                    "socket_path": self._socket_path,
                    "server_key": self._server_key,
                },
                daemon=False,
            )
            p.start()
            self._proc = p
            _prefetch_log(
                "info",
                "client_spawned_server",
                server_pid=getattr(p, "pid", None),
                socket=self._socket_path,
            )

            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                s = self._connect(0.2)
                if s is not None:
                    s.close()
                    _prefetch_log("info", "client_server_ready", socket=self._socket_path)
                    return
                time.sleep(0.05)
            _prefetch_log("error", "client_server_start_timeout", socket=self._socket_path)
            raise RuntimeError("Timed out waiting for node prefetch server")
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)
            _prefetch_log("debug", "client_lock_released", lock_path=lock_path)

    def _rpc(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send *payload* to the server and return the response dict."""
        _prefetch_log("trace", "client_rpc", socket=self._socket_path, payload=payload)
        self._ensure_server()
        # Fixed recv timeout to avoid indefinite hangs; no TOML knob.
        recv_timeout_s: float | None = 30.0

        def _do_once() -> dict[str, Any] | None:
            t0 = time.monotonic()
            s = self._connect(1.0)
            if s is None:
                dt_ms = (time.monotonic() - t0) * 1000.0
                _prefetch_log(
                    "debug",
                    "client_rpc_connect_failed",
                    socket=self._socket_path,
                    dt_ms=dt_ms,
                )
                return None
            try:
                with s:
                    # IMPORTANT: _recv() blocks until it reads the full payload size. If the server
                    # accepts but fails to respond, this can hang forever unless a recv timeout
                    # is set.
                    if recv_timeout_s is not None:
                        s.settimeout(float(recv_timeout_s))
                    dt_ms = (time.monotonic() - t0) * 1000.0
                    _prefetch_log(
                        "trace",
                        "client_rpc_connected",
                        socket=self._socket_path,
                        dt_ms=dt_ms,
                        recv_timeout_s=recv_timeout_s,
                    )
                    _send(s, payload)
                    _prefetch_log("trace", "client_rpc_sent", socket=self._socket_path)
                    return _recv(s)
            except (BrokenPipeError, ConnectionResetError, EOFError):
                return None
            except TimeoutError:
                _prefetch_log(
                    "warn",
                    "client_rpc_timeout",
                    socket=self._socket_path,
                    recv_timeout_s=recv_timeout_s,
                    payload=payload if _log_enabled("trace") else {"op": payload.get("op")},
                )
                return None
            except OSError as e:
                # socket.timeout is an OSError subclass on some Python versions.
                if isinstance(e, TimeoutError):
                    _prefetch_log(
                        "warn",
                        "client_rpc_timeout",
                        socket=self._socket_path,
                        recv_timeout_s=recv_timeout_s,
                        payload=payload if _log_enabled("trace") else {"op": payload.get("op")},
                    )
                    return None
                raise

        resp = _do_once()
        if resp is None:
            # Single retry: tolerate races with server restart/idle-shutdown during tests.
            _prefetch_log("warn", "client_rpc_retry", socket=self._socket_path)
            self._ensure_server()
            resp = _do_once()
        if resp is None:
            _prefetch_log("error", "client_rpc_failed", socket=self._socket_path)
            raise RuntimeError("Failed to RPC to node prefetch server")
        _prefetch_log("trace", "client_rpc_resp", socket=self._socket_path, resp=resp)
        if not isinstance(resp, dict):
            raise RuntimeError(f"Bad response: {type(resp)}")
        return resp

    def get(self, *, n: int, role: str) -> tuple[Any | None, int, bool]:
        """Fetch sample *n* for *role* via the server, returning (sample, mapped_idx, was_hit)."""
        _prefetch_log(
            "debug", "client_get_begin", n=int(n), role=str(role), server_key=self._server_key
        )
        resp = self._rpc(
            {
                "op": "get",
                "n": int(n),
                "role": str(role),
            }
        )
        if not resp.get("ok", False):
            _prefetch_log("error", "client_get_server_error", resp=resp)
            raise RuntimeError(f"Server error: {resp!r}")

        # Miss path: server did not return shm payload; caller should materialize locally.
        if bool(resp.get("miss", False)) or "shm_name" not in resp:
            mapped_idx = int(resp["mapped_idx"])
            _prefetch_log(
                "debug",
                "client_get_miss",
                n=int(resp.get("n", n)),
                role=str(resp.get("role", role)),
                mapped_idx=int(mapped_idx),
            )
            return None, mapped_idx, False

        shm_name = str(resp["shm_name"])
        size = int(resp["size"])
        mapped_idx = int(resp["mapped_idx"])
        was_hit = resp.get("hit", None)
        _prefetch_log(
            "debug",
            "client_get_shm",
            n=int(resp.get("n", n)),
            role=str(resp.get("role", role)),
            mapped_idx=int(mapped_idx),
            hit=was_hit,
            shm_name=shm_name,
            size=int(size),
        )

        # Attach with small retry to tolerate delayed-unlink races.
        last: Exception | None = None
        for attempt in range(50):
            try:
                shm = SharedMemory(name=shm_name, create=False)
                try:
                    resource_tracker.unregister(shm._name, "shared_memory")  # type: ignore[attr-defined]
                except (KeyError, ValueError):
                    pass
                try:
                    # Copy out of shm before closing to avoid exported-pointer BufferError.
                    b = bytes(shm.buf[:size])
                finally:
                    shm.close()
                _prefetch_log(
                    "debug",
                    "client_get_ok",
                    n=int(n),
                    role=str(role),
                    mapped_idx=int(mapped_idx),
                    hit=was_hit,
                )
                meta = pickle.loads(b)
                return (
                    _shm_unpack_client(meta),
                    mapped_idx,
                    bool(was_hit) if was_hit is not None else True,
                )
            except FileNotFoundError as e:
                last = e
                if attempt in (0, 9, 24, 49):
                    _prefetch_log(
                        "warn",
                        "client_shm_attach_retry",
                        shm_name=shm_name,
                        attempt=int(attempt),
                        n=int(n),
                        role=str(role),
                    )
                time.sleep(0.01)
        _prefetch_log(
            "error", "client_shm_attach_failed", shm_name=shm_name, n=int(n), role=str(role)
        )
        raise RuntimeError(f"shared_memory disappeared: {shm_name!r}") from last

    def shutdown_best_effort(self) -> None:
        """Ask the server to shut down, ignoring communication errors."""
        self._rpc_best_effort({"op": "shutdown"})

    def clear_cache_best_effort(self) -> None:
        """Ask the server to drop its cache, ignoring communication errors."""
        self._rpc_best_effort({"op": "clear_cache"})

    def _rpc_best_effort(self, payload: dict[str, Any]) -> None:
        """Fire-and-forget RPC wrapper that silently swallows connection errors."""
        try:
            self._rpc(payload)
        except (
            OSError,
            RuntimeError,
            EOFError,
            BrokenPipeError,
            ConnectionResetError,
            TimeoutError,
        ):
            return


_NODE_CLIENTS: dict[str, _NodeClient] = {}
# dataset object id -> server_key (typically split: "train"/"val")
_DATASET_SERVER_KEY: dict[int, str] = {}

# shm read/zero-copy unpack helpers live in `rl.utils.prefetch_shm`.
# Client-side shm handle registry for true 0-copy views.
#
# Requirements:
# - If we return a torch/numpy object that views into `SharedMemory.buf`, we MUST keep the
#   shm mapping alive until the returned object is garbage-collected. Closing early can
#   invalidate the buffer and/or raise BufferError.
#
# Implementation:
# - Keep a (shm_name -> (SharedMemory, refcount)) map.
# - Each returned tensor/ndarray registers a `weakref.finalize(...)` callback that decrements
#   the refcount; when it hits 0 we close the shm handle.
#
# NOTE: the client shm handle registry + unpacking is implemented in
# `rl.utils.prefetch_shm` and imported at the top of this file.


def _get_client(*, dataset: Any, server_key: str) -> _NodeClient:
    """Return (or create) the per-process ``_NodeClient`` for *server_key*."""
    # Recreate after fork (avoid inherited state).
    cur = int(os.getpid())
    c = _NODE_CLIENTS.get(server_key)
    if c is not None and getattr(c, "_owner_pid", cur) != cur:
        _NODE_CLIENTS.pop(server_key, None)
        c = None
    if c is None:
        c = _NodeClient(dataset=dataset, server_key=server_key)
        _NODE_CLIENTS[server_key] = c
    return c


def init_dataset_prefetch(
    *, dataset: Any, tag: str | None = None, server_key: str | None = None
) -> None:
    """Initialize/attach node server for this dataset (tag is ignored; kept for compat)."""
    cap = int(_alpamayo_cfg_get("prefetch.capacity", 8) or 0)
    if cap <= 0:
        return
    if _is_controller_role():
        return
    key = str(server_key or "default")
    _prefetch_log(
        "info",
        "init_dataset_prefetch",
        dataset_id=int(id(dataset)),
        server_key=key,
        tag=tag,
        capacity=int(cap),
    )
    _DATASET_SERVER_KEY[int(id(dataset))] = key
    _get_client(dataset=dataset, server_key=key)._ensure_server()


def get_process_prefetch_queue() -> dict[str, _NodeClient]:
    """Debug helper: return current node clients (per server_key)."""
    return dict(_NODE_CLIENTS)


def shutdown_all_prefetch_queues() -> None:
    """Shutdown servers best-effort for this process."""
    global _NODE_CLIENTS
    for c in list(_NODE_CLIENTS.values()):
        c.shutdown_best_effort()
    _NODE_CLIENTS = {}
    # Also release any client-side shm handles kept alive for zero-copy views.
    _client_shm_close_all()


def clear_all_prefetch_caches() -> None:
    """Ask servers to clear their in-memory caches (best-effort)."""
    for c in list(_NODE_CLIENTS.values()):
        c.clear_cache_best_effort()


atexit.register(shutdown_all_prefetch_queues)


def _fetch_sample(
    *, n: int, dataset: Any, role: str | None = None, **_ignored: Any
) -> tuple[Any, int]:
    """Fetch sample for logical n.

    Returns (sample, mapped_idx).
    """
    ds_len = len(dataset)
    if ds_len <= 0:
        raise ValueError("dataset is empty")
    mapped_idx = int(_alpamayo_map_idx(int(n), dataset_size=ds_len) % ds_len)

    cap = int(_alpamayo_cfg_get("prefetch.capacity", 8) or 0)
    if cap <= 0:
        # Prefetch disabled: synchronous fetch (raw only).
        sample = dataset.__getitem__(mapped_idx)
        return sample, mapped_idx

    # Prefer the server_key bound to this dataset instance via
    # init_dataset_prefetch(..., server_key=split).
    key = _DATASET_SERVER_KEY.get(int(id(dataset)), "default")
    client = _get_client(dataset=dataset, server_key=key)
    sample, mapped2, was_hit = client.get(n=int(n), role=str(role or "raw"))
    if sample is None:
        # Miss: materialize locally in the caller process.
        role2 = str(role or "raw")
        return _materialize_local_for_role(
            dataset=dataset, mapped_idx=int(mapped2), role=role2
        ), int(mapped2)
    return sample, int(mapped2)
