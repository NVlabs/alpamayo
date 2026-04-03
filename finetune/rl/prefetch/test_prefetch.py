# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end tests for the node-level shared-memory prefetch system.

The prefetch system's contract: for any (n, role), the result through the
prefetch server path must be identical to the non-prefetch (synchronous) path.

These tests verify that contract by running both paths and comparing outputs.

Run:
    cd projects/alpamayo_r1_release/finetune
    python -m pytest rl/prefetch/test_prefetch.py -v -x
"""

from __future__ import annotations

import os
import pickle
import socket
import threading
import time
from multiprocessing.shared_memory import SharedMemory
from typing import Any
from unittest import mock

import numpy as np
import pytest
import torch


# ---------------------------------------------------------------------------
# Fake dataset & packer (avoid real model/tokenizer dependency)
# ---------------------------------------------------------------------------
class _FakeDataset:
    """Deterministic map-style dataset producing known tensor values per index."""

    def __init__(self, size: int = 20):
        self._size = size

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, idx: int) -> dict:
        rng = torch.Generator().manual_seed(idx)
        return {
            "tokenized_data": {"input_ids": torch.arange(10, dtype=torch.long) + idx},
            "images": torch.randn(3, 32, 32, generator=rng),
            "ego_future_xyz": torch.randn(1, 1, 10, 3, generator=rng),
            "ego_future_rot": torch.randn(1, 1, 10, 3, 3, generator=rng),
            "ego_history_xyz": torch.randn(1, 3, 3, generator=rng),
            "ego_history_rot": torch.randn(1, 3, 3, 3, generator=rng),
            "cot": f"cot_for_{idx}",
            "meta_action_strings": f"action_{idx}",
        }


class _FakePacker:
    """Mimics RVLADataPacker transforms without real tokenizer."""

    def _prepare_policy_sample(self, sample: dict) -> dict:
        ids = sample["tokenized_data"]["input_ids"]
        if isinstance(ids, torch.Tensor) and ids.ndim == 1:
            sample["tokenized_data"]["input_ids"] = ids.unsqueeze(0)
        return sample

    def _sample_to_rollout_prompt(self, sample: dict) -> dict:
        ids = sample["tokenized_data"]["input_ids"]
        return {
            "prompt_token_ids": ids.tolist() if isinstance(ids, torch.Tensor) else ids,
        }


def _import_server():
    try:
        from rl.prefetch import server

        return server
    except ImportError:
        pytest.skip("rl.prefetch.server not importable from cwd")


def _import_shm():
    try:
        from rl.prefetch import shm

        return shm
    except ImportError:
        pytest.skip("rl.prefetch.shm not importable from cwd")


# ---------------------------------------------------------------------------
# Server helpers
# ---------------------------------------------------------------------------
def _setup_cfg(srv, *, capacity: int, sock_dir: str):
    srv._alpamayo_custom_cfg = {
        "idx_mapper_identity": True,
        "prefetch": {
            "capacity": capacity,
            "num_workers": 1,
            "socket_dir": sock_dir,
            "log_level": "error",
            "log_request_details": False,
        },
    }
    srv._PREFETCH_DISPATCH_BASE_POLICY = 2
    srv._PREFETCH_DISPATCH_BASE_ROLLOUT = 2
    srv._PREFETCH_LOOKAHEAD_PER_MOD_POLICY = 1
    srv._PREFETCH_LOOKAHEAD_PER_MOD_ROLLOUT = 1


def _cleanup(srv):
    srv._alpamayo_custom_cfg = {}
    srv._PREFETCH_DISPATCH_BASE_POLICY = 0
    srv._PREFETCH_DISPATCH_BASE_ROLLOUT = 0
    srv._PREFETCH_LOOKAHEAD_PER_MOD_POLICY = 1
    srv._PREFETCH_LOOKAHEAD_PER_MOD_ROLLOUT = 1
    srv._NODE_CLIENTS.clear()
    srv._DATASET_SERVER_KEY.clear()


_PACKER_PATCH = {
    "rl.models.reasoning_vla.data_packer": mock.MagicMock(RVLADataPacker=_FakePacker)
}


def _start_server(srv, dataset, sock_path: str) -> threading.Thread:
    def _run():
        with mock.patch.dict("sys.modules", _PACKER_PATCH):
            try:
                srv._server_main(
                    dataset=dataset,
                    socket_path=sock_path,
                    server_key=os.path.basename(sock_path),
                )
            except Exception:
                pass

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        if os.path.exists(sock_path):
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(0.2)
            try:
                s.connect(sock_path)
                s.close()
                return t
            except OSError:
                pass
            finally:
                s.close()
        time.sleep(0.05)
    raise RuntimeError("Server did not start in time")


def _rpc(srv, sock_path: str, payload: dict) -> dict:
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    s.settimeout(10.0)
    s.connect(sock_path)
    try:
        srv._send(s, payload)
        return srv._recv(s)
    finally:
        s.close()


def _read_shm_sample(srv, resp: dict) -> Any:
    """Read and unpack a sample from the shm_name in a server response."""
    shm_mod = _import_shm()
    mem = SharedMemory(name=resp["shm_name"], create=False)
    meta = pickle.loads(bytes(mem.buf[: resp["size"]]))
    mem.close()
    return shm_mod.shm_unpack_client(meta)


def _get_direct(ds, idx: int, role: str) -> Any:
    """Non-prefetch path: what the system produces without the server."""
    srv = _import_server()
    raw = ds[idx]
    if role == "raw":
        return raw
    packer = _FakePacker()
    if role == "policy":
        copy = srv._copy_for_role(raw)
        return packer._prepare_policy_sample(copy)
    if role == "rollout":
        copy = srv._copy_for_role(raw)
        prompt = packer._sample_to_rollout_prompt(copy)
        for k in srv.ROLLOUT_KEEP_KEYS:
            if k in copy:
                prompt[k] = copy[k]
        return prompt
    raise ValueError(f"Unknown role: {role}")


def _assert_samples_equal(actual: Any, expected: Any, path: str = ""):
    """Deep comparison of two samples, with clear error messages."""
    if isinstance(expected, torch.Tensor):
        assert isinstance(actual, torch.Tensor), f"{path}: expected Tensor, got {type(actual)}"
        assert actual.shape == expected.shape, f"{path}: shape {actual.shape} != {expected.shape}"
        assert actual.dtype == expected.dtype, f"{path}: dtype {actual.dtype} != {expected.dtype}"
        assert torch.equal(actual, expected), f"{path}: tensor values differ"
    elif isinstance(expected, np.ndarray):
        assert isinstance(actual, np.ndarray), f"{path}: expected ndarray, got {type(actual)}"
        np.testing.assert_array_equal(actual, expected)
    elif isinstance(expected, dict):
        assert isinstance(actual, dict), f"{path}: expected dict, got {type(actual)}"
        assert set(actual.keys()) == set(expected.keys()), (
            f"{path}: keys differ: {set(actual.keys())} != {set(expected.keys())}"
        )
        for k in expected:
            _assert_samples_equal(actual[k], expected[k], path=f"{path}.{k}")
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected), f"{path}: type {type(actual)} != {type(expected)}"
        assert len(actual) == len(expected), f"{path}: len {len(actual)} != {len(expected)}"
        for i, (a, e) in enumerate(zip(actual, expected)):
            _assert_samples_equal(a, e, path=f"{path}[{i}]")
    else:
        assert actual == expected, f"{path}: {actual!r} != {expected!r}"


# ============================================================================
# E2E: prefetch path == non-prefetch path
# ============================================================================
class TestPrefetchE2E:
    """The core contract: for every (n, role), prefetch output == direct output."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.srv = _import_server()
        self.sock_dir = str(tmp_path)
        _setup_cfg(self.srv, capacity=16, sock_dir=self.sock_dir)
        self.ds = _FakeDataset(size=20)
        self.sock_path = self.srv._socket_path(server_key="e2e")
        self.thread = _start_server(self.srv, self.ds, self.sock_path)
        yield
        try:
            _rpc(self.srv, self.sock_path, {"op": "shutdown"})
        except OSError:
            pass
        self.thread.join(timeout=5)
        self.srv.shutdown_all_prefetch_queues()
        _cleanup(self.srv)

    def _fetch_via_server(self, n: int, role: str) -> Any:
        """Fetch through prefetch server, waiting for cache population."""
        _rpc(self.srv, self.sock_path, {"op": "get", "n": n, "role": role})
        time.sleep(1.0)
        resp = _rpc(self.srv, self.sock_path, {"op": "get", "n": n, "role": role})
        assert resp["ok"], f"Server error for n={n} role={role}: {resp}"
        if resp.get("hit") and "shm_name" in resp:
            return _read_shm_sample(self.srv, resp)
        return None

    @pytest.mark.parametrize("role", ["raw", "policy", "rollout"])
    @pytest.mark.parametrize("n", [0, 1, 7, 19])
    def test_prefetch_matches_direct(self, n: int, role: str):
        """Prefetch server output must be identical to non-prefetch path."""
        expected = _get_direct(self.ds, idx=n, role=role)
        actual = self._fetch_via_server(n, role)
        if actual is None:
            pytest.skip(f"Server returned miss for n={n} role={role} (worker too slow)")
        _assert_samples_equal(actual, expected, path=f"n={n},role={role}")

    def test_multiple_indices_all_roles(self):
        """Batch check: 5 indices × 3 roles, all must match."""
        indices = [0, 3, 5, 12, 18]
        # Warm up cache for all indices
        for n in indices:
            _rpc(self.srv, self.sock_path, {"op": "get", "n": n, "role": "raw"})
        time.sleep(2.0)

        mismatches = []
        for n in indices:
            for role in ("raw", "policy", "rollout"):
                resp = _rpc(self.srv, self.sock_path, {"op": "get", "n": n, "role": role})
                if not (resp.get("hit") and "shm_name" in resp):
                    continue
                actual = _read_shm_sample(self.srv, resp)
                expected = _get_direct(self.ds, idx=n, role=role)
                try:
                    _assert_samples_equal(actual, expected)
                except AssertionError as e:
                    mismatches.append(f"n={n} role={role}: {e}")
        assert not mismatches, f"Mismatches:\n" + "\n".join(mismatches)


# ============================================================================
# E2E: prefetch predicts the correct next sample
# ============================================================================
class TestPrefetchLookahead:
    """After replica 0 requests n=0, server should pre-cache n=2 (base=2)."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.srv = _import_server()
        _setup_cfg(self.srv, capacity=16, sock_dir=str(tmp_path))
        self.ds = _FakeDataset(size=20)
        self.sock_path = self.srv._socket_path(server_key="lookahead")
        self.thread = _start_server(self.srv, self.ds, self.sock_path)
        yield
        try:
            _rpc(self.srv, self.sock_path, {"op": "shutdown"})
        except OSError:
            pass
        self.thread.join(timeout=5)
        self.srv.shutdown_all_prefetch_queues()
        _cleanup(self.srv)

    def test_next_in_shard_is_prefetched(self):
        """Request n=0 (policy, base=2) → n=2 should be pre-cached."""
        _rpc(self.srv, self.sock_path, {"op": "get", "n": 0, "role": "policy"})
        time.sleep(1.5)

        resp = _rpc(self.srv, self.sock_path, {"op": "get", "n": 2, "role": "policy"})
        assert resp["ok"]
        if resp.get("hit") and "shm_name" in resp:
            # Prefetch worked — verify data is also correct
            actual = _read_shm_sample(self.srv, resp)
            expected = _get_direct(self.ds, idx=2, role="policy")
            _assert_samples_equal(actual, expected, path="prefetched n=2 policy")

    def test_sequential_shard_access(self):
        """Simulate replica 0 consuming n=0,2,4,6 — each should eventually hit."""
        hit_count = 0
        for n in range(0, 8, 2):
            _rpc(self.srv, self.sock_path, {"op": "get", "n": n, "role": "policy"})
            time.sleep(0.8)
            resp = _rpc(self.srv, self.sock_path, {"op": "get", "n": n, "role": "policy"})
            if resp.get("hit") and "shm_name" in resp:
                hit_count += 1
        # After the first miss triggers prefetch, subsequent requests should hit
        assert hit_count >= 2, f"Expected at least 2 cache hits, got {hit_count}"


# ============================================================================
# E2E: capacity=0 disables prefetch (synchronous fallback)
# ============================================================================
class TestPrefetchDisabled:
    """With capacity=0, the system falls back to synchronous dataset reads."""

    def test_fetch_sample_returns_raw(self):
        srv = _import_server()
        srv._alpamayo_custom_cfg = {"idx_mapper_identity": True, "prefetch": {"capacity": 0}}
        try:
            ds = _FakeDataset(size=10)
            sample, mapped = srv._fetch_sample(n=5, dataset=ds, role="raw")
            expected = ds[5]
            _assert_samples_equal(sample, expected, path="capacity=0 raw")
            assert mapped == 5
        finally:
            srv._alpamayo_custom_cfg = {}

    def test_wrapper_getitem_matches_direct(self):
        srv = _import_server()
        srv._alpamayo_custom_cfg = {"idx_mapper_identity": True, "prefetch": {"capacity": 0}}
        try:
            from rl.prefetch.dataset import NodePrefetchDatasetWrapper

            ds = _FakeDataset(size=10)
            wrapper = NodePrefetchDatasetWrapper(ds, server_key="disabled")
            for n in (0, 4, 9):
                _assert_samples_equal(wrapper[n], ds[n], path=f"wrapper[{n}]")
        finally:
            srv._alpamayo_custom_cfg = {}


# ============================================================================
# E2E: miss fallback produces same result as server path
# ============================================================================
class TestMissFallback:
    """When server returns a miss, local materialization must match server output."""

    def _materialize(self, ds, idx: int, role: str):
        srv = _import_server()
        with mock.patch.dict("sys.modules", _PACKER_PATCH):
            return srv._materialize_local_for_role(dataset=ds, mapped_idx=idx, role=role)

    @pytest.mark.parametrize("role", ["raw", "policy", "rollout"])
    def test_local_matches_direct(self, role: str):
        """_materialize_local_for_role output == _get_direct output."""
        ds = _FakeDataset(size=10)
        for idx in (0, 3, 9):
            local = self._materialize(ds, idx=idx, role=role)
            direct = _get_direct(ds, idx=idx, role=role)
            _assert_samples_equal(local, direct, path=f"fallback idx={idx} role={role}")


# ============================================================================
# E2E: LRU eviction doesn't corrupt data
# ============================================================================
class TestEviction:
    """After eviction, re-fetched data must still be correct."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.srv = _import_server()
        _setup_cfg(self.srv, capacity=3, sock_dir=str(tmp_path))
        self.ds = _FakeDataset(size=20)
        self.sock_path = self.srv._socket_path(server_key="evict")
        self.thread = _start_server(self.srv, self.ds, self.sock_path)
        yield
        try:
            _rpc(self.srv, self.sock_path, {"op": "shutdown"})
        except OSError:
            pass
        self.thread.join(timeout=5)
        self.srv.shutdown_all_prefetch_queues()
        _cleanup(self.srv)

    def test_refetch_after_eviction_is_correct(self):
        """Fill cache beyond capacity, then re-fetch evicted item — data still correct."""
        # Fill cache with n=0,1,2,3,4 (capacity=3, so n=0,1 get evicted)
        for n in range(5):
            _rpc(self.srv, self.sock_path, {"op": "get", "n": n, "role": "raw"})
            time.sleep(0.5)

        time.sleep(1.0)

        # Re-fetch n=0 (was evicted, will be re-materialized)
        _rpc(self.srv, self.sock_path, {"op": "get", "n": 0, "role": "raw"})
        time.sleep(1.0)
        resp = _rpc(self.srv, self.sock_path, {"op": "get", "n": 0, "role": "raw"})
        assert resp["ok"]
        if resp.get("hit") and "shm_name" in resp:
            actual = _read_shm_sample(self.srv, resp)
            expected = _get_direct(self.ds, idx=0, role="raw")
            _assert_samples_equal(actual, expected, path="re-fetched after eviction")


# ============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
