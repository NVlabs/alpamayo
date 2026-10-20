# SPDX-License-Identifier: Apache-2.0

from unittest import mock

import pytest
import torch

from alpamayo_r1 import device as dev
from alpamayo_r1.action_space.unicycle_accel_curvature import UnicycleAccelCurvatureActionSpace
from alpamayo_r1.models.base_model import ReasoningVLAConfig

HAS_XPU = hasattr(torch, "xpu") and torch.xpu.is_available()


def fake_host(*available: str):
    return mock.patch.object(dev, "is_available", lambda name: name == "cpu" or name in available)


@pytest.mark.parametrize(
    ("available", "expected"),
    [((), "cpu"), (("cuda",), "cuda"), (("xpu",), "xpu"), (("cuda", "xpu"), "xpu")],
)
def test_default_device_type(available, expected) -> None:
    with fake_host(*available):
        assert dev.default_device_type() == expected
        assert dev.resolve_device().type == expected


def test_explicit_device_wins() -> None:
    with fake_host("xpu"):
        assert dev.resolve_device("cuda:1") == torch.device("cuda:1")
        assert dev.resolve_device("cpu") == torch.device("cpu")
        assert dev.resolve_device(None) == torch.device("xpu", 0)


def test_manual_seed_all_dispatch() -> None:
    with fake_host("xpu"), mock.patch.object(torch, "xpu", create=True) as xpu:
        dev.manual_seed_all(7)
        xpu.manual_seed_all.assert_called_once_with(7)
    with fake_host("cuda"), mock.patch.object(torch.cuda, "manual_seed_all") as seed:
        dev.manual_seed_all(7)
        seed.assert_called_once_with(7)


def test_flash_attention_falls_back_off_cuda() -> None:
    with fake_host("xpu"):
        assert dev.resolve_attn_implementation("flash_attention_2") == "sdpa"
        assert dev.resolve_attn_implementation("eager") == "eager"
    with fake_host("cuda"), mock.patch("importlib.util.find_spec", return_value=object()):
        assert dev.resolve_attn_implementation("flash_attention_2") == "flash_attention_2"
    with fake_host("cuda"), mock.patch("importlib.util.find_spec", return_value=None):
        assert dev.resolve_attn_implementation("flash_attention_2") == "sdpa"


def test_config_uses_resolved_attention() -> None:
    with fake_host("xpu"):
        config = ReasoningVLAConfig(vlm_name_or_path=None)
    assert config.attn_implementation == "sdpa"


def test_autocast_disabled_only_touches_present_backends() -> None:
    with fake_host("cuda"), mock.patch.object(torch, "autocast") as autocast:
        with dev.autocast_disabled():
            pass
        assert [call.args for call in autocast.call_args_list] == [("cpu",), ("cuda",)]
    with fake_host():
        with torch.autocast("cpu", dtype=torch.bfloat16), dev.autocast_disabled():
            assert (torch.ones(2, 2) @ torch.ones(2, 2)).dtype == torch.float32


@pytest.mark.skipif(not HAS_XPU, reason="XPU is required")
def test_action_to_traj_cpu_xpu_parity_under_autocast() -> None:
    torch.manual_seed(0)
    space = UnicycleAccelCurvatureActionSpace(accel_std=2.0, curvature_std=0.05)
    action = torch.randn(2, *space.get_action_space_dims())
    t = torch.arange(-15, 1, dtype=torch.float32) * 0.1
    hist_xyz = torch.stack([5.0 * t, 0.2 * t, torch.zeros_like(t)], dim=-1).expand(2, -1, -1)
    hist_rot = torch.eye(3).expand(2, 16, 3, 3)
    assert action.std() > 0.1

    ref_xyz, ref_rot = space.action_to_traj(action, hist_xyz, hist_rot)
    xpu = torch.device("xpu", 0)
    with torch.autocast("xpu", dtype=torch.bfloat16):
        xyz, rot = space.to(xpu).action_to_traj(action.to(xpu), hist_xyz.to(xpu), hist_rot.to(xpu))
    assert xyz.dtype == torch.float32
    torch.testing.assert_close(xyz.cpu(), ref_xyz, atol=1e-2, rtol=1e-3)
    torch.testing.assert_close(rot.cpu(), ref_rot, atol=1e-3, rtol=1e-3)
