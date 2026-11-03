from types import SimpleNamespace

import pytest

from fullwave.ultra import binary


@pytest.fixture
def launch_envs(monkeypatch):
    envs = []

    def fake_subprocess_run(_command, *, env, **_kwargs):
        envs.append(env)
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(binary, "subprocess", SimpleNamespace(run=fake_subprocess_run))
    return envs


def test_run_binary_passes_the_device_unchanged_without_visible_devices(
    tmp_path,
    monkeypatch,
    launch_envs,
):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    binary.run_binary("solver", tmp_path, device=1, verify_gpu=False)
    assert launch_envs[0]["CUDA_VISIBLE_DEVICES"] == "1"


def test_run_binary_without_gpu_check_passes_an_empty_visible_devices_variable_on(
    tmp_path,
    monkeypatch,
    launch_envs,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    binary.run_binary("solver", tmp_path, device=0, verify_gpu=False)
    assert launch_envs[0]["CUDA_VISIBLE_DEVICES"] == ""


def test_run_binary_maps_the_device_into_the_visible_devices(tmp_path, monkeypatch, launch_envs):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    binary.run_binary("solver", tmp_path, device=1, verify_gpu=False)
    assert launch_envs[0]["CUDA_VISIBLE_DEVICES"] == "3"
