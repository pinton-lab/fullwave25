import math
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fullwave.solver.launcher import Launcher, SimulationError


def dummy_run_success(command, stdout, check, shell, stderr, text, env, *, captured_envs=None):
    if captured_envs is not None:
        captured_envs.append(env)
    # Create the expected output file "genout.dat" in the current directory.
    with Path("genout.dat").open("w", encoding="utf-8") as f:
        f.write("dummy")


def dummy_run_failure(command, stdout, check, shell):
    raise subprocess.CalledProcessError(returncode=1, cmd=command)


@pytest.fixture(autouse=True)
def _no_visible_devices_variable(monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)


def fake_visible_gpus(monkeypatch, *, n_gpus):
    monkeypatch.setattr("fullwave.solver.launcher.count_visible_gpus", lambda: n_gpus)


def selective_run_wrapper(monkeypatch, *, success=True, gpu_ids=None, captured_envs=None):
    fake_visible_gpus(monkeypatch, n_gpus=1 if gpu_ids is None else len(gpu_ids))

    def selective_run(args, *p, **kw):
        if success:
            return dummy_run_success(args, *p, **kw, captured_envs=captured_envs)
        return dummy_run_failure(args, *p, **kw)

    return selective_run


def get_dummy_bin(tmp_path):
    # Create a temporary simulation directory.
    sim_dir = tmp_path / "sim_dir"
    sim_dir.mkdir()

    # Create a dummy simulation binary file.
    dummy_bin = tmp_path / "dummy_bin"
    dummy_bin.write_text("")

    # In the simulation directory, create a dummy executable with the same name.
    sim_bin = sim_dir / dummy_bin.name
    sim_bin.write_text("")
    return dummy_bin, sim_dir


def test_run_success(tmp_path, monkeypatch):
    dummy_bin, sim_dir = get_dummy_bin(tmp_path)
    gpu_ids = [0, 1]
    cuda_device_id = "0"
    # Monkeypatch subprocess.run to simulate successful execution.
    monkeypatch.setattr(
        subprocess,
        "run",
        selective_run_wrapper(success=True, monkeypatch=monkeypatch, gpu_ids=gpu_ids),
    )

    # Monkeypatch load_dat_data to return a dummy numpy array.
    monkeypatch.setattr("fullwave.solver.launcher.load_dat_data", lambda _: np.array([42]))

    launcher = Launcher(
        path_fullwave_simulation_bin=dummy_bin,
        cuda_device_id=cuda_device_id,
    )

    result = launcher.run(sim_dir)
    np.testing.assert_array_equal(result, np.array([42]))


def test_run_failure_due_to_subprocess(tmp_path, monkeypatch):
    dummy_bin, sim_dir = get_dummy_bin(tmp_path)

    gpu_ids = [0, 1]
    cuda_device_id = "0"

    # Monkeypatch subprocess.run to simulate a failure.
    monkeypatch.setattr(
        subprocess,
        "run",
        selective_run_wrapper(success=False, monkeypatch=monkeypatch, gpu_ids=gpu_ids),
    )

    launcher = Launcher(
        path_fullwave_simulation_bin=dummy_bin,
        cuda_device_id=cuda_device_id,
    )
    cwd = Path.cwd()
    with pytest.raises(SimulationError):
        launcher.run(sim_dir)

    # Ensure the current directory is restored.
    assert Path.cwd() == cwd


def test_configure_cuda_device_id_none(monkeypatch):
    gpu_ids = [0]

    # Monkeypatch subprocess.run (nvidia-smi)
    monkeypatch.setattr(
        subprocess,
        "run",
        selective_run_wrapper(monkeypatch=monkeypatch, gpu_ids=gpu_ids),
    )

    # Test with None input
    result = Launcher._configure_cuda_device_id(None)
    assert result == "0"


def test_configure_cuda_device_id_int(monkeypatch):
    gpu_ids = [0]

    # Monkeypatch subprocess.run (nvidia-smi)
    monkeypatch.setattr(
        subprocess,
        "run",
        selective_run_wrapper(monkeypatch=monkeypatch, gpu_ids=gpu_ids),
    )
    # Test with integer input
    result = Launcher._configure_cuda_device_id(0)
    assert result == "0"


def test_configure_cuda_device_id_string(monkeypatch):
    gpu_ids = [0, 1]

    # Monkeypatch subprocess.run (nvidia-smi)
    monkeypatch.setattr(
        subprocess,
        "run",
        selective_run_wrapper(monkeypatch=monkeypatch, gpu_ids=gpu_ids),
    )

    # Test with string input
    result = Launcher._configure_cuda_device_id("1")
    assert result == "1"


def test_configure_cuda_device_id_list(monkeypatch):
    gpu_ids = [0, 1, 2]

    # Monkeypatch subprocess.run (nvidia-smi)
    monkeypatch.setattr(
        subprocess,
        "run",
        selective_run_wrapper(monkeypatch=monkeypatch, gpu_ids=gpu_ids),
    )
    # Test with list input
    result = Launcher._configure_cuda_device_id([0, 1, 2])
    assert result == "0,1,2"


def test_configure_cuda_device_id_negative_int(monkeypatch):
    gpu_ids = [0]

    # Monkeypatch subprocess.run (nvidia-smi)
    monkeypatch.setattr(
        subprocess,
        "run",
        selective_run_wrapper(monkeypatch=monkeypatch, gpu_ids=gpu_ids),
    )
    # Test with negative integer
    with pytest.raises(ValueError, match="CUDA device ID must be a non-negative integer"):
        Launcher._configure_cuda_device_id(-1)


def test_configure_cuda_device_id_invalid_string(monkeypatch):
    gpu_ids = [0]

    # Monkeypatch subprocess.run (nvidia-smi)
    monkeypatch.setattr(
        subprocess,
        "run",
        selective_run_wrapper(monkeypatch=monkeypatch, gpu_ids=gpu_ids),
    )
    # Test with invalid string
    with pytest.raises(
        ValueError,
        match="CUDA device ID string must represent a non-negative integer",
    ):
        Launcher._configure_cuda_device_id("invalid")


def test_configure_cuda_device_id_list_with_negative(monkeypatch):
    gpu_ids = [0, 1, 2]

    # Monkeypatch subprocess.run (nvidia-smi)
    monkeypatch.setattr(
        subprocess,
        "run",
        selective_run_wrapper(monkeypatch=monkeypatch, gpu_ids=gpu_ids),
    )

    # Test with list containing negative integers
    with pytest.raises(
        ValueError,
        match="All CUDA device IDs in the list must be non-negative integers",
    ):
        Launcher._configure_cuda_device_id([0, -1, 2])


def test_configure_cuda_device_id_invalid_type(monkeypatch):
    gpu_ids = [0, 1, 2]

    # Monkeypatch subprocess.run (nvidia-smi)
    monkeypatch.setattr(
        subprocess,
        "run",
        selective_run_wrapper(monkeypatch=monkeypatch, gpu_ids=gpu_ids),
    )

    # Test with invalid type
    with pytest.raises(
        ValueError,
        match="CUDA device ID must be an integer, string, list, or None",
    ):
        Launcher._configure_cuda_device_id(math.pi)


def test_launcher_init_invalid_bin_path(monkeypatch):
    gpu_ids = [0, 1, 2]

    # Monkeypatch subprocess.run (nvidia-smi)
    monkeypatch.setattr(
        subprocess,
        "run",
        selective_run_wrapper(monkeypatch=monkeypatch, gpu_ids=gpu_ids),
    )

    # Test launcher initialization with invalid binary path
    invalid_path = Path("/nonexistent/path/to/binary")
    with pytest.raises(AssertionError, match="Fullwave simulation binary not found"):
        Launcher(path_fullwave_simulation_bin=invalid_path)


def test_launcher_init_defaults(tmp_path, monkeypatch):
    # Create a dummy binary for testing
    dummy_bin = tmp_path / "dummy_bin"
    dummy_bin.write_text("")
    fake_visible_gpus(monkeypatch, n_gpus=1)

    launcher = Launcher(path_fullwave_simulation_bin=dummy_bin)
    assert launcher.is_3d is False
    assert launcher.use_gpu is True
    assert launcher.cuda_device_id == "0"


def test_run_cpu_not_implemented(tmp_path, monkeypatch):
    dummy_bin, sim_dir = get_dummy_bin(tmp_path)
    fake_visible_gpus(monkeypatch, n_gpus=1)

    launcher = Launcher(path_fullwave_simulation_bin=dummy_bin, use_gpu=False)

    with pytest.raises(NotImplementedError, match="Currently, only GPU version is supported"):
        launcher.run(sim_dir)


def test_run_load_results_false(tmp_path, monkeypatch):
    dummy_bin, sim_dir = get_dummy_bin(tmp_path)

    # Monkeypatch subprocess.run to simulate successful execution
    gpu_ids = [0, 1, 2]

    # Monkeypatch subprocess.run (nvidia-smi and stdbuf)
    monkeypatch.setattr(
        subprocess,
        "run",
        selective_run_wrapper(monkeypatch=monkeypatch, gpu_ids=gpu_ids),
    )

    launcher = Launcher(path_fullwave_simulation_bin=dummy_bin)

    result = launcher.run(sim_dir, load_results=False)
    assert isinstance(result, Path)
    assert result == sim_dir.absolute() / "genout.dat"


def test_run_changes_directory_back_on_success(tmp_path, monkeypatch):
    dummy_bin, sim_dir = get_dummy_bin(tmp_path)

    # Monkeypatch subprocess.run to simulate successful execution
    gpu_ids = [0, 1, 2]

    # Monkeypatch subprocess.run (nvidia-smi and stdbuf)
    monkeypatch.setattr(
        subprocess,
        "run",
        selective_run_wrapper(monkeypatch=monkeypatch, gpu_ids=gpu_ids),
    )

    monkeypatch.setattr("fullwave.solver.launcher.load_dat_data", lambda _: np.array([42]))

    launcher = Launcher(path_fullwave_simulation_bin=dummy_bin)

    original_cwd = Path.cwd()
    launcher.run(sim_dir)
    assert Path.cwd() == original_cwd


def _replace_peer_check(monkeypatch, result):
    asked = []

    def fake_check(ordinals):
        asked.append(list(ordinals))
        if isinstance(result, Exception):
            raise result
        return result

    monkeypatch.setattr(
        "fullwave.solver.launcher.find_consecutive_gpus_without_peer_access",
        fake_check,
    )
    return asked


def test_warn_if_gpus_lack_peer_access_warns_for_each_pair(monkeypatch, caplog):
    asked = _replace_peer_check(monkeypatch, [(0, 1), (2, 3)])
    with caplog.at_level("WARNING"):
        Launcher._warn_if_gpus_lack_peer_access("0,1,2,3")
    assert asked == [[0, 1, 2, 3]]
    assert "GPU 0 <-> GPU 1, GPU 2 <-> GPU 3" in caplog.text
    assert "much slower" in caplog.text


def test_warn_if_gpus_lack_peer_access_is_silent_with_peer_access(monkeypatch, caplog):
    _replace_peer_check(monkeypatch, [])
    with caplog.at_level("WARNING"):
        Launcher._warn_if_gpus_lack_peer_access("0,1")
    assert caplog.text == ""


def test_warn_if_gpus_lack_peer_access_skips_one_gpu(monkeypatch, caplog):
    asked = _replace_peer_check(monkeypatch, [(0, 1)])
    with caplog.at_level("WARNING"):
        Launcher._warn_if_gpus_lack_peer_access("2")
    assert asked == []
    assert caplog.text == ""


def test_warn_if_gpus_lack_peer_access_checks_the_visible_ordinals(monkeypatch, caplog):
    asked = _replace_peer_check(monkeypatch, [(0, 1)])
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,2")
    with caplog.at_level("WARNING"):
        Launcher._warn_if_gpus_lack_peer_access("0,1")
    assert asked == [[0, 1]]
    assert "GPU 0 <-> GPU 1" in caplog.text


def test_warn_if_gpus_lack_peer_access_never_raises(monkeypatch, caplog):
    _replace_peer_check(monkeypatch, RuntimeError("cuDeviceGet failed with error code 101"))
    with caplog.at_level("WARNING"):
        Launcher._warn_if_gpus_lack_peer_access("0,1")
    assert caplog.text == ""


def _run_and_capture_env(
    tmp_path,
    monkeypatch,
    *,
    cuda_device_id,
    n_visible_gpus,
    verify_gpu=True,
):
    dummy_bin, sim_dir = get_dummy_bin(tmp_path)
    captured_envs = []
    monkeypatch.setattr(
        subprocess,
        "run",
        selective_run_wrapper(
            monkeypatch=monkeypatch,
            gpu_ids=list(range(n_visible_gpus)),
            captured_envs=captured_envs,
        ),
    )
    monkeypatch.setattr("fullwave.solver.launcher.load_dat_data", lambda _: np.array([42]))
    monkeypatch.setattr(
        "fullwave.solver.launcher.find_consecutive_gpus_without_peer_access",
        lambda _ordinals: [],
    )
    launcher = Launcher(
        path_fullwave_simulation_bin=dummy_bin,
        cuda_device_id=cuda_device_id,
        verify_gpu=verify_gpu,
    )
    launcher.run(sim_dir)
    return captured_envs[0]


def test_run_without_gpu_check_passes_an_empty_visible_devices_variable_on(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    env = _run_and_capture_env(
        tmp_path,
        monkeypatch,
        cuda_device_id=None,
        n_visible_gpus=0,
        verify_gpu=False,
    )
    assert env["CUDA_VISIBLE_DEVICES"] == ""


def test_run_passes_the_ids_unchanged_without_visible_devices(tmp_path, monkeypatch):
    env = _run_and_capture_env(tmp_path, monkeypatch, cuda_device_id=[2, 3], n_visible_gpus=4)
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_run_maps_the_ids_into_the_visible_devices(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    env = _run_and_capture_env(tmp_path, monkeypatch, cuda_device_id=[0, 1], n_visible_gpus=2)
    assert env["CUDA_VISIBLE_DEVICES"] == "2,3"


def test_run_uses_the_first_visible_device_by_default(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    env = _run_and_capture_env(tmp_path, monkeypatch, cuda_device_id=None, n_visible_gpus=2)
    assert env["CUDA_VISIBLE_DEVICES"] == "2"


def test_configure_cuda_device_id_rejects_an_id_beyond_the_visible_devices(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    fake_visible_gpus(monkeypatch, n_gpus=2)
    with pytest.raises(ValueError, match=r"\[2\] counts within CUDA_VISIBLE_DEVICES=2,3"):
        Launcher._configure_cuda_device_id([2])


def test_configure_cuda_device_id_rejects_an_empty_visible_devices_variable(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    fake_visible_gpus(monkeypatch, n_gpus=0)
    with pytest.raises(ValueError, match="CUDA_VISIBLE_DEVICES is empty"):
        Launcher._configure_cuda_device_id(None)


def test_configure_cuda_device_id_rejects_an_id_the_driver_does_not_list(monkeypatch):
    fake_visible_gpus(monkeypatch, n_gpus=2)
    with pytest.raises(ValueError, match="CUDA device ID 2 does not exist; this process sees 2"):
        Launcher._configure_cuda_device_id(2)
