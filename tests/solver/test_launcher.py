import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from fullwave.solver.launcher import Launcher, SimulationError


def dummy_run_success(command, stdout, check, shell, stderr, text):
    # Create the expected output file "genout.dat" in the current directory.
    with Path("genout.dat").open("w", encoding="utf-8") as f:
        f.write("dummy")


def dummy_run_failure(command, stdout, check, shell):
    raise subprocess.CalledProcessError(returncode=1, cmd=command)


def selective_run_wrapper(monkeypatch, *, success=True, gpu_ids=None):
    if gpu_ids is None:
        gpu_ids = [0]
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    def selective_run(args, *p, **kw):
        if "/usr/bin/nvidia-smi" in args:
            # Behavior for special locations
            return dummy_nvidia_smi(args, monkeypatch, gpu_ids, *p, **kw)
        if success:
            return dummy_run_success(args, *p, **kw)
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


def dummy_nvidia_smi(args, monkeypatch, gpu_ids=None, *p, **kw):
    if gpu_ids is None:
        gpu_ids = [0]

    def mock_nvidia_smi(command, check, stdout, encoding, shell):
        class MockResult:
            stdout = "\n".join(
                [
                    f"GPU {i}: NVIDIA GPU (UUID: GPU-XXXXXXXX-XXXX-XXXX-XXXX-XXXXXXXXXXXX)"
                    for i in gpu_ids
                ],
            )

        return MockResult()

    return mock_nvidia_smi(args, *p, **kw)


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
    result = Launcher._configure_cuda_device_id(None)  # noqa: SLF001
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
    result = Launcher._configure_cuda_device_id(0)  # noqa: SLF001
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
    result = Launcher._configure_cuda_device_id("1")  # noqa: SLF001
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
    result = Launcher._configure_cuda_device_id([0, 1, 2])  # noqa: SLF001
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
        Launcher._configure_cuda_device_id(-1)  # noqa: SLF001


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
        Launcher._configure_cuda_device_id("invalid")  # noqa: SLF001


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
        Launcher._configure_cuda_device_id([0, -1, 2])  # noqa: SLF001


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
        Launcher._configure_cuda_device_id(math.pi)  # noqa: SLF001


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

    # Mock nvidia-smi
    def mock_nvidia_smi(command, check, stdout, encoding, shell):
        class MockResult:
            stdout = "GPU 0: NVIDIA GPU\n"

        return MockResult()

    monkeypatch.setattr(subprocess, "run", mock_nvidia_smi)
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/nvidia-smi")

    launcher = Launcher(path_fullwave_simulation_bin=dummy_bin)
    assert launcher.is_3d is False
    assert launcher.use_gpu is True
    assert launcher.cuda_device_id == "0"


def test_run_cpu_not_implemented(tmp_path, monkeypatch):
    dummy_bin, sim_dir = get_dummy_bin(tmp_path)

    # Mock nvidia-smi
    def mock_nvidia_smi(command, check, stdout, encoding, shell):
        class MockResult:
            stdout = "GPU 0: NVIDIA GPU\n"

        return MockResult()

    monkeypatch.setattr(subprocess, "run", mock_nvidia_smi)
    monkeypatch.setattr(shutil, "which", lambda _: "/usr/bin/nvidia-smi")

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
