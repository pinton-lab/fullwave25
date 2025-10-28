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


def test_run_success(tmp_path, monkeypatch):
    # Create a temporary simulation directory.
    sim_dir = tmp_path / "sim_dir"
    sim_dir.mkdir()

    # Create a dummy simulation binary file.
    dummy_bin = tmp_path / "dummy_bin"
    dummy_bin.write_text("")

    # In the simulation directory, create a dummy executable with the same name.
    sim_bin = sim_dir / dummy_bin.name
    sim_bin.write_text("")

    # Monkeypatch subprocess.run to simulate successful execution.
    monkeypatch.setattr(subprocess, "run", dummy_run_success)

    # Monkeypatch load_dat_data to return a dummy numpy array.
    monkeypatch.setattr("fullwave.solver.launcher.load_dat_data", lambda _: np.array([42]))

    launcher = Launcher(path_fullwave_simulation_bin=dummy_bin)

    result = launcher.run(sim_dir)
    np.testing.assert_array_equal(result, np.array([42]))


def test_run_failure_due_to_subprocess(tmp_path, monkeypatch):
    sim_dir = tmp_path / "sim_dir"
    sim_dir.mkdir()

    dummy_bin = tmp_path / "dummy_bin"
    dummy_bin.write_text("")

    # Create the simulation binary file in sim_dir.
    sim_bin = sim_dir / dummy_bin.name
    sim_bin.write_text("")

    # Monkeypatch subprocess.run to simulate a failure.
    monkeypatch.setattr(subprocess, "run", dummy_run_failure)

    launcher = Launcher(path_fullwave_simulation_bin=dummy_bin)
    cwd = Path.cwd()
    with pytest.raises(SimulationError):
        launcher.run(sim_dir)

    # Ensure the current directory is restored.
    assert Path.cwd() == cwd
