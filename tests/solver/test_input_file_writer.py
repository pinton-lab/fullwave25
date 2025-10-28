from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fullwave.solver.input_file_writer import InputFileWriter
from fullwave.utils import check_functions


# Utility to create dummy Fullwave objects
def create_dummy_objects():
    grid = SimpleNamespace(
        cfl=0.5,
        dt=0.1,
        dx=0.1,
        dy=0.1,
        c0=1500,
        nx=10,
        ny=10,
        nt=100,
        is_3d=False,
    )
    medium = SimpleNamespace(
        sound_speed=np.array([1500, 1500], dtype=np.float64),
        bulk_modulus=np.array([2e9, 2e9], dtype=np.float64),
        density=np.array([1000, 1000], dtype=np.float64),
        beta=np.array([0.5, 0.5], dtype=np.float64),
        relaxation_param_dict_for_fw2={"a_pml_u1": np.array([[1.0]], dtype=np.float64)},
        n_relaxation_mechanisms=1,
        input_coords_zero=np.array([[1, 1]], dtype=np.int64),
        n_air=1,
    )
    source = SimpleNamespace(
        icmat=np.array([[1, 2], [3, 4]], dtype=np.float64),
        incoords=np.array([[1, 2], [3, 4]], dtype=np.int64),
        n_sources=2,
    )
    sensor = SimpleNamespace(
        outcoords=np.array([[1, 2], [3, 4]], dtype=np.int64),
        sampling_interval=0.5,
        n_sensors=2,
    )
    return grid, medium, source, sensor


@pytest.fixture
def work_and_bin(tmp_path):
    # Create a work directory and a fake simulation binary file.
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    bin_dir = tmp_path / "bins"
    bin_dir.mkdir()
    bin_file = bin_dir / "fullwave_solver_gpu"
    bin_file.write_text("dummy simulation binary")
    return work_dir, bin_file


def test_run_non_static_creates_simulation_files(tmp_path, work_and_bin, monkeypatch):
    work_dir, bin_file = work_and_bin
    grid, medium, source, sensor = create_dummy_objects()

    # Bypass input validations that require instance types and existing paths.
    monkeypatch.setattr(check_functions, "check_path_exists", lambda x: None)  # noqa: ARG005
    monkeypatch.setattr(check_functions, "check_instance", lambda inst, cls: None)  # noqa: ARG005

    writer = InputFileWriter(
        work_dir,
        grid,
        medium,
        source,
        sensor,
        path_fullwave_simulation_bin=bin_file,
        validate_input=False,
    )
    sim_dir_name = "sim_test"
    sim_dir = writer.run(sim_dir_name, is_static_map=False, recalculate_pml=True)
    sim_path = Path(sim_dir)
    assert sim_path.exists()

    # Check that key simulation files were created.
    expected_files = [
        "icmat.dat",
        "d.dat",
        "dmap.dat",
        "ndmap.dat",
        "dcmap.dat",
        "c.dat",
        "K.dat",
        "rho.dat",
        "beta.dat",
        bin_file.name,
    ]
    for fname in expected_files:
        file_path = sim_path / fname
        assert file_path.exists(), f"Expected file {fname} does not exist."


def test_run_static_creates_symbolic_links(tmp_path, work_and_bin, monkeypatch):
    work_dir, bin_file = work_and_bin
    grid, medium, source, sensor = create_dummy_objects()

    # Create dummy data files in work_dir to be linked.
    dummy_filenames = ["c.dat", "K.dat", "rho.dat", "beta.dat", "dX.dat"]
    for fname in dummy_filenames:
        (work_dir / fname).write_text("dummy content")

    monkeypatch.setattr(check_functions, "check_path_exists", lambda x: None)  # noqa: ARG005
    monkeypatch.setattr(check_functions, "check_instance", lambda inst, cls: None)  # noqa: ARG005

    writer = InputFileWriter(
        work_dir,
        grid,
        medium,
        source,
        sensor,
        path_fullwave_simulation_bin=bin_file,
        validate_input=False,
    )
    sim_dir_name = "sim_static"
    sim_dir = writer.run(sim_dir_name, is_static_map=True, recalculate_pml=True)
    sim_path = Path(sim_dir)
    assert sim_path.exists()

    # Check that symbolic link for one of the expected files exists.
    src_file = work_dir / "c.dat"
    dst_file = sim_path / "c.dat"
    assert dst_file.exists(), "c.dat was not created in the simulation directory."
    assert dst_file.is_symlink(), "c.dat is not a symbolic link."
    # Verify that the symlink points to the correct source.
    assert dst_file.samefile(src_file)
