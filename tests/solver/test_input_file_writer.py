from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from fullwave.solver.input_file_writer import InputFileWriter
from fullwave.utils import check_functions
from fullwave.utils.numerical import matlab_round


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
        air_coords=np.array([[0, 0], [1, 1]], dtype=np.int64),
        n_air=1,
    )
    source = SimpleNamespace(
        icmat=np.array([[1, 2], [3, 4]], dtype=np.float64),
        incoords=np.array([[1, 2], [3, 4]], dtype=np.int64),
        n_sources=2,
        p0_additive=None,
        incoords_add=None,
        n_sources_add=0,
    )
    sensor = SimpleNamespace(
        outcoords=np.array([[1, 2], [3, 4]], dtype=np.int64),
        sampling_modulus_time=0.5,
        n_sensors=2,
        mod_x=0,
        mod_y=0,
        mod_z=0,
        is_sparse_grid=False,
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


def test_the_interior_offset_is_written_without_a_sparse_sensor(
    tmp_path, work_and_bin, monkeypatch
):
    """`pml_thickness.dat` says where the interior starts, on every path.

    A solver that places an absorbing layer needs it, and the sensor mode has
    nothing to do with where the interior is.
    """
    work_dir, bin_file = work_and_bin
    grid, medium, source, sensor = create_dummy_objects()
    assert sensor.is_sparse_grid is False

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
        pml_thickness=37,
    )
    sim_path = Path(writer.run("sim_offset", is_static_map=False, recalculate_pml=True))

    written = sim_path / "pml_thickness.dat"
    assert written.exists(), "the interior offset must be written without a sparse sensor"
    assert int(np.fromfile(written, dtype=np.int32)[0]) == 37

    # The sparse strides stay behind the sensor mode, so an older binary that
    # does not expect them still sees none.
    for name in ("modX.dat", "modY.dat", "modZ.dat"):
        assert not (sim_path / name).exists(), f"{name} must stay behind the sparse sensor"

    # The layer's own files stay behind the layer, so a run that asks for no
    # layer writes none and a binary that finds none places none.
    for name in (
        "exponential_attenuation_pml_thickness.dat",
        "exponential_attenuation_pml_interior_offset.dat",
    ):
        assert not (sim_path / name).exists(), f"{name} must stay behind the absorbing layer"


def test_the_absorbing_layer_carries_its_own_two_files(tmp_path, work_and_bin, monkeypatch):
    """The C-PML layer states its thickness and where the interior starts.

    Both are its own files. ``pml_thickness.dat`` cannot serve, because it
    carries the sparse recording window and that is 0 for a whole domain
    recording while the interior still starts where it always did.
    """
    work_dir, bin_file = work_and_bin
    grid, medium, source, sensor = create_dummy_objects()

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
        pml_thickness=0,
        exponential_attenuation_pml_thickness_px=20,
        exponential_attenuation_pml_interior_offset_px=48,
    )
    sim_path = Path(writer.run("sim_layer", is_static_map=False, recalculate_pml=True))

    thickness = sim_path / "exponential_attenuation_pml_thickness.dat"
    offset = sim_path / "exponential_attenuation_pml_interior_offset.dat"
    assert thickness.exists(), "the layer must state its thickness"
    assert offset.exists(), "the layer must state where the interior starts"
    assert int(np.fromfile(thickness, dtype=np.int32)[0]) == 20
    assert int(np.fromfile(offset, dtype=np.int32)[0]) == 48

    # The recording window is a different fact and it keeps its own file.
    assert int(np.fromfile(sim_path / "pml_thickness.dat", dtype=np.int32)[0]) == 0


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


def test_run_with_p0_additive_writes_icmat_add(tmp_path, work_and_bin, monkeypatch):
    """When source has p0_additive, icmat_add.dat is written with same layout as icmat.dat."""
    work_dir, bin_file = work_and_bin
    grid, medium, source, sensor = create_dummy_objects()
    # Add additive source term (same shape as icmat: n_sources x nt)
    source.p0_additive = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)

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
    sim_dir_name = "sim_additive"
    sim_dir = writer.run(sim_dir_name, is_static_map=False, recalculate_pml=True)
    sim_path = Path(sim_dir)

    assert (sim_path / "icmat.dat").exists()
    additive_path = sim_path / "icmat_add.dat"
    assert additive_path.exists(), "icmat_add.dat should be written when p0_additive is set."

    # _write_ic stores (icmat.T).T = icmat in row-major, so (n_sources, nt)
    additive_data = np.fromfile(additive_path, dtype=np.float32)
    expected = source.p0_additive.astype(np.float32).ravel()
    np.testing.assert_array_almost_equal(additive_data, expected)


def test_run_additive_only_writes_icmat_zeros_and_icmat_add(tmp_path, work_and_bin, monkeypatch):
    """When source has only p0_additive (p0=None), icmat.dat is zeros, icmat_add.dat has data."""
    work_dir, bin_file = work_and_bin
    grid, medium, source, sensor = create_dummy_objects()
    source.icmat = np.zeros((2, 2), dtype=np.float64)
    source.p0_additive = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)

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
    sim_dir_name = "sim_additive_only"
    sim_dir = writer.run(sim_dir_name, is_static_map=False, recalculate_pml=True)
    sim_path = Path(sim_dir)

    icmat_path = sim_path / "icmat.dat"
    assert icmat_path.exists()
    icmat_data = np.fromfile(icmat_path, dtype=np.float32)
    np.testing.assert_array_almost_equal(icmat_data, np.zeros(4))

    additive_path = sim_path / "icmat_add.dat"
    assert additive_path.exists()
    additive_data = np.fromfile(additive_path, dtype=np.float32)
    expected = source.p0_additive.astype(np.float32).ravel()
    np.testing.assert_array_almost_equal(additive_data, expected)


def test_run_with_incoords_add_writes_icc_add_and_ncoords_add(
    tmp_path,
    work_and_bin,
    monkeypatch,
):
    """When source has p0_additive and incoords_add, icc_add.dat and ncoords_add.dat are written."""
    work_dir, bin_file = work_and_bin
    grid, medium, source, sensor = create_dummy_objects()
    source.p0_additive = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float64)
    source.incoords_add = np.array([[2, 3], [4, 5]], dtype=np.int64)
    source.n_sources_add = 2

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
    sim_dir_name = "sim_icc_add"
    sim_dir = writer.run(sim_dir_name, is_static_map=False, recalculate_pml=True)
    sim_path = Path(sim_dir)

    icc_add_path = sim_path / "icc_add.dat"
    assert icc_add_path.exists()
    icc_add_data = np.fromfile(icc_add_path, dtype=np.int32)
    np.testing.assert_array_equal(icc_add_data, source.incoords_add.ravel())

    ncoords_add_path = sim_path / "ncoords_add.dat"
    assert ncoords_add_path.exists()
    ncoords_add_val = np.fromfile(ncoords_add_path, dtype=np.int32)
    assert ncoords_add_val == 2


def _source_with_velocity(**vel_attrs):
    """Return a dummy source SimpleNamespace with optional velocity attributes."""
    src = SimpleNamespace(
        icmat=np.array([[1, 2], [3, 4]], dtype=np.float64),
        incoords=np.array([[1, 2], [3, 4]], dtype=np.int64),
        n_sources=2,
        p0_additive=None,
        incoords_add=None,
        n_sources_add=0,
        u0=None,
        incoords_u=None,
        n_sources_u=0,
        v0=None,
        incoords_v=None,
        n_sources_v=0,
        w0=None,
        incoords_w=None,
        n_sources_w=0,
    )
    for k, v in vel_attrs.items():
        setattr(src, k, v)
    return src


def _run_writer(work_dir, bin_file, source, monkeypatch, sim_dir_name="sim_vel"):
    """Build writer, bypass validation, run, and return sim_path."""
    grid, medium, _, sensor = create_dummy_objects()
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
    return Path(writer.run(sim_dir_name, is_static_map=False, recalculate_pml=True))


class TestVelocitySourceWriter:
    """Tests that InputFileWriter emits correct files for velocity source components."""

    def test_u0_writes_icmat_u_and_icc_u(self, work_and_bin, monkeypatch):
        """u0 + incoords_u → icmat_u.dat and icc_u.dat are created."""
        work_dir, bin_file = work_and_bin
        u0 = np.array([[0.5, 1.0]], dtype=np.float64)
        incoords_u = np.array([[2, 3]], dtype=np.int64)
        source = _source_with_velocity(u0=u0, incoords_u=incoords_u, n_sources_u=1)

        sim_path = _run_writer(work_dir, bin_file, source, monkeypatch)

        assert (sim_path / "icmat_u.dat").exists(), "icmat_u.dat not created"
        assert (sim_path / "icc_u.dat").exists(), "icc_u.dat not created"

    def test_u0_content(self, work_and_bin, monkeypatch):
        """icmat_u.dat contains u0 data; icc_u.dat contains incoords_u."""
        work_dir, bin_file = work_and_bin
        u0 = np.array([[0.5, 1.0]], dtype=np.float64)  # shape (1, 2) — transpose is still (2, 1)
        incoords_u = np.array([[2, 3]], dtype=np.int64)
        source = _source_with_velocity(u0=u0, incoords_u=incoords_u, n_sources_u=1)

        sim_path = _run_writer(work_dir, bin_file, source, monkeypatch)

        # icmat_u.dat is written as np.transpose(u0) in float32
        icmat_u_data = np.fromfile(sim_path / "icmat_u.dat", dtype=np.float32)
        np.testing.assert_array_almost_equal(icmat_u_data, u0.astype(np.float32).ravel())

        # icc_u.dat written as int32
        icc_u_data = np.fromfile(sim_path / "icc_u.dat", dtype=np.int32)
        np.testing.assert_array_equal(icc_u_data, incoords_u.ravel())

    def test_ncoords_u_dat_written_with_correct_count(self, work_and_bin, monkeypatch):
        """ncoords_u.dat is written with the correct integer count."""
        work_dir, bin_file = work_and_bin
        u0 = np.array([[0.5, 1.0], [0.2, 0.3]], dtype=np.float64)
        incoords_u = np.array([[0, 1], [1, 0]], dtype=np.int64)
        source = _source_with_velocity(u0=u0, incoords_u=incoords_u, n_sources_u=2)

        sim_path = _run_writer(work_dir, bin_file, source, monkeypatch)

        ncoords_u_path = sim_path / "ncoords_u.dat"
        assert ncoords_u_path.exists(), "ncoords_u.dat not created"
        val = np.fromfile(ncoords_u_path, dtype=np.int32)
        assert int(val) == 2

    def test_all_three_velocity_components_write_files(self, work_and_bin, monkeypatch):
        """All three u/v/w components produce their respective files."""
        work_dir, bin_file = work_and_bin
        u0 = np.array([[1.0, 2.0]], dtype=np.float64)
        v0 = np.array([[3.0, 4.0]], dtype=np.float64)
        w0 = np.array([[5.0, 6.0]], dtype=np.float64)
        source = _source_with_velocity(
            u0=u0,
            incoords_u=np.array([[0, 1]], dtype=np.int64),
            n_sources_u=1,
            v0=v0,
            incoords_v=np.array([[1, 0]], dtype=np.int64),
            n_sources_v=1,
            w0=w0,
            incoords_w=np.array([[0, 0]], dtype=np.int64),
            n_sources_w=1,
        )

        sim_path = _run_writer(work_dir, bin_file, source, monkeypatch)

        for suffix in ("u", "v", "w"):
            assert (sim_path / f"icmat_{suffix}.dat").exists(), f"icmat_{suffix}.dat missing"
            assert (sim_path / f"icc_{suffix}.dat").exists(), f"icc_{suffix}.dat missing"
            assert (sim_path / f"ncoords_{suffix}.dat").exists(), f"ncoords_{suffix}.dat missing"

    def test_no_velocity_source_no_velocity_files(self, work_and_bin, monkeypatch):
        """When no velocity source is set, no velocity-related files are created."""
        work_dir, bin_file = work_and_bin
        _, _, source, _ = create_dummy_objects()

        sim_path = _run_writer(work_dir, bin_file, source, monkeypatch)

        for suffix in ("u", "v", "w"):
            assert not (sim_path / f"icmat_{suffix}.dat").exists()
            assert not (sim_path / f"icc_{suffix}.dat").exists()
            assert not (sim_path / f"ncoords_{suffix}.dat").exists()

    def test_only_v_component_writes_only_v_files(self, work_and_bin, monkeypatch):
        """Only v present → only icmat_v/icc_v/ncoords_v written; u and w absent."""
        work_dir, bin_file = work_and_bin
        v0 = np.array([[7.0, 8.0]], dtype=np.float64)
        source = _source_with_velocity(
            v0=v0,
            incoords_v=np.array([[3, 3]], dtype=np.int64),
            n_sources_v=1,
        )

        sim_path = _run_writer(work_dir, bin_file, source, monkeypatch)

        assert (sim_path / "icmat_v.dat").exists()
        assert (sim_path / "icc_v.dat").exists()
        assert (sim_path / "ncoords_v.dat").exists()
        assert not (sim_path / "icmat_u.dat").exists()
        assert not (sim_path / "icc_u.dat").exists()
        assert not (sim_path / "ncoords_u.dat").exists()
        assert not (sim_path / "icmat_w.dat").exists()
        assert not (sim_path / "icc_w.dat").exists()
        assert not (sim_path / "ncoords_w.dat").exists()

    def test_velocity_only_no_pressure_files_are_empty(self, work_and_bin, monkeypatch):
        """Velocity-only source (n_sources=0) writes empty icmat/icc and correct ncoords=0."""
        work_dir, bin_file = work_and_bin
        u0 = np.array([[1.0, 2.0]], dtype=np.float64)
        source = _source_with_velocity(
            # override pressure fields to match a velocity-only source
            icmat=np.zeros((0, 2), dtype=np.float64),
            incoords=np.empty((0, 2), dtype=np.int64),
            n_sources=0,
            u0=u0,
            incoords_u=np.array([[0, 1]], dtype=np.int64),
            n_sources_u=1,
        )

        sim_path = _run_writer(work_dir, bin_file, source, monkeypatch)

        # ncoords.dat should be 0
        ncoords_val = np.fromfile(sim_path / "ncoords.dat", dtype=np.int32)
        assert int(ncoords_val) == 0

        # icmat_u.dat and icc_u.dat should exist with correct content
        assert (sim_path / "icmat_u.dat").exists()
        assert (sim_path / "ncoords_u.dat").exists()
        ncoords_u_val = np.fromfile(sim_path / "ncoords_u.dat", dtype=np.int32)
        assert int(ncoords_u_val) == 1

    def test_all_five_components_written_and_content_correct(
        self,
        work_and_bin,
        monkeypatch,
    ):
        """p, additive, u, v, and w all set → all files written with correct binary content."""
        work_dir, bin_file = work_and_bin

        p0 = np.array([[1.0, 2.0]], dtype=np.float64)
        p0_additive = np.array([[0.1, 0.2]], dtype=np.float64)
        u0 = np.array([[3.0, 4.0]], dtype=np.float64)
        v0 = np.array([[5.0, 6.0]], dtype=np.float64)
        w0 = np.array([[7.0, 8.0]], dtype=np.float64)

        incoords = np.array([[0, 0]], dtype=np.int64)
        incoords_add = np.array([[1, 0]], dtype=np.int64)
        incoords_u = np.array([[0, 1]], dtype=np.int64)
        incoords_v = np.array([[1, 1]], dtype=np.int64)
        incoords_w = np.array([[2, 0]], dtype=np.int64)

        source = SimpleNamespace(
            icmat=p0,
            incoords=incoords,
            n_sources=1,
            p0_additive=p0_additive,
            incoords_add=incoords_add,
            n_sources_add=1,
            u0=u0,
            incoords_u=incoords_u,
            n_sources_u=1,
            v0=v0,
            incoords_v=incoords_v,
            n_sources_v=1,
            w0=w0,
            incoords_w=incoords_w,
            n_sources_w=1,
        )

        sim_path = _run_writer(work_dir, bin_file, source, monkeypatch)

        # --- pressure (hard) ---
        icmat_data = np.fromfile(sim_path / "icmat.dat", dtype=np.float32)
        np.testing.assert_array_almost_equal(icmat_data, p0.astype(np.float32).ravel())
        icc_data = np.fromfile(sim_path / "icc.dat", dtype=np.int32)
        np.testing.assert_array_equal(icc_data, incoords.ravel())

        # --- additive (soft) ---
        icmat_add_data = np.fromfile(sim_path / "icmat_add.dat", dtype=np.float32)
        np.testing.assert_array_almost_equal(icmat_add_data, p0_additive.astype(np.float32).ravel())
        icc_add_data = np.fromfile(sim_path / "icc_add.dat", dtype=np.int32)
        np.testing.assert_array_equal(icc_add_data, incoords_add.ravel())
        ncoords_add_val = np.fromfile(sim_path / "ncoords_add.dat", dtype=np.int32)
        assert int(ncoords_add_val) == 1

        # --- velocity u, v, w ---
        for suffix, sig, icc in (
            ("u", u0, incoords_u),
            ("v", v0, incoords_v),
            ("w", w0, incoords_w),
        ):
            icmat_vel = np.fromfile(sim_path / f"icmat_{suffix}.dat", dtype=np.float32)
            np.testing.assert_array_almost_equal(icmat_vel, sig.astype(np.float32).ravel())
            icc_vel = np.fromfile(sim_path / f"icc_{suffix}.dat", dtype=np.int32)
            np.testing.assert_array_equal(icc_vel, icc.ravel())
            ncoords_vel = np.fromfile(sim_path / f"ncoords_{suffix}.dat", dtype=np.int32)
            assert int(ncoords_vel) == 1


def _dc_map_original(c_map: np.ndarray) -> np.ndarray:
    """Original _set_dc_map logic before in-place optimization."""
    return matlab_round(c_map) - matlab_round(c_map.min()) + 1


def _dc_map_optimized(c_map: np.ndarray) -> np.ndarray:
    """Optimized _set_dc_map logic using in-place operations."""
    c_min_rounded = matlab_round(c_map.min())
    dc = np.array(c_map, dtype=np.float64)
    dc += 1e-9
    np.rint(dc, out=dc)
    dc -= c_min_rounded
    dc += 1
    return dc.astype(np.int64)


@pytest.mark.parametrize(
    "c_map",
    [
        # Uniform medium
        np.full((100, 80), 1540.0),
        # Integer values (no rounding ambiguity)
        np.arange(1400, 1600).reshape(40, 5).astype(np.float64),
        # Values near 0.5 boundaries to stress MATLAB-style rounding
        np.array([1499.5, 1500.0, 1500.5, 1501.4999, 1501.5001], dtype=np.float64),
        # Random realistic sound speeds (soft tissue range)
        np.random.default_rng(42).uniform(1400, 1600, size=(200, 150)),
        # Single element
        np.array([1540.0]),
        # Negative offset to check subtraction of min
        np.array([100.3, 200.7, 300.1, 400.9], dtype=np.float64),
    ],
    ids=[
        "uniform",
        "integer_range",
        "half_boundaries",
        "random_2d",
        "single_element",
        "negative_offset",
    ],
)
def test_dc_map_optimized_matches_original(c_map):
    """Verify that the in-place optimized _set_dc_map produces identical results."""
    original = _dc_map_original(c_map)
    optimized = _dc_map_optimized(c_map)
    np.testing.assert_array_equal(original, optimized)


def _isotropic_medium(mechanisms: int, *, constrained: bool = True):
    """Return a dummy medium carrying one isotropic relaxation map of each name.

    Parameters
    ----------
    mechanisms : int
        How many relaxation mechanisms the medium carries.
    constrained : bool
        Whether the impedance constraint holds on the maps.

    Returns
    -------
    SimpleNamespace
        The dummy medium.

    """
    shape = (4, 4)
    maps = {
        "kappa_x": np.ones(shape, dtype=np.float64),
        "kappa_u": np.full(shape, 1.5, dtype=np.float64),
    }
    for nu in range(1, mechanisms + 1):
        maps[f"a_pml_u{nu}"] = np.full(shape, 0.25, dtype=np.float64)
        maps[f"b_pml_u{nu}"] = np.full(shape, 0.75, dtype=np.float64)
        maps[f"a_pml_x{nu}"] = np.zeros(shape, dtype=np.float64)
        maps[f"b_pml_x{nu}"] = np.ones(shape, dtype=np.float64)
    maps["a_pml_x1"] = np.full(shape, 0.5, dtype=np.float64)
    if not constrained:
        maps["kappa_x"][0, 0] = 1.25
        if mechanisms > 1:
            maps["a_pml_x2"][1, 1] = 0.5

    _, medium, _, _ = create_dummy_objects()
    medium.relaxation_param_dict_for_fw2 = maps
    medium.n_relaxation_mechanisms = mechanisms
    return medium


def _run_isotropic_writer(
    work_and_bin,
    monkeypatch,
    medium,
    *,
    sim_dir_name: str,
    write_maps_the_solver_does_not_read: bool | None = None,
) -> Path:
    """Run the writer on the isotropic relaxation path and return its directory.

    Parameters
    ----------
    work_and_bin : tuple[Path, Path]
        The work directory and the fake binary.
    monkeypatch : pytest.MonkeyPatch
        The pytest monkeypatch fixture.
    medium : SimpleNamespace
        The dummy medium to write.
    sim_dir_name : str
        The name of the simulation directory.
    write_maps_the_solver_does_not_read : bool | None
        Whether to write the momentum maps the solver never reads. None leaves
        the writer on its own default.

    Returns
    -------
    Path
        The simulation directory.

    """
    work_dir, bin_file = work_and_bin
    grid, _, source, sensor = create_dummy_objects()
    monkeypatch.setattr(check_functions, "check_path_exists", lambda x: None)  # noqa: ARG005
    monkeypatch.setattr(check_functions, "check_instance", lambda inst, cls: None)  # noqa: ARG005

    optional = (
        {}
        if write_maps_the_solver_does_not_read is None
        else {"write_maps_the_solver_does_not_read": write_maps_the_solver_does_not_read}
    )
    writer = InputFileWriter(
        work_dir,
        grid,
        medium,
        source,
        sensor,
        path_fullwave_simulation_bin=bin_file,
        validate_input=False,
        use_isotropic_relaxation=True,
        **optional,
    )
    return Path(writer.run(sim_dir_name, is_static_map=False, recalculate_pml=True))


class TestTheMapsTheSolverDoesNotRead:
    """The writer states the impedance constraint and can skip seven maps."""

    def test_the_record_is_written_on_the_relaxation_path(self, work_and_bin, monkeypatch):
        sim_path = _run_isotropic_writer(
            work_and_bin,
            monkeypatch,
            _isotropic_medium(4),
            write_maps_the_solver_does_not_read=True,
            sim_dir_name="sim_record",
        )
        record = np.fromfile(sim_path / "impedance_constraint.dat", dtype=np.float32)
        assert record.size == 3
        assert int(record[0]) == 4
        assert record[1] == 0.0
        assert record[2] == 0.0

    def test_the_switch_writes_every_map(self, work_and_bin, monkeypatch):
        sim_path = _run_isotropic_writer(
            work_and_bin,
            monkeypatch,
            _isotropic_medium(4),
            write_maps_the_solver_does_not_read=True,
            sim_dir_name="sim_all",
        )
        for name in ("kappax", "apmlx2", "apmlx3", "apmlx4", "bpmlx2", "bpmlx3", "bpmlx4"):
            assert (sim_path / f"{name}.dat").exists(), name

    def test_the_default_skips_seven_maps_at_four_mechanisms(self, work_and_bin, monkeypatch):
        sim_path = _run_isotropic_writer(
            work_and_bin,
            monkeypatch,
            _isotropic_medium(4),
            sim_dir_name="sim_default",
        )
        for name in ("kappax", "apmlx2", "apmlx3", "apmlx4", "bpmlx2", "bpmlx3", "bpmlx4"):
            assert not (sim_path / f"{name}.dat").exists(), name
        assert (sim_path / "impedance_constraint.dat").exists()

    def test_the_switch_skips_seven_maps_at_four_mechanisms(self, work_and_bin, monkeypatch):
        sim_path = _run_isotropic_writer(
            work_and_bin,
            monkeypatch,
            _isotropic_medium(4),
            write_maps_the_solver_does_not_read=False,
            sim_dir_name="sim_skipped",
        )
        for name in ("kappax", "apmlx2", "apmlx3", "apmlx4", "bpmlx2", "bpmlx3", "bpmlx4"):
            assert not (sim_path / f"{name}.dat").exists(), name
        for name in ("kappau", "apmlx1", "bpmlx1", "apmlu1", "bpmlu1", "apmlu4", "bpmlu4"):
            assert (sim_path / f"{name}.dat").exists(), name
        assert (sim_path / "impedance_constraint.dat").exists()

    def test_the_record_states_a_medium_that_breaks_the_constraint(self, work_and_bin, monkeypatch):
        sim_path = _run_isotropic_writer(
            work_and_bin,
            monkeypatch,
            _isotropic_medium(2, constrained=False),
            write_maps_the_solver_does_not_read=True,
            sim_dir_name="sim_free",
        )
        record = np.fromfile(sim_path / "impedance_constraint.dat", dtype=np.float32)
        assert int(record[0]) == 2
        assert record[1] == pytest.approx(0.25)
        assert record[2] == pytest.approx(0.5)
