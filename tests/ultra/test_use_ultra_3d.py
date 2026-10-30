"""CPU tests of Solver(use_ultra=True) in 3D: run directory, refusals, binary lookup."""

import inspect
import warnings
from pathlib import Path

import numpy as np
import pytest

import fullwave
from fullwave.ultra import backend, binary
from fullwave.ultra.errors import (
    NO_BINARY_MESSAGE,
    FullwaveUltraExperimentalWarning,
    UltraBinaryError,
    UltraUnsupportedError,
)

SHAPE = (36, 28, 28)
N_TIC = 30


def make_grid(*, cfl=0.30):
    f0, c0, ppw = 1.0e6, 1540.0, 6
    dx = c0 / f0 / ppw
    return fullwave.Grid(
        domain_size=tuple(n * dx for n in SHAPE),
        f0=f0,
        duration=15e-6,
        c0=c0,
        ppw=ppw,
        cfl=cfl,
    )


def make_inputs(*, alpha_power=1.0, cfl=0.30):
    grid = make_grid(cfl=cfl)
    rng = np.random.default_rng(3)
    speed = 1540.0 + 40.0 * rng.random(SHAPE)
    density = 1000.0 + 30.0 * rng.random(SHAPE)
    alpha = 0.5 * rng.random(SHAPE)
    beta = np.zeros(SHAPE)
    medium = fullwave.Medium(
        grid=grid,
        sound_speed=speed,
        density=density,
        alpha_coeff=alpha,
        alpha_power=np.full(SHAPE, alpha_power),
        beta=beta,
    )
    cells = np.argwhere(np.zeros(SHAPE, dtype=bool) == 0)[::97][:40]
    drive = rng.standard_normal((len(cells), N_TIC)).astype(np.float32)
    source = fullwave.Source(p0=drive, coords=cells, grid_shape=SHAPE)
    points = np.stack([np.arange(5, 30), np.full(25, 14), np.full(25, 14)], axis=1)
    sensor = fullwave.Sensor(coords=points, grid_shape=SHAPE, sampling_modulus_time=4)
    return grid, medium, source, sensor


@pytest.fixture
def fake_binary(tmp_path):
    path = tmp_path / "bench_3d_opt"
    path.write_bytes(b"\x00junk nZ.dat junk ncoords_add.dat icmat_add.dat \x00")
    return path


def make_solver(tmp_path, binary_path, **kwargs):
    grid, medium, source, sensor = make_inputs()
    return fullwave.Solver(
        work_dir=tmp_path / "work",
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        path_fullwave_simulation_bin=binary_path,
        use_ultra=True,
        run_on_memory=False,
        exponential_attenuation_pml_thickness_px=24,
        **kwargs,
    )


def test_run_returns_traces_and_records_the_run(tmp_path, fake_binary, monkeypatch):
    solver = make_solver(tmp_path, fake_binary)

    def fake_run(binary_path, rundir, **_kwargs):
        np.arange(7 * 25, dtype=np.float32).tofile(Path(rundir) / "genout.dat")
        return "stdout"

    monkeypatch.setattr(binary, "run_binary", fake_run)
    traces = solver.run("run")
    assert traces.shape == (25, 7)
    assert traces[0, 1] == 25.0
    assert solver.ultra.last_run["stdout"] == "stdout"
    assert solver.ultra.last_run["writer"]["mext"] == 24 + 8
    assert solver.run("run", load_results=False) == tmp_path / "work" / "run" / "genout.dat"


AIR_CELLS = np.array([[4, 5, 6], [4, 5, 7]])


def make_air_solver(tmp_path, binary_path):
    grid, medium, source, sensor = make_inputs()
    medium.air_coords = AIR_CELLS
    return fullwave.Solver(
        work_dir=tmp_path / "work",
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        path_fullwave_simulation_bin=binary_path,
        use_ultra=True,
        run_on_memory=False,
        exponential_attenuation_pml_thickness_px=24,
    )


def test_air_cells_are_written_with_the_same_shift_as_the_sources(tmp_path):
    path = tmp_path / "bench_3d_opt"
    path.write_bytes(b"nZ.dat icczero.dat nzero0.dat")
    shift = 24 + 8
    rundir = make_air_solver(tmp_path, path).run("run", generate_input_only=True)
    shifted = AIR_CELLS + shift
    assert (
        np.fromfile(rundir / "icczero.dat", dtype=np.int32).tolist() == shifted.T.ravel().tolist()
    )
    assert np.fromfile(rundir / "ncoordszero.dat", dtype=np.int32).tolist() == [2]
    n_t = np.fromfile(rundir / "nT.dat", dtype=np.int32)
    assert np.fromfile(rundir / "nzero0.dat", dtype=np.int32).tolist() == [0]
    assert np.fromfile(rundir / "nzero1.dat", dtype=np.int32).tolist() == n_t.tolist()
    assert not (rundir / "nTzero.dat").exists()


def test_a_run_without_air_writes_no_air_files(tmp_path, fake_binary):
    rundir = make_solver(tmp_path, fake_binary).run("run", generate_input_only=True)
    names = {entry.name for entry in rundir.iterdir()}
    assert not names & {"icczero.dat", "ncoordszero.dat", "nzero0.dat", "nzero1.dat"}


def test_air_needs_a_binary_that_reads_the_air_cells(tmp_path, fake_binary):
    with pytest.raises(UltraUnsupportedError, match="icczero.dat"):
        make_air_solver(tmp_path, fake_binary)
    assert not (tmp_path / "work").exists()


def test_air_needs_the_window_file_too(tmp_path):
    path = tmp_path / "bench_3d_opt"
    path.write_bytes(b"nZ.dat icczero.dat nTzero.dat")
    with pytest.raises(UltraUnsupportedError, match="nzero0.dat"):
        make_air_solver(tmp_path, path)


def test_no_binary_error_text(tmp_path, monkeypatch):
    monkeypatch.delenv("FW2U_BATCH_BIN", raising=False)
    grid, medium, source, sensor = make_inputs()
    with pytest.raises(UltraBinaryError) as info:
        fullwave.Solver(
            work_dir=tmp_path,
            grid=grid,
            medium=medium,
            source=source,
            sensor=sensor,
            use_ultra=True,
        )
    text = str(info.value)
    assert text == NO_BINARY_MESSAGE
    assert "contact the Pinton lab" in text
    assert "https://github.com/pinton-lab/fullwave2-ultra" in text
    assert "does not ship the fullwave2-ultra binary" in text


def test_missing_binary_path_error(tmp_path):
    with pytest.raises(UltraBinaryError, match="does not exist"):
        binary.resolve_binary(tmp_path / "nope", name=binary.BINARY_NAME_3D)


def test_binary_from_environment_file_and_folder(tmp_path, fake_binary, monkeypatch):
    monkeypatch.setenv("FW2U_BATCH_BIN", str(fake_binary))
    assert binary.resolve_binary(None, name=binary.BINARY_NAME_3D) == fake_binary
    monkeypatch.setenv("FW2U_BATCH_BIN", str(tmp_path))
    assert binary.resolve_binary(None, name=binary.BINARY_NAME_3D) == fake_binary


def test_binary_inspection_on_marker_files(tmp_path):
    cases = {
        b"nZ.dat": (True, False, False, False, False, False),
        b"icczero.dat": (False,) * 6,
        b"icczero.dat nzero0.dat": (False, True, False, False, True, False),
        b"ncoords_add.dat": (False, False, True, False, False, False),
        b"icmat_add.dat": (False, False, False, True, False, False),
        b"nTzero.dat": (False, False, False, False, True, False),
        b"nszero0.dat": (False, False, False, False, False, True),
        b"nothing": (False,) * 6,
        b"nZ.dat icczero.dat ncoords_add.dat icmat_add.dat nTzero.dat nzero0.dat nszero0.dat": (
            True,
        )
        * 6,
    }
    for index, (content, expected) in enumerate(cases.items()):
        path = tmp_path / f"fake_{index}"
        path.write_bytes(b"\x01" * 100 + content + b"\x02" * 100)
        f = binary.inspect_binary(path)
        found = (
            f.is_3d,
            f.has_air_cells_3d,
            f.has_additive_channel,
            f.has_additive_signals,
            f.has_air_hold_2d,
            f.has_source_release_2d,
        )
        assert found == expected
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    assert not binary.inspect_binary(empty).is_3d


def test_binary_without_the_3d_marker_is_refused(tmp_path):
    path = tmp_path / "bench_2d_batch"
    path.write_bytes(b"nTzero.dat")
    with pytest.raises(UltraUnsupportedError, match="3D"):
        make_solver(tmp_path, path)


def test_experimental_warning_comes_once(tmp_path, fake_binary):
    backend._shown_warnings.clear()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        make_solver(tmp_path, fake_binary)
        make_solver(tmp_path, fake_binary)
    kinds = [w.category for w in caught if w.category is FullwaveUltraExperimentalWarning]
    assert len(kinds) == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"pml_layer_thickness_px": 10},
        {"n_transition_layer": 3},
        {"use_gpu_pml": True},
        {"pml_alpha_entrance": 0.5},
        {"use_pml": False},
        {"m_spatial_order": 4},
        {"use_gpu": False},
        {"cuda_device_id": [0, 1]},
        {"save_gpu_memory": True},
        {"write_maps_the_solver_does_not_read": True},
    ],
)
def test_unsupported_solver_options_are_refused(tmp_path, fake_binary, kwargs):
    with pytest.raises(UltraUnsupportedError):
        make_solver(tmp_path, fake_binary, **kwargs)


def build(tmp_path, fake_binary, *, grid, medium, source, sensor):
    return fullwave.Solver(
        work_dir=tmp_path,
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        path_fullwave_simulation_bin=fake_binary,
        use_ultra=True,
        run_on_memory=False,
    )


def test_alpha_power_other_than_one_is_refused(tmp_path, fake_binary):
    grid, medium, source, sensor = make_inputs(alpha_power=1.1)
    with pytest.raises(UltraUnsupportedError, match="alpha_power"):
        build(tmp_path, fake_binary, grid=grid, medium=medium, source=source, sensor=sensor)


def test_relaxation_medium_is_refused(tmp_path, fake_binary):
    grid, medium, source, sensor = make_inputs()
    medium.__class__ = type("MediumRelaxationMaps", (fullwave.Medium,), {})
    with pytest.raises(UltraUnsupportedError, match="Medium"):
        build(tmp_path, fake_binary, grid=grid, medium=medium, source=source, sensor=sensor)


def test_cfl_above_the_3d_limit_is_refused(tmp_path, fake_binary):
    grid, medium, source, sensor = make_inputs(cfl=0.35)
    with pytest.raises(UltraUnsupportedError, match="stability"):
        build(tmp_path, fake_binary, grid=grid, medium=medium, source=source, sensor=sensor)


def test_two_dimensional_grid_with_a_3d_binary_is_refused(tmp_path, fake_binary):
    grid = fullwave.Grid(domain_size=(0.01, 0.01), f0=1e6, duration=5e-6, ppw=6, cfl=0.3)
    shape = grid.shape
    medium = fullwave.Medium(
        grid=grid,
        sound_speed=np.full(shape, 1540.0),
        density=np.full(shape, 1000.0),
        alpha_coeff=np.zeros(shape),
        alpha_power=np.ones(shape),
        beta=np.zeros(shape),
    )
    source = fullwave.Source(p0=np.ones((1, 5)), coords=np.array([[3, 3]]), grid_shape=shape)
    sensor = fullwave.Sensor(coords=np.array([[5, 5]]), grid_shape=shape)
    with pytest.raises(UltraUnsupportedError, match="3D"):
        build(tmp_path, fake_binary, grid=grid, medium=medium, source=source, sensor=sensor)


def test_air_velocity_and_sparse_inputs_are_refused(tmp_path, fake_binary):
    grid, medium, source, sensor = make_inputs()
    medium.air_coords = np.array([[1, 1, 1]])
    with pytest.raises(UltraUnsupportedError, match="air"):
        build(tmp_path, fake_binary, grid=grid, medium=medium, source=source, sensor=sensor)

    grid, medium, source, sensor = make_inputs()
    source.u0 = np.ones((1, N_TIC))
    with pytest.raises(UltraUnsupportedError, match="velocity"):
        build(tmp_path, fake_binary, grid=grid, medium=medium, source=source, sensor=sensor)

    grid, medium, source, _ = make_inputs()
    sparse = fullwave.Sensor(mod_x=2, mod_y=2, mod_z=2)
    with pytest.raises(UltraUnsupportedError, match="sparse"):
        build(tmp_path, fake_binary, grid=grid, medium=medium, source=source, sensor=sparse)


def test_run_options_that_the_backend_cannot_honor_are_refused(tmp_path, fake_binary):
    solver = make_solver(tmp_path, fake_binary)
    with pytest.raises(UltraUnsupportedError):
        solver.run("run", record_whole_domain=True)
    with pytest.raises(UltraUnsupportedError):
        solver.run("run", is_static_map=True)
    with pytest.raises(UltraUnsupportedError, match="run_batch"):
        solver.run_batch([solver.source])


def test_the_native_default_is_unchanged():
    parameter = inspect.signature(fullwave.Solver.__init__).parameters["use_ultra"]
    assert parameter.default is False
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY


ADDITIVE_FILES = ("icc_add.dat", "nTic_add.dat", "icmat_add.dat")
ADDITIVE_CELLS = np.array([[7, 8, 9], [7, 8, 10], [7, 9, 9]])
N_TIC_ADD = N_TIC - 4


def read_int(rundir, name):
    return np.fromfile(rundir / name, dtype=np.int32)


def additive_signal():
    return np.arange(len(ADDITIVE_CELLS) * N_TIC_ADD, dtype=float).reshape(-1, N_TIC_ADD)


def make_additive_solver(tmp_path, binary_path, *, hard_too, source_type="hard_released_after"):
    grid, medium, hard_source, sensor = make_inputs()
    source = fullwave.Source(
        p0=hard_source.p0 if hard_too else None,
        coords=hard_source.incoords if hard_too else None,
        p0_additive=additive_signal(),
        coords_additive=ADDITIVE_CELLS,
        grid_shape=SHAPE,
    )
    return fullwave.Solver(
        work_dir=tmp_path / "work",
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        path_fullwave_simulation_bin=binary_path,
        use_ultra=True,
        run_on_memory=False,
        exponential_attenuation_pml_thickness_px=24,
        source_type=source_type,
    )


def make_soft_solver(tmp_path, binary_path):
    grid, medium, _, sensor = make_inputs()
    plane = np.argwhere(np.ones(SHAPE[1:], dtype=bool))
    cells = np.column_stack([np.full(len(plane), 4), plane])
    drive = np.random.default_rng(4).standard_normal((len(cells), N_TIC))
    source = fullwave.Source(p0=drive, coords=cells, grid_shape=SHAPE)
    solver = fullwave.Solver(
        work_dir=tmp_path / "work",
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        path_fullwave_simulation_bin=binary_path,
        use_ultra=True,
        run_on_memory=False,
        exponential_attenuation_pml_thickness_px=24,
        source_type="soft",
    )
    return solver, grid, medium, cells, drive


def test_a_run_without_additive_sources_writes_no_additive_file(tmp_path, fake_binary):
    rundir = make_solver(tmp_path, fake_binary).run("run", generate_input_only=True)
    assert not any((rundir / name).exists() for name in ADDITIVE_FILES)
    assert read_int(rundir, "ncoords_add.dat").tolist() == [0]


def test_an_additive_source_is_written_next_to_the_hard_source(tmp_path, fake_binary):
    solver = make_additive_solver(tmp_path, fake_binary, hard_too=True)
    rundir = solver.run("run", generate_input_only=True)
    shifted = ADDITIVE_CELLS + 24 + 8
    assert read_int(rundir, "ncoords_add.dat").tolist() == [len(ADDITIVE_CELLS)]
    assert read_int(rundir, "nTic_add.dat").tolist() == [N_TIC_ADD + 1]
    assert read_int(rundir, "nTic.dat").tolist() == [N_TIC]
    assert read_int(rundir, "icc_add.dat").tolist() == shifted.T.ravel().tolist()
    assert read_int(rundir, "ncoords.dat").tolist() == [40]
    assert (rundir / "icc.dat").exists()


def test_the_additive_signal_is_written_one_time_step_after_another(tmp_path, fake_binary):
    solver = make_additive_solver(tmp_path, fake_binary, hard_too=True)
    rundir = solver.run("run", generate_input_only=True)
    written = np.fromfile(rundir / "icmat_add.dat", dtype=np.float32)
    delayed = np.pad(additive_signal().T, ((1, 0), (0, 0)))
    assert np.array_equal(written, delayed.astype(np.float32).ravel())


def test_a_purely_additive_source_writes_no_hard_source_files(tmp_path, fake_binary):
    solver = make_additive_solver(tmp_path, fake_binary, hard_too=False)
    rundir = solver.run("run", generate_input_only=True)
    assert read_int(rundir, "ncoords.dat").tolist() == [0]
    assert read_int(rundir, "nTic.dat").tolist() == [N_TIC_ADD]
    assert not (rundir / "icc.dat").exists()
    assert not (rundir / "icmat.dat").exists()
    assert (rundir / "icmat_add.dat").exists()


def test_a_soft_source_is_written_as_a_purely_additive_scaled_signal(tmp_path, fake_binary):
    solver, grid, medium, cells, drive = make_soft_solver(tmp_path, fake_binary)
    rundir = solver.run("run", generate_input_only=True)
    assert read_int(rundir, "ncoords.dat").tolist() == [0]
    assert read_int(rundir, "ncoords_add.dat").tolist() == [len(cells)]
    assert not (rundir / "icc.dat").exists()
    scale = 2 * np.asarray(medium.sound_speed)[tuple(cells.T)] * grid.dt / grid.dx
    scaled = (drive * scale[:, None]).T
    expected = np.pad(scaled, ((1, 0), (0, 0))).astype(np.float32).ravel()
    written = np.fromfile(rundir / "icmat_add.dat", dtype=np.float32)
    assert np.allclose(written, expected, rtol=1e-6)


def test_a_later_run_without_additive_sources_resets_the_additive_count(tmp_path, fake_binary):
    rundir = make_additive_solver(tmp_path, fake_binary, hard_too=True).run(
        "run", generate_input_only=True
    )
    make_solver(tmp_path, fake_binary).run("run", generate_input_only=True)
    assert read_int(rundir, "ncoords_add.dat").tolist() == [0]


def test_a_later_run_with_a_hard_source_restores_the_hard_source_files(tmp_path, fake_binary):
    make_additive_solver(tmp_path, fake_binary, hard_too=False).run("run", generate_input_only=True)
    rundir = make_solver(tmp_path, fake_binary).run("run", generate_input_only=True)
    assert (rundir / "icc.dat").exists()
    assert (rundir / "icmat.dat").exists()


def test_additive_cells_that_repeat_are_an_error(tmp_path, fake_binary):
    solver = make_additive_solver(tmp_path, fake_binary, hard_too=True)
    solver.ultra.source.incoords_add = np.vstack([ADDITIVE_CELLS, ADDITIVE_CELLS[:1]])
    with pytest.raises(ValueError, match="different"):
        solver.run("run", generate_input_only=True)


def test_an_additive_signal_longer_than_the_simulation_is_an_error(tmp_path, fake_binary):
    solver = make_additive_solver(tmp_path, fake_binary, hard_too=True)
    n_t = int(np.fromfile(solver.run("run", generate_input_only=True) / "nT.dat", np.int32)[0])
    solver.ultra.source.p0_additive = np.zeros((len(ADDITIVE_CELLS), n_t + 1))
    with pytest.raises(ValueError, match="more than"):
        solver.run("run", generate_input_only=True)


@pytest.mark.parametrize(
    "content",
    [b"nZ.dat", b"nZ.dat ncoords_add.dat", b"nZ.dat icmat_add.dat"],
)
def test_an_additive_source_needs_a_binary_that_reads_the_channel(tmp_path, content):
    path = tmp_path / "bench_3d_opt"
    path.write_bytes(content)
    with pytest.raises(UltraUnsupportedError, match="additive"):
        make_additive_solver(tmp_path, path, hard_too=True)
    with pytest.raises(UltraUnsupportedError, match="additive"):
        make_soft_solver(tmp_path, path)
    assert not (tmp_path / "work").exists()


def test_a_hard_source_does_not_need_the_additive_channel(tmp_path):
    path = tmp_path / "bench_3d_opt"
    path.write_bytes(b"nZ.dat")
    make_solver(tmp_path, path)


def test_a_soft_solver_refuses_a_source_that_is_already_additive(tmp_path, fake_binary):
    with pytest.raises(ValueError, match="already carries an additive signal"):
        make_additive_solver(tmp_path, fake_binary, hard_too=False, source_type="soft")


@pytest.fixture
def hold_binary(tmp_path):
    path = tmp_path / "bench_3d_opt"
    path.write_bytes(b"\x00junk nZ.dat nszero0.dat \x00")
    return path


def test_the_hard_source_type_writes_the_drive_as_given_and_the_hold_window(tmp_path, hold_binary):
    rundir = make_solver(tmp_path, hold_binary, source_type="hard").run(
        "run", generate_input_only=True
    )
    n_t = np.fromfile(rundir / "nT.dat", dtype=np.int32).tolist()
    assert np.fromfile(rundir / "nTic.dat", dtype=np.int32).tolist() == [N_TIC]
    assert np.fromfile(rundir / "nszero0.dat", dtype=np.int32).tolist() == [N_TIC]
    assert np.fromfile(rundir / "nszero1.dat", dtype=np.int32).tolist() == n_t
    assert not (rundir / "hold_source.dat").exists()


def test_hard_released_after_writes_no_hold_window(tmp_path, hold_binary):
    rundir = make_solver(tmp_path, hold_binary, source_type="hard_released_after").run(
        "run", generate_input_only=True
    )
    assert not (rundir / "nszero0.dat").exists()
    assert not (rundir / "nszero1.dat").exists()


def test_the_hard_source_type_refuses_a_binary_that_does_not_read_the_window(tmp_path, fake_binary):
    with pytest.raises(UltraUnsupportedError, match="nszero0.dat"):
        make_solver(tmp_path, fake_binary, source_type="hard")
    assert not (tmp_path / "work").exists()
