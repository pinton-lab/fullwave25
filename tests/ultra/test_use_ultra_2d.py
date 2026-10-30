"""CPU tests of Solver(use_ultra=True) in 2D: run directories, batches, refusals."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

import fullwave
from fullwave.ultra import binary
from fullwave.ultra.errors import UltraBinaryError, UltraUnsupportedError

SHAPE = (40, 32)
N_TIC = 30
N_MEMBERS = 4
SPONGE = 24
MOD_T = 2


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


def make_inputs(*, alpha_power=1.0, cfl=0.30, with_air=False, whole_row=False):
    grid = make_grid(cfl=cfl)
    rng = np.random.default_rng(3)
    air = np.array([[2, 3], [2, 4]]) if with_air else None
    medium = fullwave.Medium(
        grid=grid,
        sound_speed=1540.0 + 40.0 * rng.random(SHAPE),
        density=1000.0 + 30.0 * rng.random(SHAPE),
        alpha_coeff=0.5 * rng.random(SHAPE),
        alpha_power=np.full(SHAPE, alpha_power),
        beta=np.zeros(SHAPE),
        air_coords=air,
    )
    columns = np.arange(SHAPE[1]) if whole_row else np.arange(8, 20)
    cells = np.stack([np.full(len(columns), 4), columns], axis=1)
    drives = rng.standard_normal((N_MEMBERS, len(cells), N_TIC)).astype(np.float32)
    sources = [fullwave.Source(p0=d, coords=cells, grid_shape=SHAPE) for d in drives]
    points = np.stack([np.arange(5, 30), np.full(25, 14)], axis=1)
    sensor = fullwave.Sensor(coords=points, grid_shape=SHAPE, sampling_modulus_time=MOD_T)
    return grid, medium, sources, sensor


@pytest.fixture
def fake_binary(tmp_path):
    path = tmp_path / "bench_2d_batch"
    path.write_bytes(b"\x00junk nTzero.dat nszero0.dat soft_source.dat ncoords_add.dat \x00")
    return path


def digests(directory):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(directory).iterdir())
    }


def make_solver(tmp_path, binary_path, *, inputs=None, **kwargs):
    grid, medium, sources, sensor = inputs or make_inputs(
        whole_row=kwargs.get("source_type") == "soft"
    )
    return fullwave.Solver(
        work_dir=tmp_path / "work",
        grid=grid,
        medium=medium,
        source=sources[0],
        sensor=sensor,
        path_fullwave_simulation_bin=binary_path,
        use_ultra=True,
        run_on_memory=False,
        exponential_attenuation_pml_thickness_px=SPONGE,
        **kwargs,
    )


def read_int(rundir, name):
    return np.fromfile(Path(rundir) / Path(name).with_suffix(".dat"), dtype=np.int32)


def written_run(tmp_path, binary_path, *, with_air=False, **kwargs):
    inputs = make_inputs(with_air=with_air)
    solver = make_solver(tmp_path, binary_path, inputs=inputs, **kwargs)
    (rundir,) = solver.run_batch(inputs[2], "run", generate_input_only=True)
    return Path(rundir), inputs, solver


def test_run_directory_holds_the_inputs_of_the_binary(tmp_path, fake_binary):
    rundir, (grid, _, sources, sensor), solver = written_run(tmp_path, fake_binary)
    mext = SPONGE + 8
    n_x, n_y = (n + 2 * mext for n in SHAPE)
    assert read_int(rundir, "nX").tolist() == [n_x]
    assert read_int(rundir, "nY").tolist() == [n_y]
    assert read_int(rundir, "nT").tolist() == [grid.nt]
    assert read_int(rundir, "nTic").tolist() == [N_TIC]
    assert read_int(rundir, "modT").tolist() == [MOD_T]
    assert read_int(rundir, "nsims").tolist() == [N_MEMBERS]
    assert read_int(rundir, "ncoordszero").tolist() == [0]
    assert not (rundir / "icczero.dat").exists()
    for name in ("c", "rho", "K", "beta", "Aexp"):
        assert (rundir / f"{name}.dat").stat().st_size == n_x * n_y * 4
    cells = sources[0].incoords + mext
    assert read_int(rundir, "icc.dat").tolist() == cells[:, 0].tolist() + cells[:, 1].tolist()
    points = sensor.outcoords + mext
    assert read_int(rundir, "outc.dat").tolist() == points[:, 0].tolist() + points[:, 1].tolist()
    icmat = np.fromfile(rundir / "icmat.dat", dtype=np.float32)
    assert icmat.size == N_MEMBERS * len(cells) * N_TIC
    expected = np.stack([s.p0 for s in sources]).astype(np.float32).ravel()
    assert np.array_equal(icmat, expected)
    assert solver.ultra.last_run["writer"]["nTzero"] == grid.nt


def test_air_cells_are_written_with_the_same_shift_as_the_sources(tmp_path, fake_binary):
    rundir, (_, medium_air, _, _), _ = written_run(tmp_path, fake_binary, with_air=True)
    air = np.array([[2, 3], [2, 4]]) + SPONGE + 8
    assert read_int(rundir, "ncoordszero").tolist() == [2]
    assert read_int(rundir, "icczero.dat").tolist() == air[:, 0].tolist() + air[:, 1].tolist()


def test_air_cells_are_held_for_the_whole_run_in_the_window_files(tmp_path, fake_binary):
    rundir, (grid, *_), _ = written_run(tmp_path, fake_binary, with_air=True)
    assert read_int(rundir, "nzero0").tolist() == [0]
    assert read_int(rundir, "nzero1").tolist() == [grid.nt]
    assert read_int(rundir, "nTzero").tolist() == [grid.nt]


def test_no_air_cells_write_no_window_files(tmp_path, fake_binary):
    rundir, *_ = written_run(tmp_path, fake_binary)
    assert not (rundir / "nzero0.dat").exists()
    assert not (rundir / "nzero1.dat").exists()


@pytest.mark.parametrize("marker", [b"nTzero.dat", b"nzero0.dat"])
def test_air_runs_on_a_binary_that_reads_either_air_hold_file(tmp_path, marker):
    path = tmp_path / "bench_2d_batch"
    path.write_bytes(marker + b" nszero0.dat")
    make_solver(tmp_path, path, inputs=make_inputs(with_air=True))


def test_the_soft_source_flag_is_zero_and_the_release_step_is_the_last_step(tmp_path, fake_binary):
    rundir, _, solver = written_run(tmp_path, fake_binary)
    assert read_int(rundir, "soft_source").tolist() == [0]
    n_steps = solver.ultra.last_run["writer"]["nT"]
    assert read_int(rundir, "nszero0").tolist() == [n_steps]


def test_a_source_padded_to_the_grid_length_is_written_with_that_many_drive_steps(
    tmp_path, fake_binary
):
    grid, medium, sources, sensor = make_inputs()
    padded = [
        fullwave.Source(
            p0=np.pad(s.p0, ((0, 0), (0, grid.nt - N_TIC))), coords=s.incoords, grid_shape=SHAPE
        )
        for s in sources
    ]
    inputs = (grid, medium, padded, sensor)
    solver = make_solver(tmp_path, fake_binary, inputs=inputs)
    (rundir,) = solver.run_batch(padded, "run", generate_input_only=True)
    assert read_int(rundir, "nTic").tolist() == [grid.nt]
    assert read_int(rundir, "nszero0").tolist() == [grid.nt]


def test_the_hard_source_type_writes_the_drive_as_given_and_holds_it_by_the_window(
    tmp_path, fake_binary
):
    rundir, (grid, *_), _ = written_run(tmp_path, fake_binary, source_type="hard")
    assert read_int(rundir, "nTic").tolist() == [N_TIC]
    assert read_int(rundir, "nszero0").tolist() == [N_TIC]
    assert read_int(rundir, "nszero1").tolist() == [grid.nt]


def test_the_hard_released_after_source_type_writes_the_drive_as_given(tmp_path, fake_binary):
    rundir, *_ = written_run(tmp_path, fake_binary, source_type="hard_released_after")
    assert read_int(rundir, "nTic").tolist() == [N_TIC]
    assert not (rundir / "nszero1.dat").exists()


def test_hard_differs_from_hard_released_after_only_by_the_window(tmp_path, fake_binary):
    held, *_ = written_run(tmp_path / "held", fake_binary, source_type="hard")
    released, *_ = written_run(
        tmp_path / "released", fake_binary, source_type="hard_released_after"
    )
    held_digests, released_digests = digests(held), digests(released)
    differing = {
        name
        for name in held_digests.keys() | released_digests.keys()
        if held_digests.get(name) != released_digests.get(name)
    }
    assert differing == {"nszero0.dat", "nszero1.dat"}


def test_hard_refuses_a_binary_that_does_not_read_the_window(tmp_path):
    path = tmp_path / "bench_2d_batch"
    path.write_bytes(b"nTzero.dat soft_source.dat ncoords_add.dat")
    with pytest.raises(UltraUnsupportedError, match="nszero0.dat"):
        make_solver(tmp_path, path, source_type="hard")


def test_the_run_directory_is_deterministic(tmp_path, fake_binary):
    first, *_ = written_run(tmp_path / "a", fake_binary, with_air=True)
    second, *_ = written_run(tmp_path / "b", fake_binary, with_air=True)
    assert digests(first) == digests(second)


def fake_genout(rundir, n_members, n_sensors, n_frames):
    for member in range(n_members):
        name = "genout.dat" if member == 0 else f"genout_s{member}.dat"
        values = member * 1000 + np.arange(n_frames * n_sensors, dtype=np.float32)
        values.tofile(Path(rundir) / name)


def test_run_batch_splits_into_runs_and_returns_traces_in_order(tmp_path, fake_binary, monkeypatch):
    inputs = make_inputs()
    solver = make_solver(tmp_path, fake_binary, inputs=inputs)
    launched = []

    def fake_run(binary_path, rundir, **_kwargs):
        n_members = int(np.fromfile(Path(rundir) / "nsims.dat", dtype=np.int32)[0])
        launched.append((Path(rundir).name, n_members))
        fake_genout(rundir, n_members, n_sensors=25, n_frames=7)
        return "stdout"

    monkeypatch.setattr(binary, "run_binary", fake_run)
    traces = solver.run_batch(inputs[2], "batch", max_members_per_run=2)
    assert launched == [("batch_0", 2), ("batch_1", 2)]
    assert traces.shape == (N_MEMBERS, 25, 7)
    assert traces.dtype == np.float32
    assert traces[0, 1, 0] == 1.0
    assert traces[0, 0, 1] == 25.0
    assert traces[1, 0, 1] == 1000.0 + 25.0
    assert traces[2, 0, 0] == 0.0


def test_an_empty_output_file_with_exit_code_0_raises_with_the_binary_output(
    tmp_path, fake_binary, monkeypatch
):
    inputs = make_inputs()
    solver = make_solver(tmp_path, fake_binary, inputs=inputs)

    def fake_run(binary_path, rundir, **_kwargs):
        (Path(rundir) / "genout.dat").touch()
        return "genbuf alloc failed (out of memory); aborting"

    monkeypatch.setattr(binary, "run_binary", fake_run)
    with pytest.raises(UltraBinaryError, match="genout_s1.dat") as error:
        solver.run_batch(inputs[2], "batch")
    assert "out of memory" in str(error.value)
    assert "genout.dat (0 bytes)" in str(error.value)


def test_output_files_of_different_sizes_raise(tmp_path, fake_binary, monkeypatch):
    inputs = make_inputs()
    solver = make_solver(tmp_path, fake_binary, inputs=inputs)

    def fake_run(binary_path, rundir, **_kwargs):
        fake_genout(rundir, N_MEMBERS, n_sensors=25, n_frames=7)
        np.zeros(25 * 3, dtype=np.float32).tofile(Path(rundir) / "genout_s2.dat")
        return ""

    monkeypatch.setattr(binary, "run_binary", fake_run)
    with pytest.raises(UltraBinaryError, match=r"genout_s2.dat \(300 bytes\)"):
        solver.run_batch(inputs[2], "batch")


def test_run_is_a_batch_of_one(tmp_path, fake_binary, monkeypatch):
    solver = make_solver(tmp_path, fake_binary)

    def fake_run(binary_path, rundir, **_kwargs):
        fake_genout(rundir, 1, n_sensors=25, n_frames=7)
        return "stdout"

    monkeypatch.setattr(binary, "run_binary", fake_run)
    traces = solver.run("run")
    assert traces.shape == (25, 7)
    assert solver.ultra.last_run["writer"]["mext"] == SPONGE + 8
    assert solver.run("run", load_results=False) == tmp_path / "work" / "run" / "genout.dat"


def test_run_on_memory_removes_each_run_directory_after_reading(tmp_path, fake_binary, monkeypatch):
    inputs = make_inputs()
    solver = fullwave.Solver(
        work_dir=tmp_path / "work",
        grid=inputs[0],
        medium=inputs[1],
        source=inputs[2][0],
        sensor=inputs[3],
        path_fullwave_simulation_bin=fake_binary,
        use_ultra=True,
        run_on_memory=True,
    )

    def fake_run(binary_path, rundir, **_kwargs):
        fake_genout(rundir, N_MEMBERS, n_sensors=25, n_frames=7)
        return ""

    monkeypatch.setattr(binary, "run_binary", fake_run)
    solver.run_batch(inputs[2], "batch")
    assert list(solver.ultra.work_dir.iterdir()) == []
    assert not (tmp_path / "work").exists()


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
        {"source_type": "free"},
        {"save_gpu_memory": True},
        {"write_maps_the_solver_does_not_read": True},
    ],
)
def test_unsupported_solver_options_are_refused(tmp_path, fake_binary, kwargs):
    with pytest.raises(UltraUnsupportedError):
        make_solver(tmp_path, fake_binary, **kwargs)


def test_alpha_power_other_than_one_is_refused(tmp_path, fake_binary):
    inputs = make_inputs(alpha_power=1.1)
    with pytest.raises(UltraUnsupportedError, match="alpha_power"):
        make_solver(tmp_path, fake_binary, inputs=inputs)


def test_relaxation_medium_is_refused(tmp_path, fake_binary):
    inputs = make_inputs()
    inputs[1].__class__ = type("MediumRelaxationMaps", (fullwave.Medium,), {})
    with pytest.raises(UltraUnsupportedError, match="Medium"):
        make_solver(tmp_path, fake_binary, inputs=inputs)


def test_velocity_and_sparse_inputs_are_refused(tmp_path, fake_binary):
    inputs = make_inputs()
    inputs[2][0].u0 = np.ones((12, N_TIC))
    with pytest.raises(UltraUnsupportedError, match="velocity"):
        make_solver(tmp_path, fake_binary, inputs=inputs)

    grid, medium, sources, _ = make_inputs()
    sparse = fullwave.Sensor(mod_x=2, mod_y=2)
    with pytest.raises(UltraUnsupportedError, match="sparse"):
        make_solver(tmp_path, fake_binary, inputs=(grid, medium, sources, sparse))


def test_a_batch_member_with_a_velocity_source_is_refused(tmp_path, fake_binary):
    inputs = make_inputs()
    solver = make_solver(tmp_path, fake_binary, inputs=inputs)
    inputs[2][1].u0 = np.ones((12, N_TIC))
    with pytest.raises(UltraUnsupportedError, match="velocity"):
        solver.run_batch(inputs[2], generate_input_only=True)


def test_a_batch_member_that_differs_in_cells_or_length_is_an_error(tmp_path, fake_binary):
    inputs = make_inputs()
    solver = make_solver(tmp_path, fake_binary, inputs=inputs)
    shifted = fullwave.Source(
        p0=inputs[2][1].p0, coords=inputs[2][1].incoords + [1, 0], grid_shape=SHAPE
    )
    with pytest.raises(ValueError, match="source 1"):
        solver.run_batch([inputs[2][0], shifted], generate_input_only=True)
    shorter = fullwave.Source(
        p0=inputs[2][1].p0[:, :-1], coords=inputs[2][1].incoords, grid_shape=SHAPE
    )
    with pytest.raises(ValueError, match="source 1"):
        solver.run_batch([inputs[2][0], shorter], generate_input_only=True)


def test_empty_batch_and_bad_run_limit_are_errors(tmp_path, fake_binary):
    inputs = make_inputs()
    solver = make_solver(tmp_path, fake_binary, inputs=inputs)
    with pytest.raises(ValueError, match="at least one"):
        solver.run_batch([])
    with pytest.raises(ValueError, match="max_members_per_run"):
        solver.run_batch(inputs[2], max_members_per_run=0)


def test_run_options_that_the_backend_cannot_honor_are_refused(tmp_path, fake_binary):
    solver = make_solver(tmp_path, fake_binary)
    with pytest.raises(UltraUnsupportedError):
        solver.run("run", record_whole_domain=True)
    with pytest.raises(UltraUnsupportedError):
        solver.run("run", is_static_map=True)
    with pytest.raises(UltraUnsupportedError):
        solver.run_batch([solver.source], is_static_map=True)


def test_air_needs_a_binary_that_reads_the_air_hold(tmp_path):
    path = tmp_path / "bench_2d_batch"
    path.write_bytes(b"nszero0.dat")
    with pytest.raises(UltraUnsupportedError, match="air"):
        make_solver(tmp_path, path, inputs=make_inputs(with_air=True))
    make_solver(tmp_path, path)


def test_a_short_source_needs_a_binary_that_reads_the_release(tmp_path):
    path = tmp_path / "bench_2d_batch"
    path.write_bytes(b"nTzero.dat")
    with pytest.raises(UltraUnsupportedError, match="release"):
        make_solver(tmp_path, path)


def test_a_source_as_long_as_the_grid_runs_on_a_binary_without_the_release(tmp_path):
    path = tmp_path / "bench_2d_batch"
    path.write_bytes(b"nTzero.dat")
    grid, medium, sources, sensor = make_inputs()
    source = sources[0]
    padded = fullwave.Source(
        p0=np.pad(source.p0, ((0, 0), (0, grid.nt - N_TIC))),
        coords=source.incoords,
        grid_shape=SHAPE,
    )
    make_solver(tmp_path, path, inputs=(grid, medium, [padded], sensor))


ADDITIVE_FILES = ("icc_add.dat", "ncoords_add.dat", "nTic_add.dat", "icmat_add.dat")


def additive_inputs(*, hard_too):
    """Return inputs whose sources add to the pressure of a row below the hard row."""
    grid, medium, sources, sensor = make_inputs()
    rng = np.random.default_rng(5)
    cells = np.stack([np.full(10, 7), np.arange(8, 18)], axis=1)
    signals = rng.standard_normal((N_MEMBERS, len(cells), N_TIC - 4))
    additive = [
        fullwave.Source(
            p0=source.p0 if hard_too else None,
            coords=source.incoords if hard_too else None,
            p0_additive=signal,
            coords_additive=cells,
            grid_shape=SHAPE,
        )
        for source, signal in zip(sources, signals, strict=True)
    ]
    return (grid, medium, additive, sensor), cells, signals


def test_a_run_without_additive_sources_writes_no_additive_file(tmp_path, fake_binary):
    rundir, *_ = written_run(tmp_path, fake_binary)
    assert not any((rundir / name).exists() for name in ADDITIVE_FILES)


def test_an_additive_source_is_written_next_to_the_hard_source(tmp_path, fake_binary):
    inputs, cells, signals = additive_inputs(hard_too=True)
    solver = make_solver(tmp_path, fake_binary, inputs=inputs)
    (rundir,) = solver.run_batch(inputs[2], "run", generate_input_only=True)
    rundir = Path(rundir)
    shifted = cells + SPONGE + 8
    assert read_int(rundir, "ncoords_add").tolist() == [len(cells)]
    assert read_int(rundir, "nTic_add").tolist() == [N_TIC - 4 + 1]
    assert read_int(rundir, "nTic").tolist() == [N_TIC]
    assert read_int(rundir, "ncoords").tolist() == [12]
    expected_cells = shifted[:, 0].tolist() + shifted[:, 1].tolist()
    assert read_int(rundir, "icc_add.dat").tolist() == expected_cells
    written = np.fromfile(rundir / "icmat_add.dat", dtype=np.float32)
    delayed = np.pad(signals, ((0, 0), (0, 0), (1, 0)))
    assert np.array_equal(written, delayed.astype(np.float32).ravel())
    assert read_int(rundir, "soft_source").tolist() == [0]


def test_a_soft_source_is_written_as_a_purely_additive_scaled_signal(tmp_path, fake_binary):
    inputs = make_inputs(whole_row=True)
    solver = make_solver(tmp_path, fake_binary, inputs=inputs, source_type="soft")
    (rundir,) = solver.run_batch(inputs[2], "run", generate_input_only=True)
    rundir = Path(rundir)
    grid, medium, sources, _ = inputs
    cells = sources[0].incoords + SPONGE + 8
    assert read_int(rundir, "ncoords").tolist() == [0]
    assert read_int(rundir, "ncoords_add").tolist() == [len(cells)]
    assert read_int(rundir, "soft_source").tolist() == [0]
    assert (rundir / "icmat.dat").stat().st_size == 0
    assert read_int(rundir, "icc_add.dat").tolist() == cells[:, 0].tolist() + cells[:, 1].tolist()
    scale = 2 * np.asarray(medium.sound_speed)[tuple(sources[0].incoords.T)] * grid.dt / grid.dx
    scaled = np.stack([s.p0 * scale[:, None] for s in sources])
    expected = np.pad(scaled, ((0, 0), (0, 0), (1, 0))).astype(np.float32)
    written = np.fromfile(rundir / "icmat_add.dat", dtype=np.float32)
    assert np.allclose(written, expected.ravel(), rtol=1e-6)


def test_run_of_a_soft_solver_writes_the_additive_source_once(tmp_path, fake_binary):
    solver = make_solver(tmp_path, fake_binary, source_type="soft")
    rundir = solver.run("run", generate_input_only=True)
    sources = make_inputs(whole_row=True)[2]
    assert read_int(rundir, "nsims").tolist() == [1]
    assert read_int(rundir, "ncoords_add").tolist() == [len(sources[0].incoords)]


def test_a_reused_run_directory_keeps_no_stale_additive_cells(tmp_path, fake_binary):
    inputs = make_inputs(whole_row=True)
    solver = make_solver(tmp_path, fake_binary, inputs=inputs, source_type="soft")
    (rundir,) = solver.run_batch(inputs[2], "run", generate_input_only=True)
    hard = make_solver(tmp_path, fake_binary, inputs=inputs)
    hard.run_batch(inputs[2], "run", generate_input_only=True)
    assert read_int(Path(rundir), "ncoords_add").tolist() == [0]


def test_a_batch_member_that_differs_in_its_additive_cells_is_an_error(tmp_path, fake_binary):
    inputs, cells, signals = additive_inputs(hard_too=False)
    solver = make_solver(tmp_path, fake_binary, inputs=inputs)
    shifted = fullwave.Source(
        p0_additive=signals[1], coords_additive=cells + [1, 0], grid_shape=SHAPE
    )
    with pytest.raises(ValueError, match="source 1"):
        solver.run_batch([inputs[2][0], shifted], generate_input_only=True)
    with pytest.raises(ValueError, match="source 1"):
        solver.run_batch([inputs[2][0], make_inputs()[2][0]], generate_input_only=True)


def test_an_additive_source_needs_a_binary_with_the_additive_channel(tmp_path):
    path = tmp_path / "bench_2d_batch"
    path.write_bytes(b"nTzero.dat nszero0.dat")
    with pytest.raises(UltraUnsupportedError, match="additive"):
        make_solver(tmp_path, path, source_type="soft")
    make_solver(tmp_path, path)


def test_a_purely_additive_source_does_not_need_the_release_marker(tmp_path):
    path = tmp_path / "bench_2d_batch"
    path.write_bytes(b"ncoords_add.dat")
    make_solver(tmp_path, path, source_type="soft")


def test_a_soft_solver_refuses_a_source_that_is_already_additive(tmp_path, fake_binary):
    inputs, _, _ = additive_inputs(hard_too=False)
    with pytest.raises(ValueError, match="already carries an additive signal"):
        make_solver(tmp_path, fake_binary, inputs=inputs, source_type="soft")


def test_a_3d_binary_is_refused_for_a_2d_grid(tmp_path):
    path = tmp_path / "bench_3d_opt"
    path.write_bytes(b"nZ.dat nTzero.dat nszero0.dat")
    with pytest.raises(UltraUnsupportedError, match="3D binary"):
        make_solver(tmp_path, path)


def test_a_2d_binary_is_looked_up_by_name_in_a_folder(tmp_path, fake_binary, monkeypatch):
    monkeypatch.setenv("FW2U_BATCH_BIN", str(tmp_path))
    assert binary.resolve_binary(None, name=binary.BINARY_NAME_2D) == fake_binary
    solver = fullwave.Solver(
        work_dir=tmp_path / "work",
        grid=make_inputs()[0],
        medium=make_inputs()[1],
        source=make_inputs()[2][0],
        sensor=make_inputs()[3],
        use_ultra=True,
        run_on_memory=False,
    )
    assert solver.ultra.binary == fake_binary
