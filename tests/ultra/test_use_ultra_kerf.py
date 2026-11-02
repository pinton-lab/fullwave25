"""CPU tests of the kerf cells of Solver(use_ultra=True): written files and refusals."""

import hashlib
from pathlib import Path

import numpy as np
import pytest

import fullwave
from fullwave.ultra.errors import UltraUnsupportedError

SHAPE = (40, 32)
SHAPE_3D = (36, 28, 28)
N_TIC = 30
SPONGE = 24
SHIFT = SPONGE + 8
SOURCE_ROW = 4
SOURCE_COLUMNS = [8, 9, 12, 13]
KERF_COLUMNS = [10, 11]
BINARY_2D = b"\x00junk nTzero.dat nszero0.dat soft_source.dat ncoords_add.dat nkerf0.dat \x00"
BINARY_2D_WITHOUT_KERF = BINARY_2D.replace(b"nkerf0.dat", b"")


def make_grid(shape):
    f0, c0, ppw = 1.0e6, 1540.0, 6
    dx = c0 / f0 / ppw
    return fullwave.Grid(
        domain_size=tuple(n * dx for n in shape),
        f0=f0,
        duration=15e-6,
        c0=c0,
        ppw=ppw,
        cfl=0.30,
    )


def make_medium(grid, shape, *, air_coords=None):
    return fullwave.Medium(
        grid=grid,
        sound_speed=np.full(shape, 1540.0),
        density=np.full(shape, 1000.0),
        alpha_coeff=np.full(shape, 0.5),
        alpha_power=np.ones(shape),
        beta=np.zeros(shape),
        air_coords=air_coords,
    )


def drive(*, last_sample, n_cells=4, n_steps=N_TIC):
    signal = np.zeros((n_cells, n_steps), dtype=np.float32)
    signal[:, :last_sample] = 1.0
    return signal


def kerf_cells(*, row=SOURCE_ROW):
    return np.array([[row, column] for column in KERF_COLUMNS])


def source_cells():
    return np.array([[SOURCE_ROW, column] for column in SOURCE_COLUMNS])


def make_source(*, last_sample=20, kerf=True, kerf_coords=None, **keywords):
    if kerf_coords is None and kerf:
        kerf_coords = kerf_cells()
    return fullwave.Source(
        p0=drive(last_sample=last_sample),
        coords=source_cells(),
        grid_shape=SHAPE,
        kerf_coords=kerf_coords,
        **keywords,
    )


def make_solver(
    tmp_path,
    *,
    source,
    binary_bytes=BINARY_2D,
    air_coords=None,
    source_type="hard",
    **solver_options,
):
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "bench_2d_batch"
    path.write_bytes(binary_bytes)
    grid = make_grid(SHAPE)
    sensor = fullwave.Sensor(
        coords=np.stack([np.arange(5, 30), np.full(25, 14)], axis=1),
        grid_shape=SHAPE,
        sampling_modulus_time=2,
    )
    return fullwave.Solver(
        work_dir=tmp_path / "work",
        grid=grid,
        medium=make_medium(grid, SHAPE, air_coords=air_coords),
        source=source,
        sensor=sensor,
        path_fullwave_simulation_bin=path,
        source_type=source_type,
        use_ultra=True,
        run_on_memory=False,
        ultra_sponge_thickness_px=SPONGE,
        **solver_options,
    )


def read_int(rundir, name):
    return np.fromfile(Path(rundir) / name, dtype=np.int32)


def digests(directory):
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(directory).iterdir())
    }


def test_kerf_cells_are_written_with_the_shift_of_the_source_cells(tmp_path):
    solver = make_solver(tmp_path, source=make_source())
    (rundir,) = solver.run_batch([make_source()], "run", generate_input_only=True)
    shifted = kerf_cells() + SHIFT
    assert (
        read_int(rundir, "icckerf.dat").tolist() == shifted[:, 0].tolist() + shifted[:, 1].tolist()
    )
    assert read_int(rundir, "ncoordskerf.dat").tolist() == [len(shifted)]
    assert solver.ultra.last_run["writer"]["ncoordskerf"] == len(shifted)


def test_every_transmit_gets_its_own_switch_step(tmp_path):
    solver = make_solver(tmp_path, source=make_source())
    members = [make_source(last_sample=n) for n in (20, 30, 9)]
    (rundir,) = solver.run_batch(members, "run", generate_input_only=True)
    assert read_int(rundir, "nkerf0.dat").tolist() == [20, 30, 9]


def test_the_switch_step_follows_the_last_driven_cell_of_the_transmit(tmp_path):
    late_cell = drive(last_sample=5)
    late_cell[2, 17] = 0.5
    source = fullwave.Source(
        p0=late_cell, coords=source_cells(), grid_shape=SHAPE, kerf_coords=kerf_cells()
    )
    solver = make_solver(tmp_path, source=source)
    (rundir,) = solver.run_batch([source], "run", generate_input_only=True)
    assert read_int(rundir, "nkerf0.dat").tolist() == [18]


def test_a_run_is_a_batch_of_one_switch_step(tmp_path):
    solver = make_solver(tmp_path, source=make_source(last_sample=12))
    rundir = solver.ultra.run("run", generate_input_only=True)
    assert read_int(rundir, "nkerf0.dat").tolist() == [12]


def test_without_kerf_cells_no_kerf_file_is_written(tmp_path):
    solver = make_solver(tmp_path, source=make_source(kerf=False))
    (rundir,) = solver.run_batch([make_source(kerf=False)], "run", generate_input_only=True)
    names = {p.name for p in Path(rundir).iterdir()}
    assert not names & {"icckerf.dat", "ncoordskerf.dat", "nkerf0.dat"}


def test_empty_kerf_cells_write_the_same_files_as_none(tmp_path):
    runs = []
    for index, kerf in enumerate((None, np.empty((0, 2), dtype=np.int64))):
        source = make_source(kerf=False, kerf_coords=kerf)
        solver = make_solver(tmp_path / str(index), source=source)
        runs.append(digests(solver.run_batch([source], "run", generate_input_only=True)[0]))
    assert runs[0] == runs[1]


def test_a_binary_without_the_kerf_marker_is_refused(tmp_path):
    with pytest.raises(UltraUnsupportedError, match="nkerf0.dat"):
        make_solver(tmp_path, source=make_source(), binary_bytes=BINARY_2D_WITHOUT_KERF)


def test_a_binary_without_the_kerf_marker_runs_a_source_without_kerf(tmp_path):
    make_solver(tmp_path, source=make_source(kerf=False), binary_bytes=BINARY_2D_WITHOUT_KERF)


def test_kerf_cells_on_source_cells_are_refused(tmp_path):
    source = make_source(kerf_coords=np.array([[SOURCE_ROW, 10], [SOURCE_ROW, SOURCE_COLUMNS[0]]]))
    with pytest.raises(UltraUnsupportedError, match="source cells"):
        make_solver(tmp_path, source=source)


def test_kerf_cells_on_air_cells_are_refused(tmp_path):
    with pytest.raises(UltraUnsupportedError, match="air cells"):
        make_solver(tmp_path, source=make_source(), air_coords=kerf_cells()[:1])


def test_air_cells_elsewhere_are_accepted(tmp_path):
    make_solver(tmp_path, source=make_source(), air_coords=np.array([[2, 3]]))


def test_a_batch_member_with_other_kerf_cells_is_an_error(tmp_path):
    solver = make_solver(tmp_path, source=make_source())
    other = make_source(kerf_coords=kerf_cells()[:1])
    with pytest.raises(ValueError, match="source 1"):
        solver.run_batch([make_source(), other], generate_input_only=True)
    with pytest.raises(ValueError, match="source 1"):
        solver.run_batch([make_source(), make_source(kerf=False)], generate_input_only=True)


def test_a_soft_source_cannot_hold_kerf_cells(tmp_path):
    source = fullwave.Source(
        p0=drive(last_sample=20, n_cells=SHAPE[1]),
        coords=np.stack([np.full(SHAPE[1], SOURCE_ROW), np.arange(SHAPE[1])], axis=1),
        grid_shape=SHAPE,
        kerf_coords=kerf_cells(row=SOURCE_ROW + 2),
    )
    with pytest.raises(UltraUnsupportedError, match="source_type='hard', not 'soft'"):
        make_solver(tmp_path, source=source, source_type="soft")


def test_a_source_released_after_the_drive_cannot_hold_kerf_cells(tmp_path):
    with pytest.raises(UltraUnsupportedError, match="not 'hard_released_after'"):
        make_solver(tmp_path, source=make_source(), source_type="hard_released_after")


def test_kerf_cells_outside_the_grid_are_refused():
    with pytest.raises(ValueError, match="outside the grid"):
        make_source(kerf_coords=np.array([[SHAPE[0], 3]]))


def test_kerf_cells_of_the_wrong_width_are_refused():
    with pytest.raises(ValueError, match="shape"):
        make_source(kerf_coords=np.array([[1, 2, 3]]))


def make_3d_solver(
    tmp_path, *, kerf, binary_bytes=b"nZ.dat ncoords_add.dat icmat_add.dat nszero0.dat nkerf0.dat"
):
    path = tmp_path / "bench_3d_opt"
    path.write_bytes(binary_bytes)
    grid = make_grid(SHAPE_3D)
    cells = np.array([[4, 8, 10], [4, 9, 10], [4, 12, 10]])
    source = fullwave.Source(
        p0=drive(last_sample=14, n_cells=len(cells)),
        coords=cells,
        grid_shape=SHAPE_3D,
        kerf_coords=kerf,
    )
    sensor = fullwave.Sensor(
        coords=np.stack([np.arange(5, 25), np.full(20, 14), np.full(20, 14)], axis=1),
        grid_shape=SHAPE_3D,
        sampling_modulus_time=4,
    )
    return fullwave.Solver(
        work_dir=tmp_path / "work",
        grid=grid,
        medium=make_medium(grid, SHAPE_3D),
        source=source,
        sensor=sensor,
        path_fullwave_simulation_bin=path,
        source_type="hard",
        use_ultra=True,
        run_on_memory=False,
        ultra_sponge_thickness_px=SPONGE,
    )


def test_3d_kerf_cells_are_written_with_one_switch_step(tmp_path):
    kerf = np.array([[4, 10, 10], [4, 11, 10]])
    rundir = make_3d_solver(tmp_path, kerf=kerf).run("run", generate_input_only=True)
    shifted = kerf + SHIFT
    expected = shifted[:, 0].tolist() + shifted[:, 1].tolist() + shifted[:, 2].tolist()
    assert read_int(rundir, "icckerf.dat").tolist() == expected
    assert read_int(rundir, "ncoordskerf.dat").tolist() == [2]
    assert read_int(rundir, "nkerf0.dat").tolist() == [14]


def test_3d_without_kerf_cells_writes_no_kerf_file(tmp_path):
    rundir = make_3d_solver(tmp_path, kerf=None).run("run", generate_input_only=True)
    assert not (rundir / "nkerf0.dat").exists()


def test_3d_binary_without_the_kerf_marker_is_refused(tmp_path):
    with pytest.raises(UltraUnsupportedError, match="nkerf0.dat"):
        make_3d_solver(
            tmp_path,
            kerf=np.array([[4, 10, 10]]),
            binary_bytes=b"nZ.dat ncoords_add.dat icmat_add.dat nszero0.dat",
        )


def make_array_transducer(*, hold_kerf, source_type="hard", **keywords):
    grid = fullwave.Grid(domain_size=(3e-3, 3e-3), f0=1e6, duration=4e-6, c0=1540)
    geometry = fullwave.transducer.TransducerGeometry(
        grid,
        number_elements=4,
        element_width_px=3,
        element_spacing_px=2,
        element_layer_px=2,
        position_px=(1, 0),
    )
    active = keywords.get("active_source_elements", (True,) * 4)
    signal = np.ones((sum(active) * 6, grid.nt))
    return fullwave.Transducer(
        transducer_geometry=geometry,
        grid=grid,
        input_signal=signal,
        source_type=source_type,
        hold_kerf=hold_kerf,
        **keywords,
    )


def expected_gaps():
    return np.array([[row, column] for row in (1, 2) for column in (3, 4, 8, 9, 13, 14)])


def test_the_transducer_holds_the_gaps_between_its_elements():
    source = make_array_transducer(hold_kerf=True).source
    assert source.kerf_coords.tolist() == expected_gaps().tolist()
    assert not set(map(tuple, source.kerf_coords)) & set(map(tuple, source.incoords))


def test_the_transducer_holds_no_kerf_by_default():
    source = make_array_transducer(hold_kerf=False).source
    assert source.n_kerf == 0
    assert source.kerf_coords.shape == (0, 2)


@pytest.mark.parametrize(
    "active",
    [
        (False, True, True, False),
        (True, True, False, True),
        (False, False, True, False),
    ],
)
def test_a_sub_aperture_has_the_kerf_of_the_whole_array(active):
    whole = make_array_transducer(hold_kerf=True).source.kerf_coords
    sub = make_array_transducer(hold_kerf=True, active_source_elements=active).source
    assert sub.kerf_coords.tolist() == whole.tolist()


def test_no_kerf_pixel_lies_on_an_element_pixel_active_or_not():
    transducer = make_array_transducer(
        hold_kerf=True, active_source_elements=(True, True, False, True)
    )
    elements = {tuple(cell) for cell in transducer.transducer_geometry._source_coords}
    kerf = {tuple(cell) for cell in transducer.source.kerf_coords}
    assert kerf
    assert not kerf & elements
    assert not kerf & {tuple(cell) for cell in transducer.source.incoords}


def test_the_transducer_without_a_gap_has_no_kerf():
    grid = fullwave.Grid(domain_size=(3e-3, 3e-3), f0=1e6, duration=4e-6, c0=1540)
    geometry = fullwave.transducer.TransducerGeometry(
        grid,
        number_elements=4,
        element_width_px=3,
        element_spacing_px=0,
        element_layer_px=2,
        position_px=(1, 0),
    )
    transducer = fullwave.Transducer(
        transducer_geometry=geometry,
        grid=grid,
        input_signal=np.ones((geometry.n_sources, grid.nt)),
        source_type="hard",
        hold_kerf=True,
    )
    assert transducer.source.n_kerf == 0


def test_a_soft_transducer_source_carries_no_kerf():
    source = make_array_transducer(hold_kerf=True, source_type="soft").source
    assert source.n_kerf == 0


def test_a_hard_ultra_solver_keeps_the_transducer_kerf_without_a_warning(tmp_path, caplog):
    transducer = make_array_transducer(hold_kerf=True)
    grid = transducer.grid
    path = tmp_path / "bench_2d_batch"
    path.write_bytes(BINARY_2D)
    with caplog.at_level("WARNING"):
        solver = fullwave.Solver(
            work_dir=tmp_path / "work",
            grid=grid,
            medium=make_medium(grid, (grid.nx, grid.ny)),
            transducer=transducer,
            path_fullwave_simulation_bin=path,
            use_ultra=True,
            run_on_memory=False,
        )
    assert solver.source.n_kerf > 0
    assert not [r for r in caplog.records if "hold_kerf" in r.getMessage()]


def test_transducer_kerf_runs_through_the_backend(tmp_path):
    transducer = make_array_transducer(hold_kerf=True)
    grid = transducer.grid
    shape = (grid.nx, grid.ny)
    path = tmp_path / "bench_2d_batch"
    path.write_bytes(BINARY_2D)
    solver = fullwave.Solver(
        work_dir=tmp_path / "work",
        grid=grid,
        medium=make_medium(grid, shape),
        transducer=transducer,
        sensor=fullwave.Sensor(
            coords=np.array([[5, 14]]), grid_shape=shape, sampling_modulus_time=2
        ),
        path_fullwave_simulation_bin=path,
        use_ultra=True,
        run_on_memory=False,
        ultra_sponge_thickness_px=SPONGE,
    )
    (rundir,) = solver.run_batch([transducer.source], "run", generate_input_only=True)
    assert read_int(rundir, "ncoordskerf.dat").tolist() == [len(expected_gaps())]
    assert read_int(rundir, "nkerf0.dat").tolist() == [grid.nt]


def test_a_binary_without_the_kerf_marker_drops_the_transducer_kerf_with_one_warning(
    tmp_path, caplog
):
    transducer = make_array_transducer(hold_kerf=True)
    grid = transducer.grid
    path = tmp_path / "bench_2d_batch"
    path.write_bytes(BINARY_2D_WITHOUT_KERF)
    with caplog.at_level("WARNING"):
        solver = fullwave.Solver(
            work_dir=tmp_path / "work",
            grid=grid,
            medium=make_medium(grid, (grid.nx, grid.ny)),
            transducer=transducer,
            path_fullwave_simulation_bin=path,
            use_ultra=True,
            run_on_memory=False,
            ultra_sponge_thickness_px=SPONGE,
        )
    messages = [r.getMessage() for r in caplog.records if "hold_kerf" in r.getMessage()]
    assert len(messages) == 1
    assert str(path) in messages[0]
    assert "nkerf0.dat" in messages[0]
    (rundir,) = solver.run_batch([transducer.source], "run", generate_input_only=True)
    names = {p.name for p in Path(rundir).iterdir()}
    assert not names & {"icckerf.dat", "ncoordskerf.dat", "nkerf0.dat"}
