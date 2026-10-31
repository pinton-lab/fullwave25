"""CPU tests of the padding behind a held source on the grid edge: frozen shell, then sponge."""

from pathlib import Path

import numpy as np
import pytest

import fullwave
from fullwave.ultra import backend_2d
from fullwave.ultra.medium_to_aexp import (
    back_faces_of_source,
    dbmhzcm2aexp,
    extend_map,
    extend_map_faces,
    face_margins,
    sponge_nd,
)

SHAPE = (40, 32)
SHAPE_3D = (36, 28, 28)
N_TIC = 30
SPONGE = 24
MEXT = SPONGE + 8
SHELL_2D = 10
SHELL_3D = 8
BACK_2D = SHELL_2D + SPONGE
BACK_3D = SHELL_3D + SPONGE


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


def make_solver(tmp_path, binary_path, *, cells, shape=SHAPE, **kwargs):
    grid = make_grid(shape)
    rng = np.random.default_rng(3)
    medium = fullwave.Medium(
        grid=grid,
        sound_speed=1540.0 + 40.0 * rng.random(shape),
        density=1000.0 + 30.0 * rng.random(shape),
        alpha_coeff=0.5 * rng.random(shape),
        alpha_power=np.ones(shape),
        beta=np.zeros(shape),
    )
    drive = rng.standard_normal((len(cells), N_TIC)).astype(np.float32)
    source = fullwave.Source(p0=drive, coords=cells, grid_shape=shape)
    points = np.stack([np.arange(12, 30), *[np.full(18, 14)] * (len(shape) - 1)], axis=1)
    sensor = fullwave.Sensor(coords=points, grid_shape=shape, sampling_modulus_time=2)
    solver = fullwave.Solver(
        work_dir=tmp_path / "work",
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        path_fullwave_simulation_bin=binary_path,
        use_ultra=True,
        run_on_memory=False,
        ultra_sponge_thickness_px=SPONGE,
        **kwargs,
    )
    return solver, medium


def block_cells(*, rows, columns):
    return np.array([(r, c) for r in rows for c in columns], dtype=np.int64)


@pytest.fixture
def binary_2d(tmp_path):
    path = tmp_path / "bench_2d_batch"
    path.write_bytes(b"\x00junk nTzero.dat nszero0.dat soft_source.dat ncoords_add.dat \x00")
    return path


@pytest.fixture
def binary_3d(tmp_path):
    path = tmp_path / "bench_3d_opt"
    path.write_bytes(b"\x00junk nZ.dat nszero0.dat ncoords_add.dat icmat_add.dat \x00")
    return path


def read_int(rundir, name):
    return np.fromfile(Path(rundir) / name, dtype=np.int32)


def read_map(rundir, name, shape):
    return np.fromfile(Path(rundir) / name, dtype=np.float32).reshape(shape)


def run_2d(tmp_path, binary, *, cells, **kwargs):
    solver, medium = make_solver(tmp_path, binary, cells=cells, **kwargs)
    sources = [solver.source]
    (rundir,) = solver.run_batch(sources, "run", generate_input_only=True)
    return Path(rundir), solver, medium


def test_a_held_source_on_the_top_edge_has_the_top_as_its_back_face():
    cells = block_cells(rows=range(8), columns=range(8, 20))
    assert back_faces_of_source(incoords=cells, shape=SHAPE, held=True) == {"x_low"}


def test_a_held_source_on_the_bottom_edge_has_the_bottom_as_its_back_face():
    cells = block_cells(rows=range(32, 40), columns=range(8, 20))
    assert back_faces_of_source(incoords=cells, shape=SHAPE, held=True) == {"x_high"}


def test_the_source_columns_may_touch_the_side_edges_without_a_side_back_face():
    cells = block_cells(rows=range(8), columns=range(SHAPE[1]))
    assert back_faces_of_source(incoords=cells, shape=SHAPE, held=True) == {"x_low"}


def test_there_is_no_back_face_for_a_source_off_the_edge_or_not_held():
    off = block_cells(rows=range(4, 12), columns=range(8, 20))
    assert back_faces_of_source(incoords=off, shape=SHAPE, held=True) == frozenset()
    on = block_cells(rows=range(8), columns=range(8, 20))
    assert back_faces_of_source(incoords=on, shape=SHAPE, held=False) == frozenset()
    assert back_faces_of_source(incoords=on[:0], shape=SHAPE, held=True) == frozenset()


def test_a_source_without_a_unique_thin_axis_has_no_back_face():
    square = block_cells(rows=range(8), columns=range(8))
    assert back_faces_of_source(incoords=square, shape=SHAPE, held=True) == frozenset()


def test_a_back_face_of_a_3d_source_is_normal_to_its_thin_axis():
    cells = np.argwhere(np.ones((4, 12, 10), dtype=bool))
    cells[:, 1] += 16
    assert back_faces_of_source(incoords=cells, shape=SHAPE_3D, held=True) == {"x_low"}
    swapped = cells[:, [1, 0, 2]]
    assert back_faces_of_source(incoords=swapped, shape=SHAPE_3D, held=True) == {"y_low"}


def test_the_default_padding_is_the_one_of_every_face():
    volume = np.random.default_rng(1).random(SHAPE)
    margins = face_margins(ndim=2, mext=MEXT, shell=SHELL_2D, nbdy=SPONGE, back_faces=frozenset())
    assert np.array_equal(extend_map_faces(volume, margins), extend_map(volume, MEXT))
    assert set(margins.values()) == {MEXT}


def test_an_unknown_face_is_refused():
    with pytest.raises(ValueError, match="unknown faces"):
        face_margins(
            ndim=2, mext=MEXT, shell=SHELL_2D, nbdy=SPONGE, back_faces=frozenset({"z_low"})
        )


def test_the_sponge_of_a_back_face_starts_behind_the_shell_and_ends_at_the_grid():
    full = sponge_nd((50, 40), 6)
    cut = sponge_nd((50, 40), 6, back_faces=frozenset({"x_low"}), frozen_cells=3)
    assert np.all(cut[:3] == 0.0)
    assert np.array_equal(cut[3:9], full[:6])
    assert np.all(cut[9:-6, 20] == 1.0)
    assert np.array_equal(cut[-6:], full[-6:])
    assert cut[3, 20] == 0.0
    assert 0.0 < cut[8, 20] < 1.0


def test_a_zero_shell_and_no_back_face_leave_the_sponge_unchanged():
    full = sponge_nd((50, 40), 6)
    assert np.array_equal(sponge_nd((50, 40), 6, frozen_cells=3), full)
    assert np.array_equal(sponge_nd((50, 40), 6, back_faces=frozenset({"x_low"})), full)


def test_2d_held_source_on_the_top_edge_gets_the_shell_then_the_sponge_above(tmp_path, binary_2d):
    cells = block_cells(rows=range(8), columns=range(8, 20))
    rundir, solver, medium = run_2d(tmp_path, binary_2d, cells=cells, source_type="hard")
    n_x, n_y = SHAPE[0] + BACK_2D + MEXT, SHAPE[1] + 2 * MEXT
    assert read_int(rundir, "nX.dat").tolist() == [n_x]
    assert read_int(rundir, "nY.dat").tolist() == [n_y]
    writer = solver.ultra.last_run["writer"]
    assert writer["mext"] == MEXT
    assert writer["margins"] == {
        "x_low": BACK_2D,
        "x_high": MEXT,
        "y_low": MEXT,
        "y_high": MEXT,
    }
    shifted = cells + np.array([BACK_2D, MEXT])
    assert read_int(rundir, "icc.dat").tolist() == shifted[:, 0].tolist() + shifted[:, 1].tolist()
    c = read_map(rundir, "c.dat", (n_x, n_y))
    inner = np.asarray(medium.sound_speed, dtype=np.float32)
    assert np.array_equal(c[BACK_2D : BACK_2D + SHAPE[0], MEXT : MEXT + SHAPE[1]], inner)
    assert np.array_equal(c[0, MEXT : MEXT + SHAPE[1]], inner[0])
    aexp = read_map(rundir, "Aexp.dat", (n_x, n_y))
    grid = solver.grid
    decay = dbmhzcm2aexp(
        np.asarray(medium.alpha_coeff)[0], c0=grid.c0, omega0=grid.omega, dt=grid.dt
    )
    columns = slice(MEXT + 4, MEXT + SHAPE[1] - 4)
    profile = np.sin(np.pi / 2.0 * np.arange(SPONGE) / SPONGE) ** 2
    shell_and_first_sponge_rows = aexp[: SHELL_2D + 1, columns]
    sponge_rows = aexp[SHELL_2D:BACK_2D, columns]
    first_source_row = aexp[BACK_2D, columns]
    assert np.all(shell_and_first_sponge_rows == 0.0)
    assert np.allclose(sponge_rows / decay[4:-4], profile[:, None], rtol=1e-5, atol=1e-6)
    assert 0.0 < aexp[BACK_2D - 1, MEXT + 10] < decay[10]
    assert np.allclose(first_source_row, decay[4:-4], rtol=1e-6)
    assert aexp[0, 0] == 0.0
    assert aexp[n_x - 1, 20] == 0.0
    assert aexp[n_x - 1 - 5, 20] > 0.0


def test_2d_full_width_held_source_keeps_the_side_sponges(tmp_path, binary_2d):
    cells = block_cells(rows=range(8), columns=range(SHAPE[1]))
    rundir, solver, _ = run_2d(tmp_path, binary_2d, cells=cells, source_type="hard")
    assert read_int(rundir, "nY.dat").tolist() == [SHAPE[1] + 2 * MEXT]
    margins = solver.ultra.last_run["writer"]["margins"]
    assert margins["x_low"] == BACK_2D
    assert margins["y_low"] == margins["y_high"] == MEXT
    aexp = read_map(rundir, "Aexp.dat", (SHAPE[0] + BACK_2D + MEXT, SHAPE[1] + 2 * MEXT))
    assert aexp[20, 0] == 0.0
    assert aexp[20, -1] == 0.0


def test_2d_held_source_off_the_edge_is_padded_as_before(tmp_path, binary_2d):
    cells = block_cells(rows=range(4, 12), columns=range(8, 20))
    rundir, solver, _ = run_2d(tmp_path, binary_2d, cells=cells, source_type="hard")
    assert read_int(rundir, "nX.dat").tolist() == [SHAPE[0] + 2 * MEXT]
    assert set(solver.ultra.last_run["writer"]["margins"].values()) == {MEXT}


@pytest.mark.parametrize("source_type", ["hard_released_after"])
def test_2d_released_source_on_the_edge_is_padded_as_before(tmp_path, binary_2d, source_type):
    cells = block_cells(rows=range(8), columns=range(8, 20))
    rundir, solver, _ = run_2d(tmp_path, binary_2d, cells=cells, source_type=source_type)
    assert read_int(rundir, "nX.dat").tolist() == [SHAPE[0] + 2 * MEXT]
    assert set(solver.ultra.last_run["writer"]["margins"].values()) == {MEXT}


def test_2d_soft_source_on_the_edge_is_padded_as_before(tmp_path, binary_2d):
    cells = block_cells(rows=[0], columns=range(SHAPE[1]))
    rundir, solver, _ = run_2d(tmp_path, binary_2d, cells=cells, source_type="soft")
    assert read_int(rundir, "nX.dat").tolist() == [SHAPE[0] + 2 * MEXT]
    assert set(solver.ultra.last_run["writer"]["margins"].values()) == {MEXT}


def test_the_memory_estimate_uses_the_face_margins(tmp_path, binary_2d, monkeypatch):
    seen = []

    def spy(**kwargs):
        seen.append((kwargs["n_x"], kwargs["n_y"]))
        return 0, 0

    monkeypatch.setattr(backend_2d, "device_bytes_2d", spy)
    edge = block_cells(rows=range(8), columns=range(8, 20))
    inner = block_cells(rows=range(4, 12), columns=range(8, 20))
    for name, cells in (("edge", edge), ("inner", inner)):
        solver, _ = make_solver(tmp_path / name, binary_2d, cells=cells, source_type="hard")
        solver.ultra._member_bytes(n_t=100, n_tic=N_TIC, n_tic_add=0)
    assert seen == [
        (SHAPE[0] + BACK_2D + MEXT, SHAPE[1] + 2 * MEXT),
        (SHAPE[0] + 2 * MEXT, SHAPE[1] + 2 * MEXT),
    ]


def run_3d(tmp_path, binary, *, cells, **kwargs):
    solver, medium = make_solver(tmp_path, binary, cells=cells, shape=SHAPE_3D, **kwargs)
    rundir = solver.run("run", generate_input_only=True)
    return Path(rundir), solver, medium


def plate_cells(*, x0):
    cells = np.argwhere(np.ones((4, 12, 10), dtype=bool))
    return cells + np.array([x0, 8, 8])


def test_3d_held_source_on_the_top_edge_gets_the_shell_then_the_sponge_above(tmp_path, binary_3d):
    cells = plate_cells(x0=0)
    rundir, solver, _ = run_3d(tmp_path, binary_3d, cells=cells, source_type="hard")
    n_x = SHAPE_3D[0] + BACK_3D + MEXT
    n_yz = SHAPE_3D[1] + 2 * MEXT
    assert read_int(rundir, "nX.dat").tolist() == [n_x]
    assert read_int(rundir, "nY.dat").tolist() == [n_yz]
    assert read_int(rundir, "nZ.dat").tolist() == [n_yz]
    margins = solver.ultra.last_run["writer"]["margins"]
    assert margins["x_low"] == BACK_3D
    assert all(v == MEXT for k, v in margins.items() if k != "x_low")
    shifted = cells + np.array([BACK_3D, MEXT, MEXT])
    expected = shifted[:, 0].tolist() + shifted[:, 1].tolist() + shifted[:, 2].tolist()
    assert read_int(rundir, "icc.dat").tolist() == expected
    aexp = np.fromfile(rundir / "Aexp.dat", dtype=np.float32).reshape(n_x, n_yz, n_yz)
    centre = aexp[:, n_yz // 2, n_yz // 2]
    assert np.all(centre[: SHELL_3D + 1] == 0.0)
    assert np.all(centre[SHELL_3D + 1 : BACK_3D + 4] > 0.0)
    assert aexp[n_x - 1, n_yz // 2, n_yz // 2] == 0.0


def test_3d_held_source_off_the_edge_is_padded_as_before(tmp_path, binary_3d):
    rundir, solver, _ = run_3d(tmp_path, binary_3d, cells=plate_cells(x0=6), source_type="hard")
    assert read_int(rundir, "nX.dat").tolist() == [SHAPE_3D[0] + 2 * MEXT]
    assert set(solver.ultra.last_run["writer"]["margins"].values()) == {MEXT}


def test_3d_released_source_on_the_edge_is_padded_as_before(tmp_path, binary_3d):
    rundir, solver, _ = run_3d(
        tmp_path, binary_3d, cells=plate_cells(x0=0), source_type="hard_released_after"
    )
    assert read_int(rundir, "nX.dat").tolist() == [SHAPE_3D[0] + 2 * MEXT]
    assert set(solver.ultra.last_run["writer"]["margins"].values()) == {MEXT}
