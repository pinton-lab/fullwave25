"""Tests of the kerf files of the standard Solver: written files, batch steps, binary check."""

from pathlib import Path

import numpy as np
import pytest

import fullwave
from fullwave.solver import launcher as launcher_module
from fullwave.solver import solver as solver_module
from fullwave.solver.binary_markers import KerfUnsupportedError

N_STEPS = 100
KERF_ROW_OFFSET = 1
KERF_COLUMNS = (2, 3)


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    monkeypatch.setattr(launcher_module.Launcher, "_verify_cuda_devices_exist", lambda _: "0")
    monkeypatch.setattr(solver_module, "free_gpu_memory_bytes", lambda _: None)


def make_grid():
    return fullwave.Grid(domain_size=(1e-3, 1e-3), f0=1e6, duration=1.3e-6, c0=1540)


def make_medium(grid):
    shape = (grid.nx, grid.ny)
    return fullwave.Medium(
        grid=grid,
        sound_speed=np.full(shape, 1540.0),
        density=np.full(shape, 1000.0),
        alpha_coeff=np.full(shape, 0.5),
        alpha_power=np.full(shape, 1.0),
        beta=np.zeros(shape),
        use_isotropic_relaxation=True,
    )


def source_row(grid):
    return grid.nx // 2


def kerf_cells(grid):
    row = source_row(grid) + KERF_ROW_OFFSET
    return np.array([[row, column] for column in KERF_COLUMNS])


def make_transmit(grid, *, drive_end, kerf=True):
    mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    mask[source_row(grid), :] = True
    signal = np.zeros((int(mask.sum()), N_STEPS))
    signal[:, :drive_end] = 1.0
    return fullwave.Source(signal, mask, kerf_coords=kerf_cells(grid) if kerf else None)


def make_solver(tmp_path, *, binary_bytes, binary_name, source=None, transducer=None):
    grid = make_grid() if transducer is None else transducer.grid
    path = tmp_path / binary_name
    path.write_bytes(binary_bytes)
    shape = (grid.nx, grid.ny)
    return fullwave.Solver(
        work_dir=tmp_path / "work",
        grid=grid,
        medium=make_medium(grid),
        source=source,
        transducer=transducer,
        sensor=None if transducer else fullwave.Sensor(mask=np.ones(shape, dtype=bool)),
        path_fullwave_simulation_bin=path,
        use_isotropic_relaxation=True,
        source_type="hard" if transducer is None else None,
        run_on_memory=False,
        verify_gpu=False,
    )


def read_int(folder, name):
    return np.fromfile(Path(folder) / name, dtype=np.int32)


def kerf_solver(tmp_path, *, binary_bytes=b"hold_source.dat nkerf0.dat", drive_end=40):
    grid = make_grid()
    source = make_transmit(grid, drive_end=drive_end)
    return make_solver(
        tmp_path,
        binary_bytes=binary_bytes,
        binary_name="fullwave2_2d_n_relax_multi_gpu_batch",
        source=source,
    )


def test_the_kerf_files_are_written_with_the_offset_of_the_source_cells(tmp_path):
    solver = kerf_solver(tmp_path, drive_end=40)
    folder = solver.run(generate_input_only=True)
    shifted = kerf_cells(solver.grid) + solver.pml_builder.num_boundary_points
    assert read_int(folder, "icckerf.dat").tolist() == shifted.ravel().tolist()
    assert read_int(folder, "ncoordskerf.dat").tolist() == [len(shifted)]
    assert read_int(folder, "nkerf0.dat").tolist() == [40]


def test_the_kerf_cells_are_shifted_like_the_source_cells(tmp_path):
    solver = kerf_solver(tmp_path)
    folder = solver.run(generate_input_only=True)
    source_cells = read_int(folder, "icc.dat").reshape(-1, 2)
    kerf = read_int(folder, "icckerf.dat").reshape(-1, 2)
    offset = solver.pml_builder.num_boundary_points
    assert (source_cells[:, 0] == source_row(solver.grid) + offset).all()
    assert (kerf[:, 0] == source_row(solver.grid) + KERF_ROW_OFFSET + offset).all()


def test_a_batch_writes_one_switch_step_per_transmit(tmp_path):
    solver = kerf_solver(tmp_path)
    transmits = [make_transmit(solver.grid, drive_end=n) for n in (40, 90, 61)]
    (folder,) = solver.run_batch(transmits, generate_input_only=True)
    assert read_int(folder, "nkerf0.dat").tolist() == [40, 90, 61]
    assert read_int(folder, "nsims.dat").tolist() == [3]
    assert read_int(folder, "ncoordskerf.dat").tolist() == [2]


def test_a_batch_member_with_other_kerf_cells_is_an_error(tmp_path):
    solver = kerf_solver(tmp_path)
    other = make_transmit(solver.grid, drive_end=40, kerf=False)
    with pytest.raises(ValueError, match="kerf coordinates differ"):
        solver.run_batch(
            [make_transmit(solver.grid, drive_end=40), other], generate_input_only=True
        )


def test_without_kerf_cells_no_kerf_file_is_written(tmp_path):
    grid = make_grid()
    solver = make_solver(
        tmp_path,
        binary_bytes=b"hold_source.dat",
        binary_name="fullwave2_2d_n_relax_multi_gpu_batch",
        source=make_transmit(grid, drive_end=40, kerf=False),
    )
    folder = solver.run(generate_input_only=True)
    names = {p.name for p in Path(folder).iterdir()}
    assert not names & {"icckerf.dat", "ncoordskerf.dat", "nkerf0.dat"}


def test_a_binary_without_the_kerf_marker_is_refused_for_an_explicit_source(tmp_path):
    with pytest.raises(KerfUnsupportedError, match="nkerf0.dat"):
        kerf_solver(tmp_path, binary_bytes=b"hold_source.dat")


@pytest.mark.parametrize("source_type", ["soft", "hard_released_after"])
def test_an_explicit_source_with_kerf_is_refused_for_a_source_type_that_cannot_hold_it(
    tmp_path, source_type
):
    path = tmp_path / "fullwave2_2d_n_relax_multi_gpu_batch"
    path.write_bytes(b"hold_source.dat nkerf0.dat")
    grid = make_grid()
    with pytest.raises(ValueError, match=f"not {source_type!r}"):
        fullwave.Solver(
            work_dir=tmp_path / "work",
            grid=grid,
            medium=make_medium(grid),
            source=make_transmit(grid, drive_end=40),
            sensor=fullwave.Sensor(mask=np.ones((grid.nx, grid.ny), dtype=bool)),
            path_fullwave_simulation_bin=path,
            use_isotropic_relaxation=True,
            source_type=source_type,
            run_on_memory=False,
            verify_gpu=False,
        )


def array_transducer(*, hold_kerf, source_type="hard"):
    grid = make_grid()
    geometry = fullwave.transducer.TransducerGeometry(
        grid,
        number_elements=3,
        element_width_px=1,
        element_spacing_px=1,
        element_layer_px=1,
        position_px=(1, 0),
    )
    signal = np.zeros((geometry.n_sources, grid.nt))
    signal[:, :20] = 1.0
    return fullwave.Transducer(
        transducer_geometry=geometry,
        grid=grid,
        input_signal=signal,
        source_type=source_type,
        hold_kerf=hold_kerf,
    )


def test_the_transducer_kerf_is_held_by_a_binary_that_reads_it(tmp_path, caplog):
    transducer = array_transducer(hold_kerf=True)
    with caplog.at_level("WARNING"):
        solver = make_solver(
            tmp_path,
            binary_bytes=b"hold_source.dat nkerf0.dat",
            binary_name="fullwave2_2d_n_relax_multi_gpu_cuda129",
            transducer=transducer,
        )
    folder = solver.run(generate_input_only=True)
    assert read_int(folder, "nkerf0.dat").tolist() == [20]
    assert read_int(folder, "ncoordskerf.dat").tolist() == [2]
    assert not [r for r in caplog.records if "hold_kerf" in r.getMessage()]


def test_a_binary_without_the_kerf_marker_drops_the_transducer_kerf_with_one_warning(
    tmp_path, caplog
):
    transducer = array_transducer(hold_kerf=True)
    with caplog.at_level("WARNING"):
        solver = make_solver(
            tmp_path,
            binary_bytes=b"hold_source.dat",
            binary_name="fullwave2_2d_n_relax_multi_gpu_cuda129",
            transducer=transducer,
        )
    messages = [r.getMessage() for r in caplog.records if "hold_kerf" in r.getMessage()]
    assert len(messages) == 1
    assert "fullwave2_2d_n_relax_multi_gpu_cuda129" in messages[0]
    assert "nkerf0.dat" in messages[0]
    folder = solver.run(generate_input_only=True)
    names = {p.name for p in Path(folder).iterdir()}
    assert not names & {"icckerf.dat", "ncoordskerf.dat", "nkerf0.dat"}
    assert transducer.source.n_kerf == 2


@pytest.mark.parametrize("source_type", ["soft", "hard_released_after"])
def test_a_source_type_that_cannot_hold_the_kerf_drops_the_transducer_kerf_with_one_warning(
    tmp_path, caplog, source_type
):
    transducer = array_transducer(hold_kerf=True, source_type=source_type)
    with caplog.at_level("WARNING"):
        solver = make_solver(
            tmp_path,
            binary_bytes=b"hold_source.dat nkerf0.dat",
            binary_name="fullwave2_2d_n_relax_multi_gpu_cuda129",
            transducer=transducer,
        )
    assert solver.source.n_kerf == 0
    messages = [r.getMessage() for r in caplog.records if "hold_kerf" in r.getMessage()]
    assert len(messages) == 1
    assert f"not {source_type!r}" in messages[0]


@pytest.fixture
def default_binary_kerf_solver(tmp_path, monkeypatch):
    """Return a builder of Solvers that find their binaries through the default lookup."""

    def build(*, single_reads_kerf, batch_reads_kerf):
        def fake_path(*, batch=False, **_):
            reads_kerf = batch_reads_kerf if batch else single_reads_kerf
            binary = (
                tmp_path / "bins" / f"fullwave2_2d_n_relax_multi_gpu{'_batch' if batch else ''}"
            )
            binary.parent.mkdir(exist_ok=True)
            binary.write_bytes(b"hold_source.dat nkerf0.dat" if reads_kerf else b"hold_source.dat")
            return binary

        monkeypatch.setattr(solver_module, "_retrieve_fullwave_simulation_path", fake_path)
        monkeypatch.setattr(solver_module, "ensure_binary", lambda path: path)
        transducer = array_transducer(hold_kerf=True)
        return fullwave.Solver(
            work_dir=tmp_path / "work",
            grid=transducer.grid,
            medium=make_medium(transducer.grid),
            transducer=transducer,
            use_isotropic_relaxation=True,
            run_on_memory=False,
            verify_gpu=False,
        )

    return build


def two_transmits(solver):
    return [solver.transducer.source, solver.transducer.source]


def test_a_batch_binary_without_the_kerf_marker_drops_the_kerf_of_a_run_batch(
    default_binary_kerf_solver, caplog
):
    solver = default_binary_kerf_solver(single_reads_kerf=True, batch_reads_kerf=False)
    with caplog.at_level("WARNING"):
        (folder,) = solver.run_batch(two_transmits(solver), generate_input_only=True)
    names = {p.name for p in Path(folder).iterdir()}
    assert not names & {"icckerf.dat", "ncoordskerf.dat", "nkerf0.dat"}
    assert len([r for r in caplog.records if "hold_kerf" in r.getMessage()]) == 1


def test_a_batch_binary_that_reads_the_kerf_marker_holds_the_kerf_of_a_run_batch(
    default_binary_kerf_solver,
):
    solver = default_binary_kerf_solver(single_reads_kerf=False, batch_reads_kerf=True)
    (folder,) = solver.run_batch(two_transmits(solver), generate_input_only=True)
    assert read_int(folder, "nkerf0.dat").tolist() == [20, 20]
