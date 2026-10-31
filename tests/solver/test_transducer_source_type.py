from pathlib import Path

import numpy as np
import pytest

import fullwave
import fullwave.solver.solver as solver_module
from fullwave.solver import launcher as launcher_module
from fullwave.transducer import Transducer, TransducerGeometry

DRIVE_STEPS = 12


@pytest.fixture
def binary(tmp_path, monkeypatch):
    """Return a stand-in solver binary whose bytes name hold_source.dat."""
    path = tmp_path / "fullwave2_2d_n_relax_multi_gpu_cuda129"
    path.write_bytes(b"reads hold_source.dat")
    monkeypatch.setattr(
        solver_module,
        "_retrieve_fullwave_simulation_path",
        lambda *_arguments, **_keywords: path,
    )
    monkeypatch.setattr(launcher_module.Launcher, "_verify_cuda_devices_exist", lambda _self: "0")
    return path


def _grid() -> fullwave.Grid:
    return fullwave.Grid(domain_size=(1e-3, 1e-3), f0=1e6, duration=1.3e-6, c0=1540)


def _transducer(grid: fullwave.Grid, *, source_type: str, drive_steps: int | None) -> Transducer:
    geometry = TransducerGeometry(
        grid,
        number_elements=grid.ny,
        element_width_px=1,
        element_spacing_px=0,
        element_layer_px=1,
        position_px=(0, 0),
    )
    steps = grid.nt if drive_steps is None else drive_steps
    signal = np.zeros((grid.ny, grid.nt))
    signal[:, :steps] = np.sin(np.linspace(0.1, 4.0, steps))
    return Transducer(
        transducer_geometry=geometry,
        grid=grid,
        input_signal=signal,
        source_type=source_type,
    )


def _solver(
    *,
    work_dir: Path,
    transducer: Transducer,
    source_type: str | None = None,
) -> fullwave.Solver:
    grid = transducer.grid
    shape = (grid.nx, grid.ny)
    medium = fullwave.Medium(
        grid=grid,
        sound_speed=np.full(shape, 1540.0),
        density=np.full(shape, 1000.0),
        alpha_coeff=np.full(shape, 0.5),
        alpha_power=np.full(shape, 1.0),
        beta=np.zeros(shape),
        use_isotropic_relaxation=True,
    )
    return fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        transducer=transducer,
        source_type=source_type,
        use_isotropic_relaxation=True,
        run_on_memory=False,
    )


def _read_count(folder: Path, name: str) -> int:
    return int(np.fromfile(folder / f"{name}.dat", dtype=np.int32)[0])


@pytest.mark.parametrize("transducer_type", ["hard", "hard_released_after", "soft"])
def test_the_solver_takes_the_transducers_source_type(binary, tmp_path, transducer_type):
    grid = _grid()
    transducer = _transducer(grid, source_type=transducer_type, drive_steps=DRIVE_STEPS)
    solver = _solver(work_dir=tmp_path / "run", transducer=transducer)
    assert solver.source_type == transducer_type


def test_the_solver_accepts_a_source_type_equal_to_the_transducers(binary, tmp_path):
    transducer = _transducer(_grid(), source_type="hard", drive_steps=DRIVE_STEPS)
    solver = _solver(work_dir=tmp_path / "run", transducer=transducer, source_type="hard")
    assert solver.source_type == "hard"


def test_the_solver_refuses_a_source_type_that_differs_from_the_transducers(binary, tmp_path):
    transducer = _transducer(_grid(), source_type="hard", drive_steps=DRIVE_STEPS)
    with pytest.raises(ValueError, match="differs from the transducer"):
        _solver(work_dir=tmp_path / "run", transducer=transducer, source_type="soft")


def test_a_source_without_a_transducer_keeps_the_released_default(binary, tmp_path):
    grid = _grid()
    shape = (grid.nx, grid.ny)
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = True
    medium = fullwave.Medium(
        grid=grid,
        sound_speed=np.full(shape, 1540.0),
        density=np.full(shape, 1000.0),
        alpha_coeff=np.full(shape, 0.5),
        alpha_power=np.full(shape, 1.0),
        beta=np.zeros(shape),
        use_isotropic_relaxation=True,
    )
    solver = fullwave.Solver(
        work_dir=tmp_path / "run",
        grid=grid,
        medium=medium,
        source=fullwave.Source(np.ones((int(mask.sum()), DRIVE_STEPS)), mask),
        sensor=fullwave.Sensor(mask=np.ones(shape, dtype=bool)),
        use_isotropic_relaxation=True,
        run_on_memory=False,
    )
    assert solver.source_type == "hard_released_after"


@pytest.mark.parametrize("source_type", ["hard", "hard_released_after"])
def test_the_hard_source_is_trimmed_to_the_drive(source_type):
    grid = _grid()
    transducer = _transducer(grid, source_type=source_type, drive_steps=DRIVE_STEPS)
    assert transducer.source.icmat.shape[1] == DRIVE_STEPS < grid.nt


def test_a_drive_that_lasts_the_run_is_not_trimmed():
    grid = _grid()
    transducer = _transducer(grid, source_type="hard", drive_steps=None)
    assert transducer.source.icmat.shape[1] == grid.nt


def test_the_soft_source_keeps_the_whole_signal():
    grid = _grid()
    transducer = _transducer(grid, source_type="soft", drive_steps=DRIVE_STEPS)
    assert transducer.source.p0_additive.shape[1] == grid.nt


def test_the_transducer_signal_is_checked_to_the_run_length():
    grid = _grid()
    transducer = _transducer(grid, source_type="hard", drive_steps=DRIVE_STEPS)
    with pytest.raises(ValueError, match="wrong number of time points"):
        transducer.set_signal(np.ones((grid.ny, DRIVE_STEPS)))


def test_a_transducer_with_the_default_type_writes_the_hold_flag(binary, tmp_path):
    transducer = _transducer(_grid(), source_type="hard", drive_steps=DRIVE_STEPS)
    folder = _solver(work_dir=tmp_path / "run", transducer=transducer).run(
        generate_input_only=True,
    )
    assert _read_count(folder, "nTic") == DRIVE_STEPS < _read_count(folder, "nT")
    assert _read_count(folder, "hold_source") == 1


def test_a_released_transducer_writes_no_hold_flag(binary, tmp_path):
    transducer = _transducer(_grid(), source_type="hard_released_after", drive_steps=DRIVE_STEPS)
    folder = _solver(work_dir=tmp_path / "run", transducer=transducer).run(
        generate_input_only=True,
    )
    assert _read_count(folder, "nTic") == DRIVE_STEPS
    assert not (folder / "hold_source.dat").exists()


def test_the_static_map_writes_the_hold_flag_for_each_transmit(binary, tmp_path):
    transducer = _transducer(_grid(), source_type="hard", drive_steps=DRIVE_STEPS)
    solver = _solver(work_dir=tmp_path / "run", transducer=transducer)
    folders = [
        solver.run(
            f"txrx_{index}",
            is_static_map=True,
            recalculate_pml=index == 0,
            generate_input_only=True,
        )
        for index in range(2)
    ]
    for folder in folders:
        assert not (folder / "hold_source.dat").is_symlink()
        assert _read_count(folder, "hold_source") == 1
        assert _read_count(folder, "nTic") == DRIVE_STEPS


def test_the_static_map_leaves_no_stale_hold_flag_when_released(binary, tmp_path):
    work_dir = tmp_path / "run"
    held = _transducer(_grid(), source_type="hard", drive_steps=DRIVE_STEPS)
    _solver(work_dir=work_dir, transducer=held).run(
        is_static_map=True,
        generate_input_only=True,
    )
    assert (work_dir / "txrx_0" / "hold_source.dat").exists()
    released = _transducer(_grid(), source_type="hard_released_after", drive_steps=DRIVE_STEPS)
    folder = _solver(work_dir=work_dir, transducer=released).run(
        is_static_map=True,
        generate_input_only=True,
    )
    assert not (folder / "hold_source.dat").exists()
