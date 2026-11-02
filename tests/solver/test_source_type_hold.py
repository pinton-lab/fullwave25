import hashlib
from pathlib import Path

import numpy as np
import pytest

import fullwave
import fullwave.solver.solver as solver_module
from fullwave.solver import launcher as launcher_module
from fullwave.solver.binary_markers import HoldSourceUnsupportedError

SHORT_DRIVE_STEPS = 12


@pytest.fixture
def binary(tmp_path, monkeypatch):
    """Return a stand-in solver binary whose bytes name hold_source.dat, as a rebuilt one's do."""
    path = tmp_path / "fullwave2_2d_n_relax_multi_gpu_cuda129"
    path.write_bytes(b"reads hold_source.dat")
    monkeypatch.setattr(
        solver_module,
        "_retrieve_fullwave_simulation_path",
        lambda *_arguments, **_keywords: path,
    )
    monkeypatch.setattr(launcher_module.Launcher, "_verify_cuda_devices_exist", lambda _self: "0")
    return path


def _write_input_files(
    *,
    work_dir: Path,
    source_type: str,
    drive_steps: int | None = SHORT_DRIVE_STEPS,
) -> Path:
    """Write the input files of one small run and return their folder.

    ``drive_steps`` None gives a drive that lasts the whole run.
    """
    work_dir.mkdir(parents=True)
    grid = fullwave.Grid(domain_size=(1e-3, 1e-3), f0=1e6, duration=1.3e-6, c0=1540)
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
    mask = np.zeros(shape, dtype=bool)
    mask[0, :] = True
    steps = grid.nt if drive_steps is None else drive_steps
    drive = np.sin(np.linspace(0.0, 4.0, steps))
    solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        source=fullwave.Source(np.tile(drive, (int(mask.sum()), 1)), mask),
        sensor=fullwave.Sensor(mask=np.ones(shape, dtype=bool)),
        source_type=source_type,
        use_isotropic_relaxation=True,
        run_on_memory=False,
    )
    return solver.run(generate_input_only=True)


def _digests(folder: Path) -> dict[str, str]:
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(folder.iterdir())
        if path.is_file()
    }


def _read_count(folder: Path, name: str) -> int:
    return int(np.fromfile(folder / f"{name}.dat", dtype=np.int32)[0])


def test_hard_writes_the_drive_as_given_and_the_hold_flag(binary, tmp_path):
    folder = _write_input_files(work_dir=tmp_path / "held", source_type="hard")
    assert _read_count(folder, "nTic") == SHORT_DRIVE_STEPS < _read_count(folder, "nT")
    assert _read_count(folder, "hold_source") == 1


def test_hard_released_after_writes_the_drive_as_given_and_no_flag(binary, tmp_path):
    folder = _write_input_files(work_dir=tmp_path / "released", source_type="hard_released_after")
    assert _read_count(folder, "nTic") == SHORT_DRIVE_STEPS
    assert not (folder / "hold_source.dat").exists()


def test_hard_differs_from_hard_released_after_only_by_the_flag(binary, tmp_path):
    held = _digests(_write_input_files(work_dir=tmp_path / "held", source_type="hard"))
    released = _digests(
        _write_input_files(work_dir=tmp_path / "released", source_type="hard_released_after"),
    )
    assert set(held) - set(released) == {"hold_source.dat"}
    assert {name: held[name] for name in released} == released


def test_hard_with_a_drive_that_lasts_the_run_writes_no_flag(binary, tmp_path):
    folder = _write_input_files(work_dir=tmp_path / "full", source_type="hard", drive_steps=None)
    assert _read_count(folder, "nTic") == _read_count(folder, "nT")
    assert not (folder / "hold_source.dat").exists()


def test_hard_refuses_a_binary_that_does_not_read_the_flag(binary, tmp_path):
    binary.write_bytes(b"an older solver")
    with pytest.raises(HoldSourceUnsupportedError, match="FULLWAVE25_BINARY_TAG"):
        _write_input_files(work_dir=tmp_path / "held", source_type="hard")


def test_hard_with_a_drive_that_lasts_the_run_accepts_an_older_binary(binary, tmp_path):
    binary.write_bytes(b"an older solver")
    _write_input_files(work_dir=tmp_path / "full", source_type="hard", drive_steps=None)


def test_hard_released_after_accepts_an_older_binary(binary, tmp_path):
    binary.write_bytes(b"an older solver")
    _write_input_files(work_dir=tmp_path / "released", source_type="hard_released_after")


def test_the_source_types_are_hard_hard_released_after_and_soft():
    assert fullwave.solver.source_type.SOURCE_TYPES == ("hard", "hard_released_after", "soft")
