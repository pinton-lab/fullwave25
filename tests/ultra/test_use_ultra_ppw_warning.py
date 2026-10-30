import warnings

import numpy as np
import pytest

import fullwave
from fullwave.ultra import backend

SHAPE = (40, 32)
N_TIC = 30
SPONGE = 24


@pytest.fixture(autouse=True)
def _reset_warned_flag(monkeypatch):
    monkeypatch.setattr(backend, "_shown_warnings", set())


def build_solver(tmp_path, *, ppw):
    f0, c0 = 1.0e6, 1540.0
    dx = c0 / f0 / ppw
    grid = fullwave.Grid(
        domain_size=tuple(n * dx for n in SHAPE),
        f0=f0,
        duration=15e-6,
        c0=c0,
        ppw=ppw,
        cfl=0.30,
    )
    medium = fullwave.Medium(
        grid=grid,
        sound_speed=np.full(SHAPE, c0),
        density=np.full(SHAPE, 1000.0),
        alpha_coeff=np.full(SHAPE, 0.5),
        alpha_power=np.ones(SHAPE),
        beta=np.zeros(SHAPE),
    )
    cells = np.stack([np.full(12, 4), np.arange(8, 20)], axis=1)
    source = fullwave.Source(
        p0=np.ones((len(cells), N_TIC), dtype=np.float32), coords=cells, grid_shape=SHAPE
    )
    points = np.stack([np.arange(5, 30), np.full(25, 14)], axis=1)
    sensor = fullwave.Sensor(coords=points, grid_shape=SHAPE, sampling_modulus_time=2)
    binary_path = tmp_path / "bench_2d_batch"
    binary_path.write_bytes(b"\x00junk nTzero.dat nszero0.dat soft_source.dat ncoords_add.dat \x00")
    return fullwave.Solver(
        work_dir=tmp_path / "work",
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        path_fullwave_simulation_bin=binary_path,
        use_ultra=True,
        run_on_memory=False,
        exponential_attenuation_pml_thickness_px=SPONGE,
    )


def test_ppw_above_speed_optimum_warns_and_still_runs(tmp_path):
    with pytest.warns(UserWarning, match=r"optimized for speed at ppw=6; ppw=8 runs") as record:
        solver = build_solver(tmp_path, ppw=8)

    assert any(w.filename == __file__ for w in record)
    assert solver.ultra is not None
    assert solver.run(generate_input_only=True) is not None


def test_ppw_at_speed_optimum_does_not_warn(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", backend.FullwaveUltraExperimentalWarning)
        build_solver(tmp_path, ppw=6)


def test_ppw_above_speed_optimum_warns_once(tmp_path):
    with pytest.warns(UserWarning, match=r"optimized for speed at ppw=6; ppw=8 runs"):
        build_solver(tmp_path, ppw=8)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        warnings.simplefilter("ignore", backend.FullwaveUltraExperimentalWarning)
        second_dir = tmp_path / "again"
        second_dir.mkdir()
        build_solver(second_dir, ppw=8)
