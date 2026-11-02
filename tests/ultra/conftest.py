"""Shared builder of a small 2D Solver(use_ultra=True) for the warning tests."""

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


@pytest.fixture
def solver_arguments(tmp_path):
    """Return a function that gives the arguments of a 2D ``Solver``; options override defaults."""

    def arguments(*, ppw=6, **solver_options):
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
        binary_path.write_bytes(
            b"\x00junk nTzero.dat nszero0.dat soft_source.dat ncoords_add.dat \x00"
        )
        defaults = {
            "use_ultra": True,
            "use_exponential_attenuation": True,
            "ultra_sponge_thickness_px": SPONGE,
        }
        return {
            "work_dir": tmp_path / "work",
            "grid": grid,
            "medium": medium,
            "source": source,
            "sensor": sensor,
            "path_fullwave_simulation_bin": binary_path,
            "run_on_memory": False,
            **defaults,
            **solver_options,
        }

    return arguments


@pytest.fixture
def build_solver(solver_arguments):
    """Return a function that builds a 2D ``Solver``."""

    def build(**solver_options):
        return fullwave.Solver(**solver_arguments(**solver_options))

    return build
