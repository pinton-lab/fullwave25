"""CPU tests of the power-law attenuation maps next to a held source on the grid edge."""

from pathlib import Path

import numpy as np
import pytest

import fullwave

SHAPE_2D = (40, 32)
SHAPE_3D = (36, 28, 28)
N_TIC = 30
SPONGE = 24
STENCIL_MARGIN = SPONGE + 8
FROZEN_SHELL_2D = 10
FROZEN_SHELL_3D = 8
BACK_MARGIN_2D = FROZEN_SHELL_2D + SPONGE
BACK_MARGIN_3D = FROZEN_SHELL_3D + SPONGE


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


def make_solver(tmp_path, *, shape, binary_marker):
    grid = make_grid(shape)
    rng = np.random.default_rng(5)
    medium = fullwave.Medium(
        grid=grid,
        sound_speed=1540.0 + 40.0 * rng.random(shape),
        density=1000.0 + 30.0 * rng.random(shape),
        alpha_coeff=0.2 + 0.6 * rng.random(shape),
        alpha_power=1.0 + 0.5 * rng.random(shape),
        beta=np.zeros(shape),
    )
    is_3d = len(shape) == len(SHAPE_3D)
    block_shape = (4, 12, 10) if is_3d else (8, 12)
    cells = np.argwhere(np.ones(block_shape, dtype=bool)) + np.array([0, 8, 8][: len(shape)])
    drive = rng.standard_normal((len(cells), N_TIC)).astype(np.float32)
    source = fullwave.Source(p0=drive, coords=cells, grid_shape=shape)
    points = np.stack([np.arange(12, 30), *[np.full(18, 14)] * (len(shape) - 1)], axis=1)
    sensor = fullwave.Sensor(coords=points, grid_shape=shape, sampling_modulus_time=2)
    binary = tmp_path / "binary"
    binary.write_bytes(binary_marker)
    parameters = tmp_path / "parameters.dat"
    parameters.write_bytes(bytes(range(37)))
    solver = fullwave.Solver(
        work_dir=tmp_path / "work",
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        path_fullwave_simulation_bin=binary,
        use_ultra=True,
        run_on_memory=False,
        ultra_sponge_thickness_px=SPONGE,
        source_type="hard",
        use_exponential_attenuation=False,
        ultra_attenuation_parameters=parameters,
    )
    return solver, medium


def padded_shape(shape, back_margin):
    return tuple(
        n + (back_margin if axis == 0 else STENCIL_MARGIN) + STENCIL_MARGIN
        for axis, n in enumerate(shape)
    )


def read_padded(rundir, name, shape):
    return np.fromfile(Path(rundir) / name, dtype=np.float32).reshape(shape)


@pytest.mark.parametrize(
    ("shape", "back_margin", "binary_marker"),
    [
        (
            SHAPE_2D,
            BACK_MARGIN_2D,
            b"\x00junk nTzero.dat nszero0.dat ncoords_add.dat alpha_power.dat \x00",
        ),
        (
            SHAPE_3D,
            BACK_MARGIN_3D,
            b"\x00junk nZ.dat nszero0.dat ncoords_add.dat icmat_add.dat alpha_power.dat \x00",
        ),
    ],
    ids=["2d", "3d"],
)
def test_alpha_maps_have_the_shape_of_the_sound_speed_map(
    tmp_path, shape, back_margin, binary_marker
):
    solver, medium = make_solver(tmp_path, shape=shape, binary_marker=binary_marker)
    if len(shape) == len(SHAPE_2D):
        (rundir,) = solver.run_batch([solver.source], "run", generate_input_only=True)
    else:
        rundir = solver.run("run", generate_input_only=True)
    sides = [slice(STENCIL_MARGIN, -STENCIL_MARGIN)] * (len(shape) - 1)
    source_rows = (slice(back_margin, back_margin + shape[0]), *sides)
    for name, values in (
        ("c.dat", medium.sound_speed),
        ("alpha_coeff.dat", medium.alpha_coeff),
        ("alpha_power.dat", medium.alpha_power),
    ):
        padded = read_padded(rundir, name, padded_shape(shape, back_margin))
        assert np.array_equal(padded[source_rows], np.asarray(values, dtype=np.float32))
        outermost_row = padded[0][tuple(sides)]
        first_source_row = padded[back_margin][tuple(sides)]
        assert np.array_equal(outermost_row, first_source_row)
