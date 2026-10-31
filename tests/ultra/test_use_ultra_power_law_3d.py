"""CPU tests of the power-law attenuation parameter file in 3D with a fake binary."""

from pathlib import Path

import numpy as np
import pytest

import fullwave
from fullwave.ultra.errors import UltraUnsupportedError
from fullwave.ultra.medium_to_aexp import aexp_from_amap, extend_map

SHAPE = (36, 28, 28)
N_TIC = 30
SPONGE = 24
NEW_FILES = ("alpha_coeff.dat", "alpha_power.dat", "f0.dat", "attenuation_parameters.dat")
PARAMETER_BYTES = bytes(range(37))
MARGIN = SPONGE + 8
EXTENDED = tuple(n + 2 * MARGIN for n in SHAPE)


def make_inputs(*, alpha_power):
    f0, c0, ppw = 1.0e6, 1540.0, 6
    dx = c0 / f0 / ppw
    grid = fullwave.Grid(
        domain_size=tuple(n * dx for n in SHAPE),
        f0=f0,
        duration=15e-6,
        c0=c0,
        ppw=ppw,
        cfl=0.30,
    )
    rng = np.random.default_rng(3)
    medium = fullwave.Medium(
        grid=grid,
        sound_speed=1540.0 + 40.0 * rng.random(SHAPE),
        density=1000.0 + 30.0 * rng.random(SHAPE),
        alpha_coeff=0.2 + 0.6 * rng.random(SHAPE),
        alpha_power=alpha_power(rng),
        beta=np.zeros(SHAPE),
    )
    cells = np.argwhere(np.zeros(SHAPE, dtype=bool) == 0)[::97][:40]
    drive = rng.standard_normal((len(cells), N_TIC)).astype(np.float32)
    source = fullwave.Source(p0=drive, coords=cells, grid_shape=SHAPE)
    points = np.stack([np.arange(5, 30), np.full(25, 14), np.full(25, 14)], axis=1)
    sensor = fullwave.Sensor(coords=points, grid_shape=SHAPE, sampling_modulus_time=4)
    return grid, medium, source, sensor


@pytest.fixture
def parameter_file(tmp_path):
    path = tmp_path / "parameters.dat"
    path.write_bytes(PARAMETER_BYTES)
    return path


@pytest.fixture
def capable_binary(tmp_path):
    path = tmp_path / "capable_binary_3d"
    path.write_bytes(
        b"\x00junk nZ.dat ncoords_add.dat icmat_add.dat alpha_power.dat --unlimited-materials \x00"
    )
    return path


@pytest.fixture
def build_power_law_solver(tmp_path, parameter_file, capable_binary):
    def build(*, alpha_power=lambda rng: 1.0 + 0.5 * rng.random(SHAPE), **overrides):
        grid, medium, source, sensor = make_inputs(alpha_power=alpha_power)
        options = {
            "work_dir": tmp_path / "work",
            "grid": grid,
            "medium": medium,
            "source": source,
            "sensor": sensor,
            "path_fullwave_simulation_bin": capable_binary,
            "use_ultra": True,
            "use_exponential_attenuation": False,
            "run_on_memory": False,
            "ultra_sponge_thickness_px": SPONGE,
            "ultra_attenuation_parameters": parameter_file,
        }
        return fullwave.Solver(**{**options, **overrides})

    return build


def read_map(rundir, name):
    return np.fromfile(rundir / name, dtype=np.float32).reshape(EXTENDED)


def test_files_are_written_with_the_extended_grid_sizes(build_power_law_solver):
    solver = build_power_law_solver()
    rundir = Path(solver.ultra.run("run", generate_input_only=True))
    for name in ("alpha_coeff.dat", "alpha_power.dat"):
        assert (rundir / name).stat().st_size == EXTENDED[0] * EXTENDED[1] * EXTENDED[2] * 4
    assert np.fromfile(rundir / "f0.dat", dtype=np.float32)[0] == np.float32(solver.grid.f0)


def test_maps_equal_the_padded_medium_maps(build_power_law_solver):
    solver = build_power_law_solver()
    rundir = Path(solver.ultra.run("run", generate_input_only=True))
    for name, values in (
        ("alpha_coeff.dat", solver.medium.alpha_coeff),
        ("alpha_power.dat", solver.medium.alpha_power),
    ):
        expected = extend_map(np.asarray(values), MARGIN).astype(np.float32)
        assert np.array_equal(read_map(rundir, name), expected)


def test_padded_maps_have_the_shape_of_the_sound_speed_map(build_power_law_solver):
    solver = build_power_law_solver()
    rundir = Path(solver.ultra.run("run", generate_input_only=True))
    sizes = {
        name: (rundir / name).stat().st_size
        for name in ("c.dat", "alpha_coeff.dat", "alpha_power.dat")
    }
    assert len(set(sizes.values())) == 1


def test_parameter_file_is_copied_byte_for_byte(build_power_law_solver):
    rundir = Path(build_power_law_solver().ultra.run("run", generate_input_only=True))
    assert (rundir / "attenuation_parameters.dat").read_bytes() == PARAMETER_BYTES


def test_attenuation_map_file_holds_the_sponge_only(build_power_law_solver):
    solver = build_power_law_solver()
    rundir = Path(solver.ultra.run("run", generate_input_only=True))
    grid = solver.grid
    sponge_only = aexp_from_amap(
        np.zeros(EXTENDED), c0=grid.c0, omega0=grid.omega, dt=grid.dt, nbdy=SPONGE
    )
    assert np.array_equal(read_map(rundir, "Aexp.dat"), sponge_only.astype(np.float32))


def test_run_directory_without_the_keyword_has_none_of_the_files(build_power_law_solver):
    solver = build_power_law_solver(
        alpha_power=lambda _rng: np.ones(SHAPE),
        ultra_attenuation_parameters=None,
        use_exponential_attenuation=True,
    )
    rundir = Path(solver.ultra.run("run", generate_input_only=True))
    assert not any((rundir / name).exists() for name in NEW_FILES)


def test_alpha_power_other_than_one_is_converted_without_the_keyword(build_power_law_solver):
    with pytest.warns(UserWarning, match="converted to power 1"):
        build_power_law_solver(ultra_attenuation_parameters=None, use_exponential_attenuation=True)


def test_keyword_without_use_ultra_is_refused(build_power_law_solver):
    with pytest.raises(UltraUnsupportedError, match="needs use_ultra=True"):
        build_power_law_solver(use_ultra=False)


def test_binary_without_the_marker_is_refused(build_power_law_solver, tmp_path):
    plain = tmp_path / "plain_binary_3d"
    plain.write_bytes(b"\x00junk nZ.dat ncoords_add.dat icmat_add.dat \x00")
    with pytest.raises(UltraUnsupportedError, match="alpha_power.dat"):
        build_power_law_solver(path_fullwave_simulation_bin=plain)


def test_exponential_attenuation_with_the_keyword_is_refused(build_power_law_solver):
    with pytest.raises(UltraUnsupportedError, match="use_exponential_attenuation"):
        build_power_law_solver(use_exponential_attenuation=True)
