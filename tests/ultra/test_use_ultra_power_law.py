"""CPU tests of the power-law attenuation parameter file in 2D with a fake binary."""

import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import fullwave
from fullwave.ultra import binary
from fullwave.ultra.backend import power_one_alpha_coeff
from fullwave.ultra.errors import UltraUnsupportedError
from fullwave.ultra.medium_to_aexp import aexp_from_amap, extend_map

SHAPE = (40, 32)
N_TIC = 30
SPONGE = 24
NEW_FILES = ("alpha_coeff.dat", "alpha_power.dat", "f0.dat", "attenuation_parameters.dat")
PARAMETER_BYTES = bytes(range(37))
MARGIN = SPONGE + 8
EXTENDED = tuple(n + 2 * MARGIN for n in SHAPE)


@pytest.fixture
def parameter_file(tmp_path):
    path = tmp_path / "parameters.dat"
    path.write_bytes(PARAMETER_BYTES)
    return path


@pytest.fixture
def capable_binary(tmp_path):
    path = tmp_path / "capable_bench_2d_batch"
    path.write_bytes(
        b"\x00junk nTzero.dat nszero0.dat ncoords_add.dat alpha_power.dat"
        b" --unlimited-materials \x00"
    )
    return path


@pytest.fixture
def power_law_medium(solver_arguments):
    grid = solver_arguments()["grid"]
    rng = np.random.default_rng(5)
    return fullwave.Medium(
        grid=grid,
        sound_speed=np.full(SHAPE, 1540.0),
        density=np.full(SHAPE, 1000.0),
        alpha_coeff=0.2 + 0.6 * rng.random(SHAPE),
        alpha_power=1.0 + 0.5 * rng.random(SHAPE),
        beta=np.zeros(SHAPE),
    )


@pytest.fixture
def build_power_law_solver(solver_arguments, power_law_medium, parameter_file, capable_binary):
    def build(**overrides):
        options = {
            "medium": power_law_medium,
            "use_exponential_attenuation": False,
            "ultra_attenuation_parameters": parameter_file,
            "path_fullwave_simulation_bin": capable_binary,
        }
        return fullwave.Solver(**solver_arguments(**{**options, **overrides}))

    return build


def input_only_run(solver, *, members=1, **kwargs):
    cells = solver.source.incoords
    drive = np.ones((len(cells), N_TIC), dtype=np.float32)
    sources = [fullwave.Source(p0=drive, coords=cells, grid_shape=SHAPE) for _ in range(members)]
    return [Path(p) for p in solver.run_batch(sources, "run", generate_input_only=True, **kwargs)]


def read_map(rundir, name):
    return np.fromfile(rundir / name, dtype=np.float32).reshape(EXTENDED)


def test_files_are_written_with_the_extended_grid_sizes(build_power_law_solver):
    solver = build_power_law_solver()
    (rundir,) = input_only_run(solver)
    for name in ("alpha_coeff.dat", "alpha_power.dat"):
        assert (rundir / name).stat().st_size == EXTENDED[0] * EXTENDED[1] * 4
    assert (rundir / "f0.dat").stat().st_size == 4
    assert np.fromfile(rundir / "f0.dat", dtype=np.float32)[0] == np.float32(solver.grid.f0)


def test_maps_equal_the_padded_medium_maps(build_power_law_solver, power_law_medium):
    (rundir,) = input_only_run(build_power_law_solver())
    for name, values in (
        ("alpha_coeff.dat", power_law_medium.alpha_coeff),
        ("alpha_power.dat", power_law_medium.alpha_power),
    ):
        expected = extend_map(np.asarray(values), MARGIN).astype(np.float32)
        assert np.array_equal(read_map(rundir, name), expected)


def test_parameter_file_is_copied_byte_for_byte(build_power_law_solver):
    (rundir,) = input_only_run(build_power_law_solver())
    assert (rundir / "attenuation_parameters.dat").read_bytes() == PARAMETER_BYTES


def test_attenuation_map_file_holds_the_sponge_only(build_power_law_solver):
    solver = build_power_law_solver()
    (rundir,) = input_only_run(solver)
    grid = solver.grid
    sponge_only = aexp_from_amap(
        np.zeros(EXTENDED), c0=grid.c0, omega0=grid.omega, dt=grid.dt, nbdy=SPONGE
    )
    assert np.array_equal(read_map(rundir, "Aexp.dat"), sponge_only.astype(np.float32))


def test_every_launch_directory_gets_the_files(build_power_law_solver):
    runs = input_only_run(build_power_law_solver(), members=3, max_members_per_run=2)
    assert len(runs) == 2
    for rundir in runs:
        assert (rundir / "attenuation_parameters.dat").read_bytes() == PARAMETER_BYTES
        for name in NEW_FILES:
            assert (rundir / name).is_file()


def test_single_run_writes_the_files(build_power_law_solver):
    solver = build_power_law_solver()
    rundir = solver.ultra.run("single", generate_input_only=True)
    assert all((Path(rundir) / name).is_file() for name in NEW_FILES)


def test_run_directory_without_the_keyword_has_none_of_the_files(build_solver):
    (rundir,) = input_only_run(build_solver())
    assert not any((rundir / name).exists() for name in NEW_FILES)


def test_relaxation_model_warning_is_not_given(build_power_law_solver):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_power_law_solver()
    assert not [str(w.message) for w in caught if "frequency" in str(w.message)]


def uniform_medium(grid, *, alpha_coeff, alpha_power):
    return fullwave.Medium(
        grid=grid,
        sound_speed=np.full(SHAPE, 1540.0),
        density=np.full(SHAPE, 1000.0),
        alpha_coeff=np.full(SHAPE, alpha_coeff),
        alpha_power=np.full(SHAPE, alpha_power),
        beta=np.zeros(SHAPE),
    )


def aexp_of_run(build_solver, solver_arguments, *, alpha_coeff, alpha_power):
    medium = uniform_medium(
        solver_arguments()["grid"], alpha_coeff=alpha_coeff, alpha_power=alpha_power
    )
    (rundir,) = input_only_run(build_solver(medium=medium))
    return (rundir / "Aexp.dat").read_bytes()


def test_power_two_is_converted_to_power_one_at_f0(build_solver, solver_arguments):
    f0_mhz = solver_arguments()["grid"].f0 / 1e6
    converted = aexp_of_run(build_solver, solver_arguments, alpha_coeff=0.4, alpha_power=2.0)
    by_hand = aexp_of_run(build_solver, solver_arguments, alpha_coeff=0.4 * f0_mhz, alpha_power=1.0)
    assert converted == by_hand


def test_power_one_medium_is_written_unchanged(solver_arguments):
    alpha_coeff = np.linspace(0.1, 0.9, 12)
    assert (
        power_one_alpha_coeff(
            alpha_coeff=alpha_coeff, alpha_power=np.ones(12), f0=solver_arguments()["grid"].f0
        )
        is alpha_coeff
    )


def test_converted_values_are_alpha_times_f0_to_the_power_minus_one():
    alpha_power = np.array([1.0, 2.0, 1.5])
    converted = power_one_alpha_coeff(
        alpha_coeff=np.full(3, 0.5), alpha_power=alpha_power, f0=3.0e6
    )
    assert np.allclose(converted, 0.5 * 3.0 ** (alpha_power - 1.0))


def test_power_conversion_warns_once_with_the_range_and_f0(build_solver, solver_arguments):
    medium = uniform_medium(solver_arguments()["grid"], alpha_coeff=0.4, alpha_power=2.0)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_solver(medium=medium)
        build_solver(medium=medium)
    messages = [str(w.message) for w in caught if "converted to power 1" in str(w.message)]
    assert len(messages) == 1
    assert "from 2 to 2" in messages[0]
    assert "f0=1 MHz" in messages[0]


def test_power_one_medium_gives_no_conversion_warning(build_solver):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        build_solver()
    assert not any("converted to power 1" in str(w.message) for w in caught)


def test_keyword_without_use_ultra_is_refused(build_power_law_solver):
    with pytest.raises(UltraUnsupportedError, match="needs use_ultra=True"):
        build_power_law_solver(use_ultra=False)


def test_missing_parameter_file_is_refused(build_power_law_solver, tmp_path):
    with pytest.raises(UltraUnsupportedError, match="is not a file"):
        build_power_law_solver(ultra_attenuation_parameters=tmp_path / "missing.dat")


def test_parameter_folder_is_refused(build_power_law_solver, tmp_path):
    with pytest.raises(UltraUnsupportedError, match="is not a file"):
        build_power_law_solver(ultra_attenuation_parameters=tmp_path)


def test_binary_path_none_is_refused(build_power_law_solver, monkeypatch, capable_binary):
    monkeypatch.setenv("FW2U_BATCH_BIN", str(capable_binary))
    with pytest.raises(UltraUnsupportedError, match="needs path_fullwave_simulation_bin"):
        build_power_law_solver(path_fullwave_simulation_bin=None)


def test_binary_without_the_marker_is_refused(build_power_law_solver, tmp_path):
    plain = tmp_path / "plain_bench_2d_batch"
    plain.write_bytes(b"\x00junk nTzero.dat nszero0.dat ncoords_add.dat \x00")
    with pytest.raises(UltraUnsupportedError, match="alpha_power.dat"):
        build_power_law_solver(path_fullwave_simulation_bin=plain)


def test_exponential_attenuation_with_the_keyword_is_refused(build_power_law_solver):
    with pytest.raises(UltraUnsupportedError, match="use_exponential_attenuation"):
        build_power_law_solver(use_exponential_attenuation=True)


@pytest.fixture
def command_lines(monkeypatch):
    """Record the command line of every binary launch; the fake binary writes a zero output."""
    commands = []

    def fake_subprocess_run(command, *, cwd, **_kwargs):
        commands.append(command)
        np.zeros(25 * 7, dtype=np.float32).tofile(Path(cwd) / "genout.dat")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(binary, "subprocess", SimpleNamespace(run=fake_subprocess_run))
    return commands


def test_run_binary_passes_the_flag_when_asked(tmp_path, command_lines):
    binary.run_binary("solver", tmp_path, verify_gpu=False, unlimited_materials=True)
    assert command_lines == [["solver", "--unlimited-materials"]]


def test_run_binary_passes_no_argument_by_default(tmp_path, command_lines):
    binary.run_binary("solver", tmp_path, verify_gpu=False)
    assert command_lines == [["solver"]]


def test_solver_with_the_keyword_alone_launches_without_arguments(
    build_power_law_solver, command_lines
):
    solver = build_power_law_solver(verify_gpu=False)
    solver.run("run", load_results=False)
    assert [command[1:] for command in command_lines] == [[]]


def test_solver_with_unlimited_materials_launches_with_the_flag(
    build_power_law_solver, command_lines
):
    solver = build_power_law_solver(verify_gpu=False, ultra_unlimited_materials=True)
    solver.run("run", load_results=False)
    assert [command[1:] for command in command_lines] == [["--unlimited-materials"]]


def test_unlimited_materials_without_the_parameter_file_is_refused(build_solver):
    with pytest.raises(UltraUnsupportedError, match="needs ultra_attenuation_parameters"):
        build_solver(ultra_unlimited_materials=True)


def test_unlimited_materials_without_use_ultra_is_refused(build_solver):
    with pytest.raises(UltraUnsupportedError, match="needs use_ultra=True"):
        build_solver(ultra_unlimited_materials=True, use_ultra=False)


def test_solver_without_the_keyword_launches_without_arguments(build_solver, command_lines):
    solver = build_solver(verify_gpu=False)
    solver.run("run", load_results=False)
    assert [len(command) for command in command_lines] == [1]
