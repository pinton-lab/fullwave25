"""CPU tests of the settings names, the binary folder and the unlimited-materials check."""

import pytest

import fullwave
from fullwave.ultra.attenuation_parameters import SETTINGS_FILES, check_request
from fullwave.ultra.errors import UltraUnsupportedError

POWER_LAW_BYTES = b"\x00junk alpha_power.dat \x00"
UNLIMITED_BYTES = (
    b"\x00junk nTzero.dat nszero0.dat ncoords_add.dat alpha_power.dat --unlimited-materials \x00"
)


@pytest.fixture
def release(tmp_path):
    """Return a release folder with ``bin`` (power-law and plain builds) and ``params``."""
    bin_folder = tmp_path / "release" / "bin"
    params_folder = tmp_path / "release" / "params"
    bin_folder.mkdir(parents=True)
    params_folder.mkdir()
    for name in ("bench_2d_batch_power_law", "bench_3d_opt_power_law"):
        (bin_folder / name).write_bytes(UNLIMITED_BYTES)
    for name in ("bench_2d_batch", "bench_3d_opt"):
        (bin_folder / name).write_bytes(b"\x00junk nTzero.dat \x00")
    for file_name in SETTINGS_FILES.values():
        (params_folder / file_name).write_bytes(b"settings")
    return tmp_path / "release"


def request(binary, *, parameters="centered", is_3d=False, unlimited_materials=False):
    return check_request(
        parameters=parameters,
        use_ultra=True,
        use_exponential_attenuation=False,
        binary=binary,
        is_3d=is_3d,
        unlimited_materials=unlimited_materials,
    )


@pytest.mark.parametrize("name", ["centered", "author"])
def test_name_resolves_to_the_params_folder_beside_the_binary_folder(release, name):
    result = request(release / "bin", parameters=name)
    assert result.parameter_file == release / "params" / SETTINGS_FILES[name]


def test_name_resolves_next_to_the_binary_first(release):
    next_to_binary = release / "bin" / SETTINGS_FILES["centered"]
    next_to_binary.write_bytes(b"next to the binary")
    assert request(release / "bin").parameter_file == next_to_binary


def test_missing_settings_file_error_lists_both_places(release):
    (release / "params" / SETTINGS_FILES["author"]).unlink()
    with pytest.raises(UltraUnsupportedError) as error:
        request(release / "bin", parameters="author")
    message = str(error.value)
    assert str(release / "bin" / SETTINGS_FILES["author"]) in message
    assert str(release / "params" / SETTINGS_FILES["author"]) in message


def test_file_path_is_used_as_given(release, tmp_path):
    own_file = tmp_path / "own.dat"
    own_file.write_bytes(b"own")
    assert request(release / "bin", parameters=own_file).parameter_file == own_file


def test_unknown_name_that_is_not_a_file_is_refused(release):
    with pytest.raises(UltraUnsupportedError, match="centered"):
        request(release / "bin", parameters="middle")


@pytest.mark.parametrize(
    ("is_3d", "expected"), [(False, "bench_2d_batch_power_law"), (True, "bench_3d_opt_power_law")]
)
def test_folder_resolves_to_the_power_law_build_of_the_dimension(release, is_3d, expected):
    assert request(release / "bin", is_3d=is_3d).binary == release / "bin" / expected


def test_binary_file_is_used_as_given(release):
    plain_name = release / "bin" / "bench_2d_batch_power_law"
    assert request(plain_name).binary == plain_name


def test_folder_without_the_power_law_build_is_refused(release):
    (release / "bin" / "bench_2d_batch_power_law").unlink()
    with pytest.raises(UltraUnsupportedError, match="bench_2d_batch_power_law"):
        request(release / "bin")


def test_plain_binary_is_refused(release):
    with pytest.raises(UltraUnsupportedError, match="alpha_power.dat"):
        request(release / "bin" / "bench_2d_batch")


def test_unlimited_materials_is_accepted_by_a_binary_with_the_flag(release):
    assert request(release / "bin", unlimited_materials=True).binary.is_file()


def test_unlimited_materials_is_refused_for_a_binary_without_the_flag(release):
    old_build = release / "bin" / "bench_2d_batch_power_law"
    old_build.write_bytes(POWER_LAW_BYTES)
    with pytest.raises(UltraUnsupportedError, match="ultra_unlimited_materials"):
        request(old_build, unlimited_materials=True)


def test_old_build_still_runs_without_unlimited_materials(release):
    old_build = release / "bin" / "bench_2d_batch_power_law"
    old_build.write_bytes(POWER_LAW_BYTES)
    assert request(old_build).binary == old_build


def test_solver_takes_the_folder_and_the_name(solver_arguments, release):
    solver = fullwave.Solver(
        **solver_arguments(
            use_exponential_attenuation=False,
            ultra_attenuation_parameters="centered",
            path_fullwave_simulation_bin=release / "bin",
        )
    )
    assert solver.ultra.binary == release / "bin" / "bench_2d_batch_power_law"
    assert solver.ultra.attenuation_parameters == release / "params" / SETTINGS_FILES["centered"]
