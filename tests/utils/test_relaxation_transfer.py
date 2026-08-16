"""Tests for the two transfer rules on the calibrated relaxation lookup."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import loadmat

import fullwave
from fullwave.utils.relaxation_parameters import (
    band_scaled_alpha_coeff,
    generate_relaxation_params,
    transfer_relaxation_params_to_band,
    transfer_relaxation_params_to_sound_speed,
)


@pytest.fixture(scope="module")
def database_path() -> Path:
    return (
        Path(fullwave.__file__).parent
        / "solver"
        / "bins"
        / "database"
        / "relaxation_params_database_num_relax=2_20260113_0957.mat"
    )


@pytest.fixture(scope="module")
def whole_grid(database_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Every attenuation pair on the shipped grid."""
    database = loadmat(database_path)
    return np.meshgrid(
        database["alpha_0_list"].ravel(),
        database["power_list"].ravel(),
        indexing="ij",
    )


@pytest.fixture(scope="module")
def whole_table(whole_grid: tuple[np.ndarray, np.ndarray]) -> dict[str, np.ndarray]:
    """Every cell, read through the lookup rather than by indexing the database."""
    alpha_coeff, alpha_power = whole_grid
    return generate_relaxation_params(alpha_coeff=alpha_coeff, alpha_power=alpha_power)


@pytest.fixture
def stretching_keys() -> tuple[str, ...]:
    return ("kappa_x1", "kappa_x2")


@pytest.fixture
def strength_keys() -> tuple[str, ...]:
    return ("d_x1_nu1", "d_x1_nu2", "d_x2_nu1", "d_x2_nu2")


@pytest.fixture
def rate_keys() -> tuple[str, ...]:
    return ("alpha_x1_nu1", "alpha_x1_nu2", "alpha_x2_nu1", "alpha_x2_nu2")


def test_default_database_path_exists(database_path: Path) -> None:
    assert database_path.exists(), database_path


def test_lookup_works_without_an_explicit_database_path() -> None:
    generate_relaxation_params(
        alpha_coeff=np.full((2, 2), 0.5),
        alpha_power=np.full((2, 2), 1.2),
    )


def test_sound_speed_transfer_is_bit_exact_identity_at_the_calibration_speed(
    whole_table: dict[str, np.ndarray],
) -> None:
    """Every simulation at 1540 m/s must be unchanged bit for bit."""
    transferred = transfer_relaxation_params_to_sound_speed(whole_table, 1540.0)
    for key, before in whole_table.items():
        np.testing.assert_array_equal(transferred[key], before, err_msg=key)


def test_sound_speed_transfer_leaves_the_relaxation_frequencies_alone(
    whole_table: dict[str, np.ndarray],
    rate_keys: tuple[str, ...],
) -> None:
    for sound_speed in (1350.0, 1412.0, 1566.0, 1800.0):
        transferred = transfer_relaxation_params_to_sound_speed(whole_table, sound_speed)
        for key in rate_keys:
            np.testing.assert_array_equal(transferred[key], whole_table[key], err_msg=key)


def test_sound_speed_transfer_scales_the_departure_of_kappa_from_one(
    whole_table: dict[str, np.ndarray],
    stretching_keys: tuple[str, ...],
) -> None:
    """It is 1 + r(kappa - 1). A plain product would invert the sign above 1540."""
    ratio = 1412.0 / 1540.0
    transferred = transfer_relaxation_params_to_sound_speed(whole_table, 1412.0)
    for key in stretching_keys:
        expected = 1.0 + ratio * (whole_table[key] - 1.0)
        np.testing.assert_allclose(transferred[key], expected, rtol=0, atol=0)
        assert not np.allclose(transferred[key], ratio * whole_table[key])


def test_kappa_of_one_is_a_fixed_point(
    stretching_keys: tuple[str, ...],
    strength_keys: tuple[str, ...],
    rate_keys: tuple[str, ...],
) -> None:
    unity = {key: np.ones((3, 3)) for key in (*stretching_keys, *strength_keys, *rate_keys)}
    for sound_speed in (1350.0, 1412.0, 1566.0, 1800.0):
        transferred = transfer_relaxation_params_to_sound_speed(unity, sound_speed)
        for key in stretching_keys:
            np.testing.assert_array_equal(transferred[key], np.ones((3, 3)), err_msg=key)


def test_transferred_kappa_stays_close_to_the_fitted_range(
    whole_table: dict[str, np.ndarray],
    stretching_keys: tuple[str, ...],
) -> None:
    """The fitted range is [0.98, 1.00] and the excursion below it is one sided."""
    for sound_speed in (1350.0, 1412.0, 1540.0, 1566.0, 1800.0):
        transferred = transfer_relaxation_params_to_sound_speed(whole_table, sound_speed)
        for key in stretching_keys:
            assert transferred[key].min() >= 0.9766
            assert transferred[key].max() <= 1.0
            assert np.abs(transferred[key] - whole_table[key]).max() <= 0.0034
            if sound_speed <= 1540.0:
                assert transferred[key].min() >= 0.98


def test_sound_speed_transfer_is_per_voxel(
    stretching_keys: tuple[str, ...],
    strength_keys: tuple[str, ...],
) -> None:
    base = generate_relaxation_params(
        alpha_coeff=np.full((4, 4), 0.5),
        alpha_power=np.full((4, 4), 1.2),
    )
    sound_speed = np.full((4, 4), 1412.0)
    sound_speed[2:, :] = 1700.0
    mixed = transfer_relaxation_params_to_sound_speed(base, sound_speed)
    slow = transfer_relaxation_params_to_sound_speed(base, 1412.0)
    fast = transfer_relaxation_params_to_sound_speed(base, 1700.0)
    for key in (*stretching_keys, *strength_keys):
        np.testing.assert_array_equal(mixed[key][:2, :], slow[key][:2, :], err_msg=key)
        np.testing.assert_array_equal(mixed[key][2:, :], fast[key][2:, :], err_msg=key)


def test_band_transfer_is_the_identity_at_unit_scale(whole_table: dict[str, np.ndarray]) -> None:
    transferred = transfer_relaxation_params_to_band(whole_table, 1.0)
    for key, before in whole_table.items():
        np.testing.assert_array_equal(transferred[key], before, err_msg=key)


def test_band_transfer_scales_every_rate_and_leaves_kappa(
    whole_table: dict[str, np.ndarray],
    stretching_keys: tuple[str, ...],
    strength_keys: tuple[str, ...],
    rate_keys: tuple[str, ...],
) -> None:
    transferred = transfer_relaxation_params_to_band(whole_table, 10.0)
    for key in (*strength_keys, *rate_keys):
        np.testing.assert_allclose(transferred[key], 10.0 * whole_table[key], rtol=0, atol=0)
    for key in stretching_keys:
        np.testing.assert_array_equal(transferred[key], whole_table[key], err_msg=key)


def test_band_key_map_is_the_identity_at_unit_exponent() -> None:
    alpha_coeff = np.full((3, 3), 0.5)
    alpha_power = np.ones((3, 3))
    for band_scale in (0.1, 1.0, 10.0):
        mapped = band_scaled_alpha_coeff(alpha_coeff, alpha_power, band_scale)
        np.testing.assert_allclose(mapped, alpha_coeff)


@pytest.mark.parametrize(
    ("band_scale", "alpha_power", "expected"),
    [(10.0, 1.9, 7.943), (10.0, 1.4, 2.512), (0.1, 1.9, 0.126), (0.1, 0.5, 3.162)],
)
def test_band_key_map_matches_the_published_factors(
    band_scale: float,
    alpha_power: float,
    expected: float,
) -> None:
    mapped = band_scaled_alpha_coeff(
        np.full((1, 1), 1.0),
        np.full((1, 1), alpha_power),
        band_scale,
    )
    assert mapped[0, 0] == pytest.approx(expected, rel=1e-3)


def test_the_two_transfers_commute() -> None:
    """Only the strengths are touched by both, and there they only multiply."""
    base = generate_relaxation_params(
        alpha_coeff=np.full((3, 3), 0.5),
        alpha_power=np.full((3, 3), 1.4),
    )
    band_first = transfer_relaxation_params_to_sound_speed(
        transfer_relaxation_params_to_band(base, 10.0),
        1412.0,
    )
    speed_first = transfer_relaxation_params_to_band(
        transfer_relaxation_params_to_sound_speed(base, 1412.0),
        10.0,
    )
    for key in band_first:
        np.testing.assert_allclose(band_first[key], speed_first[key], rtol=1e-15, err_msg=key)


def test_generate_defaults_read_the_table_as_calibrated(
    whole_grid: tuple[np.ndarray, np.ndarray],
    whole_table: dict[str, np.ndarray],
) -> None:
    alpha_coeff, alpha_power = whole_grid
    again = generate_relaxation_params(
        alpha_coeff=alpha_coeff,
        alpha_power=alpha_power,
        band_scale=1.0,
        sound_speed=None,
    )
    for key, before in whole_table.items():
        np.testing.assert_array_equal(again[key], before, err_msg=key)


def test_generate_at_the_calibration_sound_speed_is_bit_identical(
    whole_grid: tuple[np.ndarray, np.ndarray],
    whole_table: dict[str, np.ndarray],
) -> None:
    alpha_coeff, alpha_power = whole_grid
    transferred = generate_relaxation_params(
        alpha_coeff=alpha_coeff,
        alpha_power=alpha_power,
        sound_speed=1540.0,
    )
    for key, before in whole_table.items():
        np.testing.assert_array_equal(transferred[key], before, err_msg=key)


def test_generate_applies_the_band_key_map_before_the_lookup(
    stretching_keys: tuple[str, ...],
) -> None:
    """At y = 1.4 and s = 0.1 the request is 0.199, served as 0.20, a different cell."""
    plain = generate_relaxation_params(
        alpha_coeff=np.full((2, 2), 0.5),
        alpha_power=np.full((2, 2), 1.4),
    )
    served = generate_relaxation_params(
        alpha_coeff=np.full((2, 2), 0.20),
        alpha_power=np.full((2, 2), 1.4),
    )
    transferred = generate_relaxation_params(
        alpha_coeff=np.full((2, 2), 0.5),
        alpha_power=np.full((2, 2), 1.4),
        band_scale=0.1,
    )
    for key in stretching_keys:
        np.testing.assert_array_equal(transferred[key], served[key], err_msg=key)
    assert not np.allclose(transferred["kappa_x1"], plain["kappa_x1"])


def test_the_transfers_do_not_modify_their_input(whole_table: dict[str, np.ndarray]) -> None:
    before = {key: value.copy() for key, value in whole_table.items()}
    transfer_relaxation_params_to_sound_speed(whole_table, 1412.0)
    transfer_relaxation_params_to_band(whole_table, 10.0)
    for key, value in before.items():
        np.testing.assert_array_equal(whole_table[key], value, err_msg=key)
