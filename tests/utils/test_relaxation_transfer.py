"""Tests for the two transfer rules on the calibrated relaxation lookup."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from scipy.io import loadmat

import fullwave
from fullwave.solver.shipped_database import ShippedDatabase
from fullwave.utils import relaxation_parameters
from fullwave.utils.relaxation_parameters import (
    RelaxationParametersGenerator,
    band_scaled_alpha_coeff,
    band_scaled_sound_speed,
    generate_relaxation_params,
    scale_relaxation_attenuation,
    transfer_relaxation_params_to_band,
    transfer_relaxation_params_to_sound_speed,
)

if TYPE_CHECKING:
    from pathlib import Path


@pytest.fixture(scope="module")
def database_path() -> Path:
    """Return the table the package ships as its default."""
    return ShippedDatabase.table


@pytest.fixture(scope="module")
def two_mechanism_table() -> Path:
    """Return the two mechanism table, which this release no longer ships.

    The paper counted its calibrated cells on that table, so the check stays here.
    Where the file is absent the check is skipped rather than failed.
    """
    table = ShippedDatabase.root / "relaxation_params_database_num_relax=2_20260113_0957.mat"
    if not table.exists():
        pytest.skip(f"this release ships no two mechanism table: {table.name}")
    return table


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


@pytest.fixture(scope="module")
def calibrated_table(
    whole_grid: tuple[np.ndarray, np.ndarray], whole_table: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Return the cells the fit served, which is where a transfer rule has to hold.

    A cell the fit refused carries zeros, so its stretching factor is zero and
    every rule that divides by it is undefined there. The lookup still serves it
    and warns, and no medium should ask for one.
    """
    alpha_coeff, alpha_power = whole_grid
    generator = RelaxationParametersGenerator()
    calibrated = np.asarray(generator.is_calibrated(alpha_coeff, alpha_power))
    return {key: np.asarray(value)[calibrated] for key, value in whole_table.items()}


@pytest.fixture
def stretching_keys() -> tuple[str, ...]:
    return ("kappa_x1", "kappa_x2")


@pytest.fixture
def strength_keys() -> tuple[str, ...]:
    return tuple(
        f"d_{direction}_nu{order}"
        for direction in ("x1", "x2")
        for order in range(1, ShippedDatabase.mechanisms + 1)
    )


@pytest.fixture
def rate_keys() -> tuple[str, ...]:
    return tuple(
        f"alpha_{direction}_nu{order}"
        for direction in ("x1", "x2")
        for order in range(1, ShippedDatabase.mechanisms + 1)
    )


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
    calibrated_table: dict[str, np.ndarray],
    rate_keys: tuple[str, ...],
) -> None:
    for sound_speed in (1350.0, 1412.0, 1566.0, 1800.0):
        transferred = transfer_relaxation_params_to_sound_speed(calibrated_table, sound_speed)
        for key in rate_keys:
            np.testing.assert_array_equal(transferred[key], calibrated_table[key], err_msg=key)


def test_sound_speed_transfer_moves_kappa_toward_one(
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


def test_transferred_kappa_stays_strictly_positive(
    calibrated_table: dict[str, np.ndarray],
    stretching_keys: tuple[str, ...],
) -> None:
    """A stretching factor at or below zero makes the memory recursion unstable.

    The shipped table was fitted over a wide range, so `kappa_x1` runs from
    0.300000 to 1.006347 over the cells the fit served. The transfer moves it
    away from one as the sound speed rises, and the lowest value it reaches over
    1350 to 1800 m/s is 0.181818.
    """
    lowest = 1.0
    for sound_speed in (1350.0, 1412.0, 1540.0, 1566.0, 1800.0):
        transferred = transfer_relaxation_params_to_sound_speed(calibrated_table, sound_speed)
        for key in stretching_keys:
            assert transferred[key].min() > 0.0, key
            lowest = min(lowest, float(transferred[key].min()))
    assert lowest == pytest.approx(0.181818, abs=1e-6)


def test_the_shipped_stretching_factors_hold_their_measured_range(
    calibrated_table: dict[str, np.ndarray],
) -> None:
    """The momentum operator is pinned and the constitutive one carries the fit."""
    assert calibrated_table["kappa_x2"].min() == pytest.approx(1.0)
    assert calibrated_table["kappa_x2"].max() == pytest.approx(1.0)
    assert calibrated_table["kappa_x1"].min() == pytest.approx(0.3, abs=1e-9)
    assert calibrated_table["kappa_x1"].max() == pytest.approx(1.002289, abs=1e-6)


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


def _causal_phase_velocity(
    hertz: float,
    coefficient: float,
    exponent: float,
    reference_hz: float,
    speed: float,
) -> float:
    """Return the phase velocity a causal power law implies, written independently.

    The library must not be the only implementation of this relation, or the
    check below compares an implementation against itself. This one follows
    Waters et al. 2000 equations 6, 10 and 13.
    """
    if coefficient == 0.0:
        return speed
    decibel_nepers = 20.0 / np.log(10.0)
    here = coefficient * 100.0 * (hertz / 1e6) ** exponent / decibel_nepers
    there = coefficient * 100.0 * (reference_hz / 1e6) ** exponent / decibel_nepers
    if exponent % 2 == 1:
        slope = there / (2.0 * np.pi * reference_hz)
        inverse = 1.0 / speed - (2.0 / np.pi) * slope * np.log(hertz / reference_hz)
    else:
        inverse = 1.0 / speed + np.tan(exponent * np.pi / 2.0) * (
            here / (2.0 * np.pi * hertz) - there / (2.0 * np.pi * reference_hz)
        )
    return 1.0 / inverse


def test_the_band_anchor_is_the_analytic_curve_at_the_moved_reference() -> None:
    """The whole point. A transferred entry quotes its speed at 5 MHz times the scale."""
    for coefficient in (0.0022, 0.5, 1.0):
        for exponent in (0.5, 1.0001, 1.4, 1.5, 1.999):
            for band_scale in (0.1, 0.5, 2.0, 10.0):
                transferred = band_scaled_sound_speed(
                    1540.0, coefficient, exponent, band_scale, 5.0e6
                )
                expected = _causal_phase_velocity(
                    band_scale * 5.0e6, coefficient, exponent, 5.0e6, 1540.0
                )
                assert transferred == pytest.approx(expected, abs=1e-9)


def test_a_band_scale_of_one_leaves_the_sound_speed_alone() -> None:
    assert band_scaled_sound_speed(1540.0, 0.5, 1.4, 1.0) == 1540.0
    assert band_scaled_sound_speed(1412.0, 1.0, 0.7, 1.0) == 1412.0


def test_a_lossless_medium_needs_no_anchor(whole_grid: tuple[np.ndarray, np.ndarray]) -> None:
    """A medium with no attenuation has no dispersion, so no anchor moves."""
    _, alpha_power = whole_grid
    alpha_coeff = np.zeros_like(alpha_power)
    for band_scale in (0.1, 0.5, 2.0, 10.0):
        transferred = band_scaled_sound_speed(
            np.full(alpha_power.shape, 1540.0), alpha_coeff, alpha_power, band_scale
        )
        np.testing.assert_allclose(transferred, 1540.0, rtol=0, atol=1e-9)


def test_the_exponent_one_joins_its_neighbours_continuously() -> None:
    """The tangent has a pole at 1 and the band factor a matching zero, so 1 is a limit."""
    below = band_scaled_sound_speed(1540.0, 0.5, 0.9999, 0.1)
    here = band_scaled_sound_speed(1540.0, 0.5, 1.0, 0.1)
    above = band_scaled_sound_speed(1540.0, 0.5, 1.0001, 0.1)
    assert here == pytest.approx(0.5 * (below + above), abs=1e-6)
    assert abs(here - 1540.0) > 1.0


def test_the_anchor_is_per_voxel() -> None:
    alpha_coeff = np.array([[0.0, 0.5], [1.0, 0.0022]])
    alpha_power = np.array([[1.4, 1.0001], [1.5, 1.9]])
    sound_speed = np.full(alpha_coeff.shape, 1540.0)
    together = band_scaled_sound_speed(sound_speed, alpha_coeff, alpha_power, 0.1)
    for index in np.ndindex(alpha_coeff.shape):
        alone = band_scaled_sound_speed(
            1540.0, float(alpha_coeff[index]), float(alpha_power[index]), 0.1
        )
        assert together[index] == pytest.approx(alone, abs=1e-9)


def test_the_anchor_stays_under_two_per_cent_across_the_whole_grid(
    whole_grid: tuple[np.ndarray, np.ndarray],
) -> None:
    """Measured worst is 1.81% at a band scale of 10, at the top of the coefficient axis."""
    alpha_coeff, alpha_power = whole_grid
    sound_speed = np.full(alpha_coeff.shape, 1540.0)
    for band_scale in (0.1, 0.5, 2.0, 10.0):
        transferred = band_scaled_sound_speed(sound_speed, alpha_coeff, alpha_power, band_scale)
        assert np.abs(transferred - 1540.0).max() / 1540.0 < 0.02


def _medium_at(band_scale: float, alpha_power: float = 1.4) -> fullwave.MediumRelaxationMaps:
    """Build one small medium through the lookup, at the given band scale."""
    grid = fullwave.Grid(
        domain_size=(2e-3, 2e-3), f0=1.0e6, duration=1e-6, c0=1540.0, ppw=16, cfl=0.2
    )
    shape = (grid.nx, grid.ny)
    return fullwave.Medium(
        grid=grid,
        sound_speed=np.full(shape, 1540.0),
        density=np.full(shape, 1000.0),
        alpha_coeff=np.full(shape, 0.5),
        alpha_power=np.full(shape, alpha_power),
        beta=np.zeros(shape),
        band_scale=band_scale,
    ).build()


def test_medium_without_a_band_transfer_keeps_its_sound_speed() -> None:
    """Every medium built before the anchor existed must be unchanged."""
    np.testing.assert_array_equal(_medium_at(1.0).sound_speed, 1540.0)


def test_medium_with_a_band_transfer_moves_its_sound_speed() -> None:
    """The base speed moves so the medium carries 1540 m/s at 5 MHz again."""
    expected = band_scaled_sound_speed(1540.0, 0.5, 1.4, 0.1)
    np.testing.assert_allclose(_medium_at(0.1).sound_speed, expected, rtol=1e-12)
    assert abs(expected - 1540.0) > 1.0


def test_a_lossless_medium_with_a_band_transfer_keeps_its_sound_speed() -> None:
    grid = fullwave.Grid(
        domain_size=(2e-3, 2e-3), f0=1.0e6, duration=1e-6, c0=1540.0, ppw=16, cfl=0.2
    )
    shape = (grid.nx, grid.ny)
    built = fullwave.Medium(
        grid=grid,
        sound_speed=np.full(shape, 1540.0),
        density=np.full(shape, 1000.0),
        alpha_coeff=np.zeros(shape),
        alpha_power=np.full(shape, 1.4),
        beta=np.zeros(shape),
        band_scale=0.1,
    ).build()
    np.testing.assert_allclose(built.sound_speed, 1540.0, rtol=0, atol=1e-9)


def test_the_attenuation_scaling_is_a_bit_exact_identity_at_one(
    whole_table: dict[str, np.ndarray],
) -> None:
    scaled = scale_relaxation_attenuation(whole_table, 1.0)
    for key, before in whole_table.items():
        np.testing.assert_array_equal(scaled[key], before, err_msg=key)


def test_the_sound_speed_transfer_is_the_attenuation_scaling(
    calibrated_table: dict[str, np.ndarray],
) -> None:
    """One operation, two callers. Two implementations of one thing drift."""
    for sound_speed in (1350.0, 1412.0, 1566.0, 1800.0):
        transferred = transfer_relaxation_params_to_sound_speed(calibrated_table, sound_speed)
        scaled = scale_relaxation_attenuation(calibrated_table, sound_speed / 1540.0)
        for key in calibrated_table:
            np.testing.assert_array_equal(transferred[key], scaled[key], err_msg=key)


def test_the_shortfall_is_one_on_a_calibrated_level(
    database_path: Path, whole_grid: tuple[np.ndarray, np.ndarray]
) -> None:
    """Every request that sits on the axis is served exactly, so nothing is scaled."""
    alpha_coeff, _ = whole_grid
    generator = RelaxationParametersGenerator(path_database=database_path)
    np.testing.assert_allclose(generator.alpha_coeff_shortfall(alpha_coeff), 1.0, rtol=1e-12)


def test_the_shortfall_reports_the_quantization_and_the_clip(database_path: Path) -> None:
    """The axis steps by 0.01 and stops at 0.0022 and 1.00."""
    generator = RelaxationParametersGenerator(path_database=database_path)
    asked = np.array([0.0, 0.001, 0.155, 0.5, 3.1623])
    shortfall = generator.alpha_coeff_shortfall(asked)
    assert shortfall[0] == 0.0
    assert shortfall[1] == pytest.approx(0.001 / 0.0022)
    assert shortfall[2] == pytest.approx(0.155 / 0.16)
    assert shortfall[3] == pytest.approx(1.0)
    assert shortfall[4] == pytest.approx(3.1623)


def test_scaling_to_the_request_leaves_a_calibrated_level_alone(
    whole_grid: tuple[np.ndarray, np.ndarray],
) -> None:
    """Every cell of the shipped grid is a calibrated level, so nothing may move."""
    alpha_coeff, alpha_power = whole_grid
    generator = RelaxationParametersGenerator()
    calibrated = np.asarray(generator.is_calibrated(alpha_coeff, alpha_power))
    alpha_coeff, alpha_power = alpha_coeff[calibrated], alpha_power[calibrated]
    plain = generate_relaxation_params(alpha_coeff=alpha_coeff, alpha_power=alpha_power)
    scaled = generate_relaxation_params(
        alpha_coeff=alpha_coeff,
        alpha_power=alpha_power,
        scale_to_requested_alpha_coeff=True,
    )
    for key, before in plain.items():
        np.testing.assert_allclose(scaled[key], before, rtol=1e-12, err_msg=key)


def _stretched_operator(
    relaxation_param_dict: dict[str, np.ndarray], direction: str, omega: np.ndarray
) -> np.ndarray:
    """Return S = 1 / kappa - gamma for one direction, at each angular frequency.

    This is the factor the wavenumber is built from. Written here independently
    of the library, so a check on it is not a check of the library against
    itself.
    """
    kappa = relaxation_param_dict[f"kappa_{direction}"].ravel()[0]
    total = np.zeros_like(omega, dtype=complex)
    for i_relax in (1, 2):
        strength = relaxation_param_dict[f"d_{direction}_nu{i_relax}"].ravel()[0]
        rate = relaxation_param_dict[f"alpha_{direction}_nu{i_relax}"].ravel()[0]
        total += (strength / kappa**2) / (strength / kappa + rate + 1j * omega)
    return 1.0 / kappa - total


def test_the_exact_rule_holds_each_operator_exactly_proportional(
    whole_table: dict[str, np.ndarray],
) -> None:
    """S - 1 carries the factor exactly, at every frequency."""
    omega = 2.0 * np.pi * np.logspace(6, np.log10(20e6), 50)
    one_cell = {key: value[50:51, 11:12] for key, value in whole_table.items()}
    for factor in (0.2, 0.5, 1.5, 3.1623):
        scaled = scale_relaxation_attenuation(one_cell, factor, exact=True)
        for direction in ("x1", "x2"):
            before = _stretched_operator(one_cell, direction, omega)
            after = _stretched_operator(scaled, direction, omega)
            np.testing.assert_allclose(after - 1.0, factor * (before - 1.0), rtol=1e-10)


def test_the_first_order_rule_is_not_exactly_proportional(
    whole_table: dict[str, np.ndarray],
) -> None:
    """The default rule is first order, and this states how far off it is."""
    omega = 2.0 * np.pi * np.logspace(6, np.log10(20e6), 50)
    one_cell = {key: value[50:51, 11:12] for key, value in whole_table.items()}
    scaled = scale_relaxation_attenuation(one_cell, 3.1623)
    before = _stretched_operator(one_cell, "x1", omega)
    after = _stretched_operator(scaled, "x1", omega)
    proportional = np.abs((after - 1.0) / (3.1623 * (before - 1.0)) - 1.0).max()
    assert proportional > 0.01


def test_the_exact_rule_keeps_every_rate_positive(
    calibrated_table: dict[str, np.ndarray],
    rate_keys: tuple[str, ...],
    stretching_keys: tuple[str, ...],
) -> None:
    """A negative rate makes the solver recursion coefficient exceed one.

    The stretching factor has no ceiling of one on the shipped table, because the
    fit was given a wide range so it could raise the phase velocity at a high
    exponent. It must stay strictly above zero, which is the stability limit.
    """
    for factor in (0.1, 0.5, 2.0, 3.1623):
        scaled = scale_relaxation_attenuation(calibrated_table, factor, exact=True)
        for key in rate_keys:
            assert scaled[key].min() >= 0.0, key
        for key in stretching_keys:
            assert scaled[key].min() > 0.0, key


def test_scaling_to_the_request_moves_a_quantized_request(
    stretching_keys: tuple[str, ...],
) -> None:
    """0.155 is served 0.16, which is 3.2% too much attenuation."""
    asked = np.full((2, 2), 0.155)
    power = np.full((2, 2), 1.2)
    plain = generate_relaxation_params(alpha_coeff=asked, alpha_power=power)
    scaled = generate_relaxation_params(
        alpha_coeff=asked, alpha_power=power, scale_to_requested_alpha_coeff=True
    )
    expected = scale_relaxation_attenuation(plain, 0.155 / 0.16, exact=True)
    for key in (*stretching_keys, "d_x1_nu1", "alpha_x1_nu1"):
        np.testing.assert_allclose(scaled[key], expected[key], rtol=1e-12, err_msg=key)
    assert not np.allclose(scaled["kappa_x1"], plain["kappa_x1"])
    np.testing.assert_allclose(scaled["kappa_x2"], plain["kappa_x2"], rtol=1e-12)


def test_scaling_to_the_request_makes_a_lossless_voxel_exactly_lossless() -> None:
    """A bracket of exactly one carries no attenuation and no dispersion."""
    scaled = generate_relaxation_params(
        alpha_coeff=np.zeros((2, 2)),
        alpha_power=np.full((2, 2), 1.2),
        scale_to_requested_alpha_coeff=True,
    )
    np.testing.assert_array_equal(scaled["kappa_x1"], np.ones((2, 2)))
    np.testing.assert_array_equal(scaled["kappa_x2"], np.ones((2, 2)))
    np.testing.assert_array_equal(scaled["d_x1_nu1"], np.zeros((2, 2)))


def test_scaling_to_the_request_recovers_a_band_transfer_that_clips(
    stretching_keys: tuple[str, ...],
) -> None:
    """At (1.0, 0.5) and s = 0.1 the rule asks for 3.1623 against a ceiling of 1.00."""
    asked = np.full((2, 2), 1.0)
    power = np.full((2, 2), 0.5)
    plain = generate_relaxation_params(alpha_coeff=asked, alpha_power=power, band_scale=0.1)
    scaled = generate_relaxation_params(
        alpha_coeff=asked,
        alpha_power=power,
        band_scale=0.1,
        scale_to_requested_alpha_coeff=True,
    )
    ratio = 1.0 * 0.1 ** (0.5 - 1.0)
    assert ratio == pytest.approx(3.16227766)
    expected = scale_relaxation_attenuation(plain, ratio, exact=True)
    for key, value in expected.items():
        np.testing.assert_allclose(scaled[key], value, rtol=1e-12, err_msg=key)
    moved = (scaled["kappa_x1"] / plain["kappa_x1"]) ** 2
    assert scaled["d_x1_nu1"] / plain["d_x1_nu1"] == pytest.approx(ratio * moved, rel=1e-9)
    del stretching_keys


def test_a_stretching_factor_of_one_is_a_fixed_point_of_the_exact_rule() -> None:
    """A stretching factor of one is already lossless, so no factor moves it."""
    one_cell = {
        "kappa_x1": np.ones((1, 1)),
        "kappa_x2": np.full((1, 1), 0.99),
        "d_x1_nu1": np.full((1, 1), 1.0e5),
        "alpha_x1_nu1": np.full((1, 1), 1.0e7),
        "d_x2_nu1": np.full((1, 1), 1.0e5),
        "alpha_x2_nu1": np.full((1, 1), 1.0e7),
        "d_x1_nu2": np.full((1, 1), 1.0e5),
        "alpha_x1_nu2": np.full((1, 1), 1.0e7),
        "d_x2_nu2": np.full((1, 1), 1.0e5),
        "alpha_x2_nu2": np.full((1, 1), 1.0e7),
    }
    scaled = scale_relaxation_attenuation(one_cell, 3.0, 2, exact=True)
    np.testing.assert_array_equal(scaled["kappa_x1"], np.ones((1, 1)))
    assert scaled["kappa_x2"] < one_cell["kappa_x2"]
    np.testing.assert_allclose(scaled["d_x1_nu1"], 3.0 * 1.0e5, rtol=1e-12)


def test_medium_carries_the_request_scaling_through_to_the_build() -> None:
    grid = fullwave.Grid(
        domain_size=(2e-3, 2e-3), f0=1.0e6, duration=1e-6, c0=1540.0, ppw=16, cfl=0.2
    )
    shape = (grid.nx, grid.ny)
    built = [
        fullwave.Medium(
            grid=grid,
            sound_speed=np.full(shape, 1540.0),
            density=np.full(shape, 1000.0),
            alpha_coeff=np.full(shape, 0.155),
            alpha_power=np.full(shape, 1.2),
            beta=np.zeros(shape),
            scale_to_requested_alpha_coeff=asked,
        ).build()
        for asked in (False, True)
    ]
    plain, scaled = (one.relaxation_param_dict for one in built)
    expected = scale_relaxation_attenuation(plain, 0.155 / 0.16, exact=True)
    np.testing.assert_allclose(scaled["kappa_x1"], expected["kappa_x1"], rtol=1e-12)
    assert not np.allclose(scaled["kappa_x1"], plain["kappa_x1"])


def test_the_shipped_table_flags_124_of_its_1717_cells(
    whole_grid: tuple[np.ndarray, np.ndarray],
) -> None:
    """1593 cells passed the calibration of the four mechanism table.

    102 of the 124 come from the fit's own gate, 8 from the curve gate and 14
    from the absorbing layer gate, which the shipped table now carries.
    """
    alpha_coeff, alpha_power = whole_grid
    generator = RelaxationParametersGenerator()
    calibrated = generator.is_calibrated(alpha_coeff, alpha_power)
    assert calibrated.size == 1717
    assert int(calibrated.sum()) == 1593
    assert int((~calibrated).sum()) == 124


def test_the_flagged_cells_gather_at_the_high_exponents(
    whole_grid: tuple[np.ndarray, np.ndarray],
) -> None:
    """110 of the 124 flagged cells sit at an exponent of 1.5 or above."""
    alpha_coeff, alpha_power = whole_grid
    generator = RelaxationParametersGenerator()
    flagged = ~generator.is_calibrated(alpha_coeff, alpha_power)
    assert int(flagged[alpha_power >= 1.5].sum()) == 110
    assert int(flagged[alpha_power < 1.5].sum()) == 14


def test_the_two_mechanism_table_flags_389_of_its_1717_cells(
    two_mechanism_table: Path,
) -> None:
    """1328 cells passed that calibration, which is the number the paper prints.

    The count is checked against the two mechanism table the paper used, rather
    than against the table the package now defaults to.
    """
    generator = RelaxationParametersGenerator(
        n_relaxation_mechanisms=2, path_database=two_mechanism_table
    )
    alpha_coeff, alpha_power = np.meshgrid(
        np.asarray(generator.alpha_list).ravel(),
        np.asarray(generator.power_list).ravel(),
        indexing="ij",
    )
    calibrated = generator.is_calibrated(alpha_coeff, alpha_power)
    assert calibrated.size == 1717
    assert int(calibrated.sum()) == 1328
    assert int((~calibrated).sum()) == 389


def test_the_coefficient_axis_is_full_except_at_three_exponents(
    whole_grid: tuple[np.ndarray, np.ndarray],
) -> None:
    """The four mechanism fit reaches the top of the axis at 14 of the 17 exponents.

    It stops at 0.94 at an exponent of 1.7, at 0.71 at 1.8 and at 0.7 at 1.999.
    """
    alpha_coeff, alpha_power = whole_grid
    generator = RelaxationParametersGenerator()
    calibrated = generator.is_calibrated(alpha_coeff, alpha_power)
    exponents = alpha_power[0, :]
    largest = {
        float(exponents[column]): float(alpha_coeff[:, column][calibrated[:, column]].max())
        for column in range(calibrated.shape[1])
    }
    assert len(largest) == 17
    assert largest[1.7] == pytest.approx(0.94)
    assert largest[1.8] == pytest.approx(0.71)
    assert largest[1.999] == pytest.approx(0.7)
    full = [exponent for exponent, held in largest.items() if held == pytest.approx(1.0)]
    assert len(full) == 14


def test_a_request_off_the_axis_reports_the_flag_of_the_cell_it_clips_to(
    database_path: Path,
) -> None:
    """A clipped request is served an entry, so it inherits that entry's flag."""
    generator = RelaxationParametersGenerator(path_database=database_path)
    at_the_ceiling = generator.is_calibrated(np.array([1.0]), np.array([1.999]))
    over_the_ceiling = generator.is_calibrated(np.array([5.0]), np.array([1.999]))
    np.testing.assert_array_equal(over_the_ceiling, at_the_ceiling)


def _memory_decay(relaxation_param_dict: dict[str, np.ndarray], time_step: float) -> np.ndarray:
    """Return the memory variable decay coefficient b of every mechanism.

    The update is `psi_n = b psi_{n-1} + a (dq/dx)`, with
    `b = exp(-(d / kappa + rate) * dt)`. A b at or above one grows without bound.
    """
    held = []
    for direction in ("x1", "x2"):
        kappa = relaxation_param_dict[f"kappa_{direction}"]
        for i_relax in range(1, ShippedDatabase.mechanisms + 1):
            strength = relaxation_param_dict[f"d_{direction}_nu{i_relax}"]
            rate = relaxation_param_dict[f"alpha_{direction}_nu{i_relax}"]
            held.append(np.exp(-(strength / kappa + rate) * time_step))
    return np.stack(held)


def test_the_exact_rule_leaves_the_memory_decay_untouched(
    calibrated_table: dict[str, np.ndarray],
) -> None:
    """It holds d / kappa + rate fixed, so the solver recursion does not move at all.

    No cell the fit served refuses the exact rule at a factor of 0.5 or below, so
    the whole calibrated grid is comparable there.
    """
    time_step = 0.2 / (1.0e6 * 16)
    before = _memory_decay(calibrated_table, time_step)
    for factor in (0.1, 0.3162, 0.5):
        scaled = scale_relaxation_attenuation(calibrated_table, factor, exact=True)
        np.testing.assert_allclose(_memory_decay(scaled, time_step), before, rtol=1e-12)


def test_the_exact_rule_holds_the_memory_decay_up_to_a_large_factor(
    whole_table: dict[str, np.ndarray],
) -> None:
    """One cell that the rule admits at every factor tested."""
    time_step = 0.2 / (1.0e6 * 16)
    one_cell = {key: value[50:51, 11:12] for key, value in whole_table.items()}
    before = _memory_decay(one_cell, time_step)
    for factor in (0.1, 0.5, 2.0, 3.1623, 10.0):
        scaled = scale_relaxation_attenuation(one_cell, factor, exact=True)
        np.testing.assert_allclose(_memory_decay(scaled, time_step), before, rtol=1e-12)


def test_the_memory_decay_stays_below_one_under_both_rules(
    calibrated_table: dict[str, np.ndarray],
) -> None:
    """A decay coefficient at or above one is an unstable recursion.

    A cell the fit refused carries zeros, so its decay is exactly one. It is read
    over the cells the fit served, which is where a medium may sit. Both rules
    are admissible at a factor of 0.5 or below.
    """
    time_step = 0.2 / (1.0e6 * 16)
    for factor in (0.1, 0.5):
        for exact in (False, True):
            scaled = scale_relaxation_attenuation(calibrated_table, factor, exact=exact)
            decay = _memory_decay(scaled, time_step)
            assert decay.max() < 1.0
            assert decay.min() >= 0.0


def test_only_the_exact_rule_scales_the_shipped_table_up(
    calibrated_table: dict[str, np.ndarray],
) -> None:
    """The smallest stretching factor the fit served is 0.3, so 1 - factor reaches zero.

    The first order rule carries kappa to `1 - factor (1 - kappa)`, which is at or
    below zero for a factor of 2 and above. The exact rule cannot reach zero, so
    it is the only route that scales this table up.
    """
    time_step = 0.2 / (1.0e6 * 16)
    for factor in (2.0, 2.5, 3.0, 3.1623):
        with pytest.raises(ValueError, match="unstable"):
            scale_relaxation_attenuation(calibrated_table, factor)
        scaled = scale_relaxation_attenuation(calibrated_table, factor, exact=True)
        assert scaled["kappa_x1"].min() > 0.0
        assert _memory_decay(scaled, time_step).max() < 1.0


def test_the_first_order_rule_refuses_a_factor_that_kills_the_stretching_factor(
    calibrated_table: dict[str, np.ndarray],
) -> None:
    """The smallest kappa the fit served is 0.300000, so a large factor reaches zero."""
    with pytest.raises(ValueError, match="unstable"):
        scale_relaxation_attenuation(calibrated_table, 60.0)


def test_the_exact_rule_survives_where_the_first_order_rule_refuses(
    whole_table: dict[str, np.ndarray],
) -> None:
    """It maps kappa through its reciprocal, so a positive factor cannot reach zero."""
    one_cell = {key: value[50:51, 11:12] for key, value in whole_table.items()}
    with pytest.raises(ValueError, match="unstable"):
        scale_relaxation_attenuation(one_cell, 60.0)
    scaled = scale_relaxation_attenuation(one_cell, 60.0, exact=True)
    for key in ("kappa_x1", "kappa_x2"):
        assert scaled[key].min() > 0.0
        assert scaled[key].max() <= 1.0


def test_the_exact_rule_is_refused_only_at_already_flagged_cells(
    database_path: Path,
    whole_grid: tuple[np.ndarray, np.ndarray],
    whole_table: dict[str, np.ndarray],
) -> None:
    """At the factors a band transfer asks for, every refusal is a cell the table flags."""
    alpha_coeff, alpha_power = whole_grid
    generator = RelaxationParametersGenerator(path_database=database_path)
    calibrated = generator.is_calibrated(alpha_coeff, alpha_power)
    for factor in (0.1, 0.5, 2.0, 3.1623):
        _, admissible = relaxation_parameters._exact_attenuation(
            whole_table, np.asarray(factor), 2, xp=np
        )
        assert not (~admissible & calibrated).any()
