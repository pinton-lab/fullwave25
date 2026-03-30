"""Tests for gaussian_modulated_sinusoidal_signal."""

import time

import numpy as np
import pytest

from fullwave.utils.pulse import gaussian_modulated_sinusoidal_signal


@pytest.fixture
def base_params():
    return {
        "nt": 40000,
        "duration": 1e-4,
        "ncycles": 2,
        "drop_off": 2,
        "f0": 5e6,
        "p0": 1.0,
    }


def test_basic_output_shape(base_params):
    y = gaussian_modulated_sinusoidal_signal(**base_params)
    assert y.shape == (base_params["nt"],)
    assert y.dtype == np.float64


def test_zero_amplitude(base_params):
    base_params["p0"] = 0.0
    y = gaussian_modulated_sinusoidal_signal(**base_params)
    np.testing.assert_array_equal(y, 0.0)


def test_delay_shifts_signal(base_params):
    y_no_delay = gaussian_modulated_sinusoidal_signal(**base_params)
    base_params["delay_sec"] = 1e-5
    y_delayed = gaussian_modulated_sinusoidal_signal(**base_params)
    assert not np.allclose(y_no_delay, y_delayed)


def test_i_layer_requires_dt_and_cfl(base_params):
    with pytest.raises(ValueError, match="dt_for_layer_delay"):
        gaussian_modulated_sinusoidal_signal(**base_params, i_layer=1)
    with pytest.raises(ValueError, match="cfl_for_layer_delay"):
        gaussian_modulated_sinusoidal_signal(
            **base_params,
            i_layer=1,
            dt_for_layer_delay=1e-8,
        )


def test_i_layer_shifts_signal(base_params):
    y_base = gaussian_modulated_sinusoidal_signal(**base_params)
    y_layer = gaussian_modulated_sinusoidal_signal(
        **base_params,
        i_layer=5,
        dt_for_layer_delay=1e-8,
        cfl_for_layer_delay=0.4,
    )
    assert not np.allclose(y_base, y_layer)


def test_drop_off_values(base_params):
    results = {}
    for d in [1, 2, 3]:
        base_params["drop_off"] = d
        results[d] = gaussian_modulated_sinusoidal_signal(**base_params)
    # Different drop_off values should produce different signals
    assert not np.allclose(results[1], results[2])
    assert not np.allclose(results[2], results[3])


def test_float32_dtype(base_params):
    y = gaussian_modulated_sinusoidal_signal(**base_params, dtype=np.float32)
    assert y.dtype == np.float32


def test_fractional_layer_delay_preserves_phase(base_params):
    """Fractional-sample layer delays must match a manual integer shift.

    Previously, fractional delays caused carrier phase errors that inverted
    the peak sign for odd layers (Finding 19 in cross-physics experiment).
    The fix corrects carrier phase so that the layered signal matches an
    integer-sample-shifted version of the base (high NCC, same peak sign).
    """
    # cfl=0.3 => dt_layer/cfl = 3.33e-8 s per i_layer
    # dt_sim = 2.5e-9 s => 13.33 sim samples per i_layer (fractional)
    dt_layer_frac = 1e-8
    cfl_frac = 0.3
    dt_sim = base_params["duration"] / base_params["nt"]
    delay_per_layer_sec = dt_layer_frac / cfl_frac

    y_base = gaussian_modulated_sinusoidal_signal(**base_params)

    for i in range(1, 6):
        y_layer = gaussian_modulated_sinusoidal_signal(
            **base_params,
            i_layer=i,
            dt_for_layer_delay=dt_layer_frac,
            cfl_for_layer_delay=cfl_frac,
        )
        # Manual integer shift of base signal
        shift = round(delay_per_layer_sec * i / dt_sim)
        y_manual = np.zeros_like(y_base)
        y_manual[shift:] = y_base[: len(y_base) - shift]

        # NCC must be high (signals should be nearly identical)
        ncc = np.dot(y_layer, y_manual) / (np.linalg.norm(y_layer) * np.linalg.norm(y_manual))
        assert ncc > 0.999, f"i_layer={i}: NCC={ncc:.6f}, expected > 0.999"


def test_integer_layer_delay_unchanged(base_params):
    """Integer-sample layer delays should produce NCC ~ 1.0 vs manual shift."""
    dt_layer = 1e-8
    cfl_layer = 0.4
    # delay per i_layer = 2.5e-8 s = 10 sim samples (integer)
    dt_sim = base_params["duration"] / base_params["nt"]
    delay_per_layer_samples = (dt_layer / cfl_layer) / dt_sim
    assert delay_per_layer_samples == 10.0  # confirm integer

    y_base = gaussian_modulated_sinusoidal_signal(**base_params)

    for i_layer in [1, 2, 4]:
        y_layer = gaussian_modulated_sinusoidal_signal(
            **base_params,
            i_layer=i_layer,
            dt_for_layer_delay=dt_layer,
            cfl_for_layer_delay=cfl_layer,
        )
        # Manual integer shift
        shift = int(round(delay_per_layer_samples * i_layer))
        y_manual = np.zeros_like(y_base)
        y_manual[shift:] = y_base[: len(y_base) - shift]

        # NCC should be very high
        ncc = np.dot(y_layer, y_manual) / (np.linalg.norm(y_layer) * np.linalg.norm(y_manual))
        assert ncc > 0.999, f"i_layer={i_layer}: NCC={ncc:.6f}, expected > 0.999"


def test_performance(base_params):
    """Ensure the function completes in well under 1 second."""
    start = time.perf_counter()
    for _ in range(100):
        gaussian_modulated_sinusoidal_signal(**base_params)
    elapsed = time.perf_counter() - start
    # 100 calls should complete in under 1 second total
    assert elapsed < 1.0, f"100 calls took {elapsed:.2f}s, expected < 1.0s"
