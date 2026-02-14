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


def test_performance(base_params):
    """Ensure the function completes in well under 1 second."""
    start = time.perf_counter()
    for _ in range(100):
        gaussian_modulated_sinusoidal_signal(**base_params)
    elapsed = time.perf_counter() - start
    # 100 calls should complete in under 1 second total
    assert elapsed < 1.0, f"100 calls took {elapsed:.2f}s, expected < 1.0s"
