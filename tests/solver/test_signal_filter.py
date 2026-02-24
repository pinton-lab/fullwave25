"""Unit tests for fullwave.utils.signal_filter — no GPU required."""

from unittest.mock import MagicMock

import numpy as np
import pytest

import fullwave
from fullwave.solver.solver import Solver
from fullwave.utils.signal_filter import _build_frequency_mask, apply_filter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(dt: float, n_t: int, frequencies_hz: list[float]) -> np.ndarray:
    """Sum of pure sinusoids, shape [1, n_t]."""
    t = np.arange(n_t) * dt
    sig = sum(np.sin(2 * np.pi * f * t) for f in frequencies_hz)
    return np.asarray(sig, dtype=np.float64)[np.newaxis, :]


# ---------------------------------------------------------------------------
# _build_frequency_mask
# ---------------------------------------------------------------------------


def test_mask_shape():
    n_fft = 512
    dt = 1e-8
    mask = _build_frequency_mask(n_fft, dt)
    assert mask.shape == (n_fft // 2 + 1,)


def test_mask_all_ones_when_no_cutoff():
    mask = _build_frequency_mask(256, 1e-8)
    np.testing.assert_array_equal(mask, np.ones(256 // 2 + 1))


def test_mask_highpass_dc_zero():
    """DC bin (index 0) must be zero for any high-pass cutoff > 0."""
    mask = _build_frequency_mask(256, 1e-8, f_low_hz=1e6)
    assert mask[0] == 0.0


def test_mask_highpass_passband_one():
    """Bins well above the cutoff should be ≈ 1."""
    dt = 1e-8
    n_fft = 1024
    f_low_hz = 1e6
    mask = _build_frequency_mask(n_fft, dt, f_low_hz=f_low_hz)
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    passband = freqs > f_low_hz * 1.1
    np.testing.assert_allclose(mask[passband], 1.0, atol=1e-10)


def test_mask_bandpass_range():
    """Bins well inside the band ≈ 1; bins well outside ≈ 0."""
    dt = 1e-8
    n_fft = 1024
    f_low_hz = 1e6
    f_high_hz = 5e6
    mask = _build_frequency_mask(n_fft, dt, f_low_hz=f_low_hz, f_high_hz=f_high_hz)
    freqs = np.fft.rfftfreq(n_fft, d=dt)

    passband = (freqs > f_low_hz * 1.2) & (freqs < f_high_hz * 0.8)
    stopband_low = freqs < f_low_hz * 0.8
    stopband_high = freqs > f_high_hz * 1.2

    np.testing.assert_allclose(mask[passband], 1.0, atol=1e-10)
    np.testing.assert_allclose(mask[stopband_low], 0.0, atol=1e-10)
    np.testing.assert_allclose(mask[stopband_high], 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# apply_filter — functional tests (CPU only via use_gpu=False)
# ---------------------------------------------------------------------------


def test_no_filter_passthrough():
    """With no cutoffs the output must equal the input (up to float rounding)."""
    dt = 1e-8
    n_t = 512
    rng = np.random.default_rng(0)
    data = rng.standard_normal((8, n_t))
    result = apply_filter(data, dt, use_gpu=False)
    np.testing.assert_allclose(result, data, atol=1e-10)


def test_highpass_removes_dc():
    """DC + sine: after high-pass the DC component should vanish, sine should survive."""
    dt = 1e-8
    n_t = 2048
    f_signal_hz = 3e6  # 3 MHz — well above the 0.5 MHz cutoff
    f_low_hz = 0.5e6

    t = np.arange(n_t) * dt
    dc = 5.0 * np.ones(n_t)
    sine = np.sin(2 * np.pi * f_signal_hz * t)
    data = (dc + sine)[np.newaxis, :]

    result = apply_filter(data, dt, f_low_hz=f_low_hz, use_gpu=False)

    # DC (mean) should be nearly zero
    assert abs(result[0].mean()) < 0.05

    # Sine amplitude should be close to 1.0 — check via RMS
    rms = np.sqrt(np.mean(result[0] ** 2))
    assert 0.6 < rms < 1.1, f"Expected RMS ~0.7, got {rms}"


def test_bandpass_filter_gain_matches_mask():
    """Gold-standard performance test: measured per-frequency gain must match the mask.

    The mask returned by ``_build_frequency_mask`` is the specification.  We feed
    pure sinusoids at two frequencies (one well inside the band, one well outside),
    run the filter, then recover the actual gain at each frequency via FFT.  The
    measured gain must agree with the mask value at that bin to within 1 %.
    """
    dt = 1e-8
    n_t = 8192  # large n for clean spectral resolution
    f_inband_hz = 3e6  # 3 MHz — flat passband, mask ≈ 1
    f_outband_hz = 0.2e6  # 0.2 MHz — well into stopband, mask ≈ 0

    f_low_hz = 1e6
    f_high_hz = 5e6

    t = np.arange(n_t) * dt
    data = (np.sin(2 * np.pi * f_inband_hz * t) + np.sin(2 * np.pi * f_outband_hz * t))[
        np.newaxis,
        :,
    ]

    result = apply_filter(data, dt, f_low_hz=f_low_hz, f_high_hz=f_high_hz, use_gpu=False)

    # --- gold standard: mask values at the exact test frequencies ---
    freqs = np.fft.rfftfreq(n_t, d=dt)
    mask = _build_frequency_mask(n_t, dt, f_low_hz=f_low_hz, f_high_hz=f_high_hz)

    idx_inband = int(np.argmin(np.abs(freqs - f_inband_hz)))
    idx_outband = int(np.argmin(np.abs(freqs - f_outband_hz)))

    expected_gain_inband = mask[idx_inband]  # should be ≈ 1.0
    expected_gain_outband = mask[idx_outband]  # should be ≈ 0.0

    # --- measured gains via amplitude spectrum ---
    input_spec = np.abs(np.fft.rfft(data[0]))
    output_spec = np.abs(np.fft.rfft(result[0]))

    measured_gain_inband = output_spec[idx_inband] / input_spec[idx_inband]
    measured_gain_outband = output_spec[idx_outband] / input_spec[idx_outband]

    tol = 0.01  # 1 % absolute tolerance on gain

    assert abs(measured_gain_inband - expected_gain_inband) < tol, (
        f"In-band gain {measured_gain_inband:.4f} deviates from mask {expected_gain_inband:.4f}"
    )
    assert abs(measured_gain_outband - expected_gain_outband) < tol, (
        f"Out-of-band gain {measured_gain_outband:.4f} deviates from mask "
        f"{expected_gain_outband:.4f}"
    )


# ---------------------------------------------------------------------------
# Validation via solver.run() — no GPU, no binary needed
# ---------------------------------------------------------------------------


def test_highpass_and_bandpass_raises():
    """solver.run() must raise ValueError when both filter params are set."""
    grid = MagicMock(spec=fullwave.Grid)
    grid.is_3d = False
    grid.ppw = 4
    grid.nx = 100
    grid.ny = 100
    grid.nz = 1
    grid.dt = 1e-8

    medium = MagicMock(spec=fullwave.Medium)
    medium.n_relaxation_mechanisms = 2
    medium.use_isotropic_relaxation = True

    source = MagicMock(spec=fullwave.Source)
    sensor = MagicMock(spec=fullwave.Sensor)

    fake_bin = MagicMock()
    fake_bin.exists.return_value = True

    # Bypass __init__ entirely, just test the validation in run()
    solver = object.__new__(Solver)
    solver.run_on_memory = False
    solver.work_dir = MagicMock()
    solver.grid = grid
    solver.medium = medium
    solver.is_3d = False
    solver.use_gpu = False
    solver.use_exponential_attenuation = False
    solver.use_isotropic_relaxation = True
    solver.n_relax_mechanisms = 2
    solver.source = source
    solver.sensor = sensor
    solver.transducer = None
    solver.path_fullwave_simulation_bin = fake_bin
    solver.cuda_device_id = None
    solver.use_pml = True
    solver.save_gpu_memory = False

    pml_builder = MagicMock()
    solver.pml_builder = pml_builder
    solver.fullwave_launcher = MagicMock()

    with pytest.raises(ValueError, match="cannot both be specified"):
        solver.run(
            highpass_cutoff_mhz=0.5,
            bandpass_cutoff_mhz=(1.0, 5.0),
        )


def test_filter_requires_load_results():
    """Passing a filter option with load_results=False must raise ValueError."""
    grid = MagicMock(spec=fullwave.Grid)
    grid.is_3d = False
    grid.ppw = 4
    grid.nx = 100
    grid.ny = 100
    grid.nz = 1
    grid.dt = 1e-8

    medium = MagicMock(spec=fullwave.Medium)
    medium.n_relaxation_mechanisms = 2
    medium.use_isotropic_relaxation = True

    source = MagicMock(spec=fullwave.Source)
    sensor = MagicMock(spec=fullwave.Sensor)

    solver = object.__new__(Solver)
    solver.run_on_memory = False
    solver.work_dir = MagicMock()
    solver.grid = grid
    solver.medium = medium
    solver.is_3d = False
    solver.use_gpu = False
    solver.use_exponential_attenuation = False
    solver.use_isotropic_relaxation = True
    solver.n_relax_mechanisms = 2
    solver.source = source
    solver.sensor = sensor
    solver.transducer = None
    solver.path_fullwave_simulation_bin = MagicMock()
    solver.cuda_device_id = None
    solver.use_pml = True
    solver.save_gpu_memory = False
    solver.pml_builder = MagicMock()
    solver.fullwave_launcher = MagicMock()

    with pytest.raises(ValueError, match="load_results=True"):
        solver.run(
            highpass_cutoff_mhz=0.5,
            load_results=False,
        )
