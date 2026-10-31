"""Unit tests for fullwave.utils.signal_filter — no GPU required."""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
from scipy import signal

import fullwave
from fullwave.solver.solver import Solver
from fullwave.utils import signal_filter
from fullwave.utils.signal_filter import apply_filter

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_signal(dt: float, n_t: int, frequencies_hz: list[float]) -> np.ndarray:
    """Sum of pure sinusoids, shape [1, n_t]."""
    t = np.arange(n_t) * dt
    sig = sum(np.sin(2 * np.pi * f * t) for f in frequencies_hz)
    return np.asarray(sig, dtype=np.float64)[np.newaxis, :]


# ---------------------------------------------------------------------------
# apply_filter — functional tests (CPU only via use_gpu=False)
# ---------------------------------------------------------------------------


def _middle(x: np.ndarray) -> np.ndarray:
    n_t = x.shape[-1]
    return x[..., n_t // 4 : 3 * n_t // 4]


def _amplitude(x: np.ndarray) -> float:
    return float(np.sqrt(2.0) * np.sqrt(np.mean(x**2)))


def test_highpass_dc_gives_zero_over_whole_record():
    """A constant input must give about zero everywhere, with no slow sinusoid."""
    dt = 1e-8
    data = 20000.0 * np.ones((2, 4096))
    result = apply_filter(data, dt, f_low_hz=0.05e6, use_gpu=False)
    assert np.max(np.abs(result)) < 1e-6 * 20000.0


@pytest.mark.parametrize(
    "kwargs", [{"f_low_hz": 1e6}, {"f_high_hz": 8e6}, {"f_low_hz": 1e6, "f_high_hz": 8e6}]
)
def test_passband_tone_changes_less_than_0p1_db_in_middle(kwargs):
    dt = 1e-8
    data = _make_signal(dt, 8192, [4e6])
    result = apply_filter(data, dt, use_gpu=False, **kwargs)
    gain_db = 20 * np.log10(_amplitude(_middle(result)) / _amplitude(_middle(data)))
    assert abs(gain_db) < 0.1


def test_stopband_tone_is_attenuated():
    dt = 1e-8
    n_t = 8192
    low = apply_filter(_make_signal(dt, n_t, [0.1e6]), dt, f_low_hz=1e6, use_gpu=False)
    high = apply_filter(_make_signal(dt, n_t, [20e6]), dt, f_high_hz=5e6, use_gpu=False)
    assert _amplitude(_middle(low)) < 0.01
    assert _amplitude(_middle(high)) < 0.05


def test_no_wraparound_from_end_of_record():
    """An impulse near the end of the record must leave no energy at the start."""
    dt = 1e-8
    n_t = 4096
    data = np.zeros((1, n_t))
    data[0, n_t - 20] = 20000.0
    for kwargs in ({"f_low_hz": 1.3e6}, {"f_high_hz": 5e6}, {"f_low_hz": 1e6, "f_high_hz": 5e6}):
        result = apply_filter(data, dt, use_gpu=False, **kwargs)
        assert np.max(np.abs(result[0, : n_t // 4])) < 1e-3 * 20000.0


def test_order_keyword_sets_steepness():
    dt = 1e-8
    data = _make_signal(dt, 8192, [0.5e6])
    soft = apply_filter(data, dt, f_low_hz=1e6, order=1, use_gpu=False)
    steep = apply_filter(data, dt, f_low_hz=1e6, order=6, use_gpu=False)
    assert _amplitude(_middle(steep)) < _amplitude(_middle(soft))


def _low_frequency_floor_db(x: np.ndarray, dt: float, f_max_hz: float) -> float:
    """Largest spectral amplitude below ``f_max_hz`` in dB re the spectral peak."""
    freqs = np.fft.rfftfreq(x.shape[-1], d=dt)
    amp = np.abs(np.fft.rfft(x[0]))
    return float(20 * np.log10(amp[(freqs > 0) & (freqs < f_max_hz)].max() / amp.max()))


def test_pulse_at_first_sample_adds_no_floor_beyond_the_record_truncation():
    """A large pulse at sample 0 must not raise the low-frequency floor above the ideal filter.

    The ideal output filters a record padded with zeros on both sides and then crops it.
    Cropping a non-causal filter response already leaves a floor in the spectrum of the
    record, so the reference is that floor and not -inf.  Odd reflection (the SciPy default)
    mirrors the pulse into the record and lands about 10 dB above it.
    """
    dt = 1e-7
    n_t = 4096
    f_low_hz = 0.5e6
    data = _make_signal(dt, n_t, [2e6]) * 1000.0
    data[0, 0] += 2e5

    result = apply_filter(data, dt, f_low_hz=f_low_hz, use_gpu=False)

    sos = signal.butter(4, f_low_hz, btype="highpass", fs=1.0 / dt, output="sos")
    padded = np.pad(data, ((0, 0), (n_t, n_t)))
    ideal = signal.sosfiltfilt(sos, padded, axis=-1, padtype=None)[:, n_t : 2 * n_t]

    floor = _low_frequency_floor_db(result, dt, 0.4 * f_low_hz)
    ideal_floor = _low_frequency_floor_db(ideal, dt, 0.4 * f_low_hz)
    assert floor < ideal_floor + 1.0


def _cupyx_signal_and_gpu_data_or_skip(data_cpu):
    """Return ``cupyx.scipy.signal`` and ``data_cpu`` on the GPU, or skip the test.

    Any exception counts, because a missing CUDA library raises ImportError (not
    ModuleNotFoundError) at import or at the first call, and ``importorskip`` lets it through.
    """
    try:
        import cupy as cp
        from cupyx.scipy import signal as cupyx_signal

        data = cp.asarray(data_cpu)
        sos = cp.asarray(signal.butter(4, [0.5e6, 3e6], btype="bandpass", fs=1e7, output="sos"))
        cupyx_signal.sosfiltfilt(sos, data, axis=-1, padtype="even")
    except Exception as error:
        pytest.skip(f"GPU filtering is unusable on this machine: {error}")
    return cupyx_signal, data


def test_gpu_matches_cpu_and_takes_the_cupyx_branch(monkeypatch):
    """With use_gpu=True the cupyx filter runs, and it agrees with SciPy to 1e-9 of the peak."""
    data_cpu = np.random.default_rng(1).standard_normal((4, 2048)) * 2e4
    cupyx_signal, data = _cupyx_signal_and_gpu_data_or_skip(data_cpu)
    monkeypatch.setattr(signal_filter, "_check_cupy", lambda: True)
    calls = []
    original = cupyx_signal.sosfiltfilt

    def recording(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(cupyx_signal, "sosfiltfilt", recording)

    dt = 1e-7
    gpu = apply_filter(data, dt, f_low_hz=0.5e6, f_high_hz=3e6, use_gpu=True)
    cpu = apply_filter(data_cpu, dt, f_low_hz=0.5e6, f_high_hz=3e6, use_gpu=False)

    assert len(calls) == 1
    np.testing.assert_allclose(gpu, cpu, rtol=0, atol=1e-9 * np.abs(cpu).max())


@pytest.fixture
def fresh_gpu_probe():
    """Run the GPU probe again in the test and leave no cached result behind."""
    signal_filter._check_cupy.cache_clear()
    yield
    signal_filter._check_cupy.cache_clear()


def test_gpu_probe_import_error_falls_back_to_cpu_and_warns(fresh_gpu_probe, monkeypatch, caplog):
    """A failing GPU probe (for example missing cuBLAS) gives the CPU result and one warning."""
    import builtins

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "cupy":
            msg = "Failure finding libcublasLt.so.13 or libcublasLt.so.12"
            raise ImportError(msg)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    data = np.random.default_rng(2).standard_normal((2, 1024))
    dt = 1e-7
    with caplog.at_level(logging.WARNING, logger=signal_filter.logger.name):
        first = apply_filter(data, dt, f_low_hz=0.5e6, use_gpu=True)
        second = apply_filter(data, dt, f_low_hz=0.5e6, use_gpu=True)

    expected = apply_filter(data, dt, f_low_hz=0.5e6, use_gpu=False)
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    warnings = [r for r in caplog.records if "libcublasLt" in r.getMessage()]
    assert len(warnings) == 1


def test_gpu_call_failing_after_a_passing_probe_falls_back_to_cpu_and_warns(
    fresh_gpu_probe, monkeypatch, caplog
):
    """A GPU call that fails only for arrays larger than the probe still gives the CPU result.

    The probe passes, so every failing call logs its own warning.
    """
    data = np.random.default_rng(3).standard_normal((4, 2048))
    cupyx_signal, _ = _cupyx_signal_and_gpu_data_or_skip(data)
    original = cupyx_signal.sosfiltfilt

    def failing_for_large_arrays(sos, x, *args, **kwargs):
        if x.size > 64:
            msg = "Failure finding libcublasLt.so.13 or libcublasLt.so.12"
            raise ImportError(msg)
        return original(sos, x, *args, **kwargs)

    monkeypatch.setattr(cupyx_signal, "sosfiltfilt", failing_for_large_arrays)

    dt = 1e-7
    with caplog.at_level(logging.WARNING, logger=signal_filter.logger.name):
        first = apply_filter(data, dt, f_low_hz=0.5e6, use_gpu=True)
        second = apply_filter(data, dt, f_low_hz=0.5e6, use_gpu=True)

    expected = apply_filter(data, dt, f_low_hz=0.5e6, use_gpu=False)
    np.testing.assert_array_equal(first, expected)
    np.testing.assert_array_equal(second, expected)
    assert len([r for r in caplog.records if "libcublasLt" in r.getMessage()]) == 2


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


def test_bandpass_gain_is_half_at_cutoffs():
    """The forward-backward Butterworth gain is 0.5 at each cut-off frequency."""
    dt = 1e-8
    f_low_hz, f_high_hz = 1e6, 5e6
    for f_hz in (f_low_hz, f_high_hz):
        data = _make_signal(dt, 16384, [f_hz])
        result = apply_filter(data, dt, f_low_hz=f_low_hz, f_high_hz=f_high_hz, use_gpu=False)
        gain = _amplitude(_middle(result)) / _amplitude(_middle(data))
        assert abs(gain - 0.5) < 0.02


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
