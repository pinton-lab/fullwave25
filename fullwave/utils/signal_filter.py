"""Zero-phase Butterworth filtering for sensor data.

GPU backend: ``cupyx.scipy.signal.sosfiltfilt`` when CuPy is available;
falls back silently to SciPy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy import signal

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger("__main__." + __name__)

# Module-level cache to avoid repeated import overhead
_CUPY_AVAILABLE: bool | None = None


def _check_cupy() -> bool:
    """Return True if CuPy and ``cupyx.scipy.signal.sosfiltfilt`` are importable.

    The result is cached after the first call.
    """
    global _CUPY_AVAILABLE  # noqa: PLW0603
    if _CUPY_AVAILABLE is None:
        try:
            import cupy  # noqa: F401, PLC0415
            from cupyx.scipy import signal as cupy_signal  # noqa: PLC0415

            _CUPY_AVAILABLE = hasattr(cupy_signal, "sosfiltfilt")
        except ImportError:
            _CUPY_AVAILABLE = False
    return _CUPY_AVAILABLE


def _design_sos(
    dt: float,
    f_low_hz: float | None,
    f_high_hz: float | None,
    order: int,
) -> NDArray[np.float64] | None:
    """Design Butterworth second-order sections, or return None if no cut-off is given."""
    if f_low_hz is not None and f_high_hz is not None:
        band, btype = [f_low_hz, f_high_hz], "bandpass"
    elif f_low_hz is not None:
        band, btype = f_low_hz, "highpass"
    elif f_high_hz is not None:
        band, btype = f_high_hz, "lowpass"
    else:
        return None
    return signal.butter(order, band, btype=btype, fs=1.0 / dt, output="sos")


def apply_filter(
    data: NDArray[np.float64],
    dt: float,
    f_low_hz: float | None = None,
    f_high_hz: float | None = None,
    *,
    order: int = 4,
    use_gpu: bool = True,
) -> NDArray[np.float64]:
    """Apply a zero-phase Butterworth filter along the time axis of sensor data.

    The filter is a Butterworth filter in second-order sections, applied forward
    and backward with ``sosfiltfilt``.  It has no phase shift, does not wrap
    around the ends of the record, and does not ring like a brick-wall FFT mask.
    Because of the forward-backward pass, the gain at each cut-off frequency is
    -6 dB (0.5).  The ends of the record are extended by even reflection, which
    keeps a constant input at zero output and, unlike the default odd reflection,
    does not mirror a large pulse near the first sample into the record.

    Parameters
    ----------
    data : NDArray[np.float64]
        Sensor time traces, shape ``[n_sensors, n_t]``.
    dt : float
        Simulation time step in seconds.
    f_low_hz : float | None
        High-pass cut-off frequency in Hz.  Pass ``None`` to skip high-passing.
    f_high_hz : float | None
        Low-pass cut-off frequency in Hz.  Pass ``None`` to skip low-passing.
        If both are given, the filter is a band-pass.
    order : int
        Butterworth order of each pass.  Default 4.
    use_gpu : bool
        If ``True`` (default), use ``cupyx.scipy.signal.sosfiltfilt`` when CuPy
        provides it; otherwise the filter runs on the CPU with SciPy.

    Returns
    -------
    NDArray[np.float64]
        Filtered data, same shape as ``data``.

    """
    sos = _design_sos(dt, f_low_hz, f_high_hz, order)
    if sos is None:
        return np.array(data, dtype=np.float64)

    if use_gpu and _check_cupy():
        import cupy as cp  # noqa: PLC0415
        from cupyx.scipy import signal as cupy_signal  # noqa: PLC0415

        logger.debug("apply_filter: using CuPy GPU backend")
        data_gpu = cp.asarray(data, dtype=cp.float64)
        filtered = cupy_signal.sosfiltfilt(cp.asarray(sos), data_gpu, axis=-1, padtype="even")
        return cp.asnumpy(filtered)

    logger.debug("apply_filter: using SciPy CPU backend")
    return signal.sosfiltfilt(sos, np.asarray(data, dtype=np.float64), axis=-1, padtype="even")
