"""FFT-based frequency-domain filtering for sensor data.

GPU backend: CuPy when available; falls back silently to NumPy.
No new hard dependencies — CuPy is already listed under the ``examples`` optional extra.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger("__main__." + __name__)

# Module-level cache to avoid repeated import overhead
_CUPY_AVAILABLE: bool | None = None


def _check_cupy() -> bool:
    """Return True if CuPy is importable; result is cached after the first call."""
    global _CUPY_AVAILABLE  # noqa: PLW0603
    if _CUPY_AVAILABLE is None:
        try:
            import cupy  # noqa: F401, PLC0415

            _CUPY_AVAILABLE = True
        except ImportError:
            _CUPY_AVAILABLE = False
    return _CUPY_AVAILABLE


def _build_frequency_mask(
    n_fft: int,
    dt: float,
    f_low_hz: float | None = None,
    f_high_hz: float | None = None,
    taper_ratio: float = 0.1,
) -> NDArray[np.float64]:
    """Build a frequency-domain gain mask with cosine (Hann) tapers.

    Parameters
    ----------
    n_fft : int
        FFT length (number of time samples before zero-padding, i.e. ``n_t``).
    dt : float
        Simulation time step in seconds.
    f_low_hz : float | None
        High-pass cut-off frequency in Hz.  Frequencies below this value are
        attenuated.  The mask transitions smoothly from 0 to 1 in a window of
        width ``f_low_hz * taper_ratio`` centred at ``f_low_hz``.
    f_high_hz : float | None
        Low-pass cut-off frequency in Hz.  Frequencies above this value are
        attenuated.  The mask transitions smoothly from 1 to 0 in a window of
        width ``f_high_hz * taper_ratio`` centred at ``f_high_hz``.
    taper_ratio : float
        Fractional width of each cosine taper relative to its centre frequency.
        Default is 0.1 (10 %).

    Returns
    -------
    NDArray[np.float64]
        Frequency-domain gain mask of shape ``[n_fft // 2 + 1]``.

    """
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    mask = np.ones(len(freqs), dtype=np.float64)

    if f_low_hz is not None:
        half_width = f_low_hz * taper_ratio / 2.0
        f_start = f_low_hz - half_width
        f_end = f_low_hz + half_width
        width = f_end - f_start  # == f_low_hz * taper_ratio

        in_taper = (freqs >= f_start) & (freqs <= f_end)
        below_taper = freqs < f_start

        mask[below_taper] = 0.0
        mask[in_taper] = 0.5 * (1.0 - np.cos(np.pi * (freqs[in_taper] - f_start) / width))

    if f_high_hz is not None:
        half_width = f_high_hz * taper_ratio / 2.0
        f_start = f_high_hz - half_width
        f_end = f_high_hz + half_width
        width = f_end - f_start  # == f_high_hz * taper_ratio

        in_taper = (freqs >= f_start) & (freqs <= f_end)
        above_taper = freqs > f_end

        lp_taper = np.ones(len(freqs), dtype=np.float64)
        lp_taper[in_taper] = 0.5 * (1.0 + np.cos(np.pi * (freqs[in_taper] - f_start) / width))
        lp_taper[above_taper] = 0.0
        mask *= lp_taper

    return mask


def apply_filter(
    data: NDArray[np.float64],
    dt: float,
    f_low_hz: float | None = None,
    f_high_hz: float | None = None,
    taper_ratio: float = 0.1,
    *,
    use_gpu: bool = True,
) -> NDArray[np.float64]:
    """Apply a frequency-domain filter to sensor data.

    The filter is built as a cosine-tapered gain mask (see :func:`_build_frequency_mask`).
    When CuPy is available and ``use_gpu=True``, the FFT operations run on the GPU
    for maximum throughput; otherwise NumPy is used transparently.

    Parameters
    ----------
    data : NDArray[np.float64]
        Sensor time traces, shape ``[n_sensors, n_t]``.
    dt : float
        Simulation time step in seconds.
    f_low_hz : float | None
        High-pass edge frequency in Hz.  Pass ``None`` to skip high-passing.
    f_high_hz : float | None
        Low-pass edge frequency in Hz.  Pass ``None`` to skip low-passing.
    taper_ratio : float
        Fractional taper width relative to each cut-off frequency.  Default 0.1.
    use_gpu : bool
        If ``True`` (default), attempt to use CuPy for GPU-accelerated FFTs.
        Falls back to NumPy silently if CuPy is unavailable.

    Returns
    -------
    NDArray[np.float64]
        Filtered data, same shape as ``data``.

    """
    n_t = data.shape[1]
    mask = _build_frequency_mask(
        n_t,
        dt,
        f_low_hz=f_low_hz,
        f_high_hz=f_high_hz,
        taper_ratio=taper_ratio,
    )

    if use_gpu and _check_cupy():
        import cupy as cp  # noqa: PLC0415

        logger.debug("apply_filter: using CuPy GPU backend")
        data_gpu = cp.asarray(data, dtype=cp.float64)
        mask_gpu = cp.asarray(mask, dtype=cp.float64)
        spec = cp.fft.rfft(data_gpu, axis=1)
        spec *= mask_gpu[cp.newaxis, :]
        filtered = cp.fft.irfft(spec, n=n_t, axis=1)
        return cp.asnumpy(filtered)

    logger.debug("apply_filter: using NumPy CPU backend")
    spec = np.fft.rfft(data, axis=1)
    spec *= mask[np.newaxis, :]
    return np.fft.irfft(spec, n=n_t, axis=1)
