"""Module for generating pulse signals used in the Fullwave simulation."""

import numpy as np
from numpy.typing import NDArray


def gaussian_modulated_sinusoidal_signal(
    nt: int,
    duration: float,
    ncycles: int,
    drop_off: int,
    f0: float,
    p0: float,
    delay_sec: float = 0.0,
    i_layer: int | None = None,
    dt_for_layer_delay: float | None = None,
    cfl_for_layer_delay: float | None = None,
    *,
    dtype: np.dtype = np.float64,
) -> NDArray[np.float64]:
    """Generate Gaussian-modulated sinusoidal signal.

    Parameters
    ----------
    nt: int
        Number of time samples of the simulation.
    duration: float
        Total duration of the simulation.
    ncycles: int
        Number of cycles in the pulse.
    drop_off: int
        Controls the pulse decay.
    f0: float
        Frequency of the pulse.
    p0: float
        Amplitude scaling factor.
    delay_sec: float
        Delay in seconds. Default is 0.0.
    i_layer: int
        Index of the layer where the source is located. Default is None.
        This variable is used to shift the pulse signal in time
        so that the signal is emmitted within the transducer layer correctly.
    dt_for_layer_delay: float
        Time step of the simulation. Default is None.
        This variable is used to shift the pulse signal in time
        so that the signal is emmitted within the transducer layer correctly.
    cfl_for_layer_delay: float
        Courant-Friedrichs-Lewy number. Default is None.
        This variable is used to shift the pulse signal in time
        so that the signal is emmitted within the transducer layer correctly.
    dtype: data-type
        Desired data-type for the output array. Default is np.float64.

    Returns
    -------
    NDArray[np.float64]: The generated pulse signal.


    """
    # Build time array
    dt = duration / nt
    t_offset = ncycles / f0 + delay_sec

    phase_correction = 0.0
    if i_layer is not None:
        if dt_for_layer_delay is None:
            error_msg = "dt_for_layer_delay must be provided if i_layer is provided"
            raise ValueError(error_msg)
        if cfl_for_layer_delay is None:
            error_msg = "cfl_for_layer_delay must be provided if i_layer is provided"
            raise ValueError(error_msg)
        layer_delay_sec = (dt_for_layer_delay / cfl_for_layer_delay) * i_layer
        t_offset += layer_delay_sec
        # Correct carrier phase for fractional-sample delays.
        # A continuous time shift of a fractional number of samples changes
        # the carrier phase at each discrete sample point, causing sign
        # inversion and wrong peak positions. Remove the fractional phase
        # offset so the carrier stays on the same discrete phase grid as
        # the base (i_layer=0) signal.
        delay_samples = layer_delay_sec / dt
        fractional_samples = delay_samples - round(delay_samples)
        phase_correction = 2.0 * np.pi * f0 * fractional_samples * dt

    t = np.arange(nt, dtype=dtype) * dt - t_offset

    omega0 = 2.0 * np.pi * f0
    w_t = t * omega0

    # Compute envelope
    coeff = 1.05 / (ncycles * np.pi)
    a_sq = (coeff * w_t) ** 2

    # Fast path for common drop_off values
    if drop_off == 1:
        env = np.exp(-a_sq)
    elif drop_off == 2:
        env = np.exp(-a_sq * a_sq)
    else:
        env = np.exp(-(a_sq**drop_off))

    # Compute final signal with phase-corrected carrier
    return env * np.sin(w_t + phase_correction) * p0
