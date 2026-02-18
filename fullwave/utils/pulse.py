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

    if i_layer is not None:
        if dt_for_layer_delay is None:
            error_msg = "dt_for_layer_delay must be provided if i_layer is provided"
            raise ValueError(error_msg)
        if cfl_for_layer_delay is None:
            error_msg = "cfl_for_layer_delay must be provided if i_layer is provided"
            raise ValueError(error_msg)
        t_offset += (dt_for_layer_delay / cfl_for_layer_delay) * i_layer

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

    # Compute final signal
    return env * np.sin(w_t) * p0
