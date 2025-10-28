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
) -> NDArray[np.float64]:
    """Generate a pulse signal based on input parameters.

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

    Returns
    -------
    NDArray[np.float64]: The generated pulse signal.

    """
    t = (np.arange(0, nt)) / nt * duration - ncycles / f0
    t = t - delay_sec

    if i_layer:
        assert dt_for_layer_delay, "dt must be provided if i_layer is provided"
        assert cfl_for_layer_delay, "cfl must be provided if i_layer is provided"
        t = t - (dt_for_layer_delay / cfl_for_layer_delay) * i_layer

    omega0 = 2 * np.pi * f0
    return (
        np.multiply(
            np.exp(
                -((1.05 * t * omega0 / (ncycles * np.pi)) ** (2 * drop_off)),
            ),
            np.sin(t * omega0),
        )
        * p0
    )
