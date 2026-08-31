"""The excitation and the transmit delays a transducer emits.

`Transducer.plane_wave`, `Transducer.focus`, `Transducer.diverging` and
`Transducer.synthetic_aperture` are the interface a caller uses. This module holds
the pulse those methods emit and the delays they apply.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from fullwave.utils import pulse as pulse_utils

if TYPE_CHECKING:
    from numpy.typing import NDArray

    import fullwave

logger = logging.getLogger("__main__." + __name__)


@dataclass(frozen=True)
class Pulse:
    """The excitation one element emits, with its unit.

    Attributes
    ----------
    pressure : float
        The amplitude of one element [Pa].
    cycles : float
        How many cycles the excitation holds [-].
    drop_off : float
        How fast the envelope falls [-].
    center_hertz : float | None
        The center frequency [Hz]. None takes the grid's own frequency.

    """

    pressure: float = 1.0e5
    cycles: float = 2.0
    drop_off: float = 2.0
    center_hertz: float | None = None


def element_centers_m(
    geometry: fullwave.TransducerGeometry,
    step_m: float,
) -> NDArray[np.float64]:
    """Return the center of each element, in meters.

    Parameters
    ----------
    geometry : fullwave.TransducerGeometry
        The transducer geometry.
    step_m : float
        The grid step [m].

    Returns
    -------
    NDArray[np.float64]
        One row for each element, of shape [number_elements, ndim].

    """
    coords = np.asarray(geometry._source_coords, dtype=float)
    identifiers = np.asarray(geometry._source_ids)
    centers = np.zeros((geometry.number_elements, coords.shape[1]))
    for element in range(1, geometry.number_elements + 1):
        chosen = identifiers == element
        if chosen.any():
            centers[element - 1] = coords[chosen].mean(axis=0)
    return centers * step_m


def plane_wave_delays(
    centers_m: NDArray[np.float64],
    angle_deg: float,
    sound_speed: float,
    elevation_angle_deg: float = 0.0,
) -> NDArray[np.float64]:
    """Return the transmit delay of each element for one steered plane wave.

    The wave front is a plane, so the delay is the projection of the element
    position onto the steering direction. The earliest element fires at zero.

    Parameters
    ----------
    centers_m : NDArray[np.float64]
        The center of each element, in meters.
    angle_deg : float
        The steering angle in the lateral plane [degrees].
    sound_speed : float
        The sound speed [m/s].
    elevation_angle_deg : float, optional
        The steering angle in the elevation plane [degrees]. Three dimensions only.

    Returns
    -------
    NDArray[np.float64]
        The delay of each element [s].

    """
    centers = np.asarray(centers_m, dtype=float)
    travel = centers[:, 1] * math.sin(math.radians(angle_deg))
    if centers.shape[1] > 2 and elevation_angle_deg != 0.0:
        travel = travel + centers[:, 2] * math.sin(math.radians(elevation_angle_deg))
    delay = travel / sound_speed
    return delay - delay.min()


def focused_delays(
    centers_m: NDArray[np.float64],
    focus_m: NDArray[np.float64],
    sound_speed: float,
) -> NDArray[np.float64]:
    """Return the transmit delay of each element for a focus at one point.

    The element farthest from the focus fires first, so every element reaches the
    focus together.

    Parameters
    ----------
    centers_m : NDArray[np.float64]
        The center of each element, in meters.
    focus_m : NDArray[np.float64]
        The focus, in meters, of the same length as one center.
    sound_speed : float
        The sound speed [m/s].

    Returns
    -------
    NDArray[np.float64]
        The delay of each element [s].

    """
    centers = np.asarray(centers_m, dtype=float)
    focus = np.asarray(focus_m, dtype=float)
    distance = np.sqrt(((centers - focus[None, :]) ** 2).sum(axis=1))
    return (distance.max() - distance) / sound_speed


def diverging_delays(
    centers_m: NDArray[np.float64],
    virtual_source_m: NDArray[np.float64],
    sound_speed: float,
) -> NDArray[np.float64]:
    """Return the transmit delay of each element for a wave from a virtual source.

    The virtual source sits behind the aperture, so the element nearest to it
    fires first and the wave spreads.

    Parameters
    ----------
    centers_m : NDArray[np.float64]
        The center of each element, in meters.
    virtual_source_m : NDArray[np.float64]
        The virtual source, in meters, of the same length as one center.
    sound_speed : float
        The sound speed [m/s].

    Returns
    -------
    NDArray[np.float64]
        The delay of each element [s].

    """
    centers = np.asarray(centers_m, dtype=float)
    source = np.asarray(virtual_source_m, dtype=float)
    distance = np.sqrt(((centers - source[None, :]) ** 2).sum(axis=1))
    return (distance - distance.min()) / sound_speed


def tukey_weights(elements: int, alpha: float) -> NDArray[np.float64]:
    """Return the transmit weight of each element.

    A taper softens the edge of the aperture, which lowers the ripple a hard edge
    diffracts.

    Parameters
    ----------
    elements : int
        How many elements the aperture holds.
    alpha : float
        The Tukey parameter. 0 gives no taper and 1 gives a full cosine.

    Returns
    -------
    NDArray[np.float64]
        One weight for each element.

    """
    if alpha <= 0.0:
        return np.ones(elements)
    from scipy.signal.windows import tukey  # noqa: PLC0415

    return np.asarray(tukey(elements, min(alpha, 1.0)), dtype=float)


def layer_of_each_pixel(
    coords: NDArray[np.int64],
    identifiers: NDArray[np.int64],
) -> NDArray[np.int64]:
    """Return how many rows deep each source pixel sits inside its own element.

    The depth is measured inside each element, so an element on an arc gives the
    same answer as one on a line.

    Parameters
    ----------
    coords : NDArray[np.int64]
        The grid position of each source pixel.
    identifiers : NDArray[np.int64]
        The element each source pixel belongs to.

    Returns
    -------
    NDArray[np.int64]
        The row of each pixel inside its element, counted from the face.

    """
    axial = np.asarray(coords)[:, 0]
    identifiers = np.asarray(identifiers)
    layer = np.zeros(len(axial), dtype=np.int64)
    for element in np.unique(identifiers):
        chosen = identifiers == element
        layer[chosen] = axial[chosen] - axial[chosen].min()
    return layer


def signal_of(
    grid: fullwave.Grid,
    coords: NDArray[np.int64],
    identifiers: NDArray[np.int64],
    pulse: Pulse,
    delays_s: NDArray[np.float64],
    weights: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Return the excitation of every source pixel.

    Every pixel of one element carries that element's delay and weight. A pixel
    deeper inside the element carries one further grid step of travel, so the rows
    of a thick element add together as the wave moves forward.

    Parameters
    ----------
    grid : fullwave.Grid
        The grid, read for the step count, the time step and the Courant number.
    coords : NDArray[np.int64]
        The grid position of each source pixel.
    identifiers : NDArray[np.int64]
        The element each source pixel belongs to, counted from one.
    pulse : Pulse
        The excitation one element emits.
    delays_s : NDArray[np.float64]
        The delay of each element [s].
    weights : NDArray[np.float64]
        The weight of each element [-].

    Returns
    -------
    NDArray[np.float64]
        The excitation, of shape [n_pixels, grid.nt].

    """
    identifiers = np.asarray(identifiers)
    layers = layer_of_each_pixel(coords, identifiers)
    center_hertz = grid.f0 if pulse.center_hertz is None else pulse.center_hertz

    signal = np.zeros((len(identifiers), grid.nt))
    for index, element in enumerate(identifiers):
        weight = float(weights[element - 1])
        if weight == 0.0:
            continue
        signal[index] = pulse_utils.gaussian_modulated_sinusoidal_signal(
            nt=grid.nt,
            f0=center_hertz,
            duration=grid.duration,
            ncycles=pulse.cycles,
            drop_off=pulse.drop_off,
            p0=pulse.pressure * weight,
            i_layer=int(layers[index]),
            dt_for_layer_delay=grid.dt,
            cfl_for_layer_delay=grid.cfl,
            delay_sec=float(delays_s[element - 1]),
        )
    return signal
