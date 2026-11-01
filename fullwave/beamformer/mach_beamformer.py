"""Delay-and-sum beamformer on the GPU, built on mach.

mach (https://github.com/Forest-Neurotech/mach) is an open-source GPU beamformer with an
optimized delay-and-sum CUDA kernel. mach, pymust and cupy come with the optional
extra ``fullwave25[mach]``. This module imports without them; ``MachBeamformer.beamform``
raises an ImportError that names the extra.

Conventions
-----------
Positions are ``(lateral, elevation, depth)`` in meters. Depth 0 is the probe face
(``Transducer.face_row_px``), and depth grows into the medium. A flat array has its
receivers one grid row behind the face, so their depth is ``-dx``.

Transmit arrival times are in seconds, measured from the moment the earliest firing
element has its pulse at the probe face. ``time_zero_s`` is that moment in the clock of
the recorded traces, whose first sample is the first time step of the simulation.
"""

from __future__ import annotations

import functools
import importlib
import logging
from typing import TYPE_CHECKING

import numpy as np
from scipy.signal import hilbert

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import NDArray

    import fullwave

logger = logging.getLogger("__main__." + __name__)

MACH_INSTALL_HINT = (
    "MachBeamformer needs mach-beamform, pymust and cupy. "
    "Install them with: pip install 'fullwave25[mach]'"
)


@functools.cache
def _load_mach_modules() -> tuple[ModuleType, ModuleType, ModuleType]:
    """Import cupy, pymust and mach.experimental, or raise a clear ImportError."""
    try:
        cupy = importlib.import_module("cupy")
        pymust = importlib.import_module("pymust")
        experimental = importlib.import_module("mach.experimental")
    except ImportError as error:
        logger.error(MACH_INSTALL_HINT)
        raise ImportError(MACH_INSTALL_HINT) from error
    return cupy, pymust, experimental


def plane_wave_arrival_s(
    points_m: NDArray[np.float64],
    element_positions_m: NDArray[np.float64],
    *,
    angle_deg: float,
    sound_speed: float,
    elevation_angle_deg: float = 0.0,
) -> NDArray[np.float64]:
    """Return the arrival time of a steered plane wave at each point.

    Same steering as ``Transducer.plane_wave``: the element delay is the projection of
    its position on the steering direction, and the earliest element fires at zero.

    Parameters
    ----------
    points_m : NDArray[np.float64]
        Points of shape [n_points, 3] (lateral, elevation, depth) [m].
    element_positions_m : NDArray[np.float64]
        Element positions of shape [n_elements, 3] at the probe face [m].
    angle_deg : float
        Steering angle in the lateral plane [degrees].
    sound_speed : float
        Sound speed [m/s].
    elevation_angle_deg : float, optional
        Steering angle in the elevation plane [degrees].

    Returns
    -------
    NDArray[np.float64]
        Arrival time at each point [s], shape [n_points].

    """
    sin_lateral = np.sin(np.radians(angle_deg))
    sin_elevation = np.sin(np.radians(elevation_angle_deg))
    axial = np.sqrt(1.0 - sin_lateral**2 - sin_elevation**2)
    direction = np.array([sin_lateral, sin_elevation, axial])
    points = np.asarray(points_m, dtype=float)
    elements = np.asarray(element_positions_m, dtype=float)
    return (points @ direction - (elements @ direction).min()) / sound_speed


def focused_arrival_s(
    points_m: NDArray[np.float64],
    element_positions_m: NDArray[np.float64],
    *,
    focus_m: NDArray[np.float64],
    sound_speed: float,
) -> NDArray[np.float64]:
    """Return the arrival time of a transmit focused at one point.

    Same timing as ``Transducer.focus``: the element farthest from the focus fires at
    zero and every element reaches the focus together. Before the focus the wave
    converges, after the focus it diverges.

    Parameters
    ----------
    points_m : NDArray[np.float64]
        Points of shape [n_points, 3] (lateral, elevation, depth) [m].
    element_positions_m : NDArray[np.float64]
        Element positions of shape [n_elements, 3] at the probe face [m].
    focus_m : NDArray[np.float64]
        Focus (lateral, elevation, depth) [m]. The depth must be positive.
    sound_speed : float
        Sound speed [m/s].

    Returns
    -------
    NDArray[np.float64]
        Arrival time at each point [s], shape [n_points].

    Raises
    ------
    ValueError
        If the focus is not in front of the probe face.

    """
    focus = np.asarray(focus_m, dtype=float)
    if focus[2] <= 0.0:
        error_msg = "The focus must be in front of the probe face (positive depth)."
        logger.error(error_msg)
        raise ValueError(error_msg)
    points = np.asarray(points_m, dtype=float)
    elements = np.asarray(element_positions_m, dtype=float)
    farthest_m = np.linalg.norm(elements - focus, axis=1).max()
    to_focus_m = np.linalg.norm(points - focus, axis=1)
    before_focus = points[:, 2] < focus[2]
    return (farthest_m + np.where(before_focus, -to_focus_m, to_focus_m)) / sound_speed


def synthetic_aperture_arrival_s(
    points_m: NDArray[np.float64],
    element_positions_m: NDArray[np.float64],
    *,
    element: int,
    sound_speed: float,
) -> NDArray[np.float64]:
    """Return the arrival time of a spherical wave from one element.

    Parameters
    ----------
    points_m : NDArray[np.float64]
        Points of shape [n_points, 3] (lateral, elevation, depth) [m].
    element_positions_m : NDArray[np.float64]
        Element positions of shape [n_elements, 3] at the probe face [m].
    element : int
        The firing element, counted from one as in ``Transducer.synthetic_aperture``.
    sound_speed : float
        Sound speed [m/s].

    Returns
    -------
    NDArray[np.float64]
        Arrival time at each point [s], shape [n_points].

    Raises
    ------
    ValueError
        If the element is outside the aperture.

    """
    elements = np.asarray(element_positions_m, dtype=float)
    if not 1 <= element <= len(elements):
        error_msg = f"element {element} is outside the aperture of {len(elements)} elements"
        logger.error(error_msg)
        raise ValueError(error_msg)
    points = np.asarray(points_m, dtype=float)
    return np.linalg.norm(points - elements[element - 1], axis=1) / sound_speed


def user_arrival_s(
    arrival_s: NDArray[np.float64],
    *,
    n_points: int,
) -> NDArray[np.float64]:
    """Check arrival times a caller computed, and return them as float64.

    Parameters
    ----------
    arrival_s : NDArray[np.float64]
        Arrival time at each point [s], shape [n_points] or [n_transmit, n_points],
        measured from the moment the earliest firing element is at the probe face.
    n_points : int
        How many image points there are.

    Returns
    -------
    NDArray[np.float64]
        The same times, shape [n_points] or [n_transmit, n_points].

    Raises
    ------
    ValueError
        If the last axis does not hold ``n_points`` values.

    """
    arrival = np.asarray(arrival_s, dtype=float)
    if arrival.ndim not in (1, 2) or arrival.shape[-1] != n_points:
        error_msg = f"arrival_s must have shape [{n_points}] or [n_transmit, {n_points}]"
        logger.error(error_msg)
        raise ValueError(error_msg)
    return arrival


def face_crossing_time_s(transducer: fullwave.Transducer) -> float:
    """Return when the earliest firing element has its pulse at the probe face.

    The method reads the drive and not the recorded traces. The pulse of the earliest
    firing source pixel peaks (envelope of its Hilbert transform) at the drive time
    ``t_peak``. The drive is applied to every source row at once, and the wave seen
    in front of the array leaves the back source row (measured on the 12-row L7-4
    layout: the pulse peak is 29 time steps after the drive peak at the receiver row,
    which is 12 row steps of ``dx / c0``). The pulse is therefore at the face
    ``(face_row_px - back_source_row) * dx / c0`` after ``t_peak``.
    The first sample of the recorded traces is time step 0 of the simulation.

    Parameters
    ----------
    transducer : fullwave.Transducer
        A flat array with its excitation set.

    Returns
    -------
    float
        The time of the pulse peak at the probe face [s].

    """
    signal = np.asarray(transducer.signal, dtype=float)
    driven = np.flatnonzero((signal != 0).any(axis=1))
    peak_steps = np.argmax(np.abs(hilbert(signal[driven], axis=1)), axis=1)
    peak_step = int(peak_steps.min())
    grid = transducer.grid
    back_source_row = int(transducer.source_coords[:, 0].min())
    rows_to_face = transducer.face_row_px - back_source_row
    return peak_step * float(grid.dt) + rows_to_face * float(grid.dx) / float(grid.c0)


class MachBeamformer:
    """Delay-and-sum beamformer that runs on the GPU through mach.

    The traces are converted to IQ with ``pymust.rf2iq`` and nothing else is applied to
    them. ``f_number`` is the standard F = depth / aperture width: mach sums over the
    receivers within a total aperture of ``depth / f_number`` around the image point,
    with a Tukey window of ``tukey_alpha`` across it (0 is rectangular). ``Beamformer``
    instead takes the half width as ``depth / F``, so ``f_number / 2`` here matches it.

    Time zero: depth 0 is the probe face. A transmit arrival time is zero when the
    earliest firing element has its pulse at the face, and ``time_zero_s`` is that
    moment in the clock of the traces. mach starts the traces at ``-time_zero_s``.
    ``from_transducer`` derives ``time_zero_s`` from the drive and the source-to-face
    rows (``face_crossing_time_s``).

    References
    ----------
    If you use this beamformer, please cite mach:
    C. Guan, A. P. Rockhill, M. Sode and G. Pinton, "mach: ultrafast ultrasound
    beamforming", Journal of Medical Imaging 13(6), 062203 (2026),
    https://doi.org/10.1117/1.JMI.13.6.062203
    """

    def __init__(
        self,
        *,
        lateral_position_m: NDArray[np.float64],
        axial_position_m: NDArray[np.float64],
        receiver_positions_m: NDArray[np.float64],
        sound_speed: float,
        center_frequency_hz: float,
        sampling_frequency_hz: float,
        f_number: float = 1.0,
        time_zero_s: float = 0.0,
        tukey_alpha: float = 0.5,
    ) -> None:
        """Initialize the beamformer.

        Parameters
        ----------
        lateral_position_m : NDArray[np.float64]
            Lateral positions of the image columns [m].
        axial_position_m : NDArray[np.float64]
            Depths of the image rows below the probe face [m].
        receiver_positions_m : NDArray[np.float64]
            Receiver positions of shape [n_elements, 3] (lateral, elevation, depth) [m].
        sound_speed : float
            Sound speed used for the delays [m/s].
        center_frequency_hz : float
            Center frequency of the IQ demodulation [Hz].
        sampling_frequency_hz : float
            Sampling frequency of the traces [Hz].
        f_number : float, optional
            Receive f-number F = depth / aperture width, by default 1.0.
        time_zero_s : float, optional
            When the earliest firing element has its pulse at the face, in the clock of
            the traces [s], by default 0.0.
        tukey_alpha : float, optional
            Tukey parameter of the receive aperture window, by default 0.5.

        """
        self.lateral_position_m = np.asarray(lateral_position_m, dtype=float)
        self.axial_position_m = np.asarray(axial_position_m, dtype=float)
        self.receiver_positions_m = np.asarray(receiver_positions_m, dtype=float)
        self.sound_speed = float(sound_speed)
        self.center_frequency_hz = float(center_frequency_hz)
        self.sampling_frequency_hz = float(sampling_frequency_hz)
        self.f_number = float(f_number)
        self.time_zero_s = float(time_zero_s)
        self.tukey_alpha = float(tukey_alpha)

    @classmethod
    def from_transducer(
        cls,
        transducer: fullwave.Transducer,
        *,
        lateral_position_m: NDArray[np.float64],
        axial_position_m: NDArray[np.float64],
        f_number: float = 1.0,
        tukey_alpha: float = 0.5,
    ) -> MachBeamformer:
        """Build the beamformer from a flat array whose excitation is set.

        Lateral 0 is the center of the array. The sound speed is ``grid.c0``, the center
        frequency is the grid frequency, and the sampling follows
        ``transducer.sampling_modulus_time``.

        Parameters
        ----------
        transducer : fullwave.Transducer
            A flat array with its excitation set (the transmit of the traces).
        lateral_position_m : NDArray[np.float64]
            Lateral positions of the image columns [m].
        axial_position_m : NDArray[np.float64]
            Depths of the image rows below the probe face [m].
        f_number : float, optional
            Receive f-number F = depth / aperture width, by default 1.0.
        tukey_alpha : float, optional
            Tukey parameter of the receive aperture window, by default 0.5.

        Returns
        -------
        MachBeamformer
            The beamformer.

        """
        grid = transducer.grid
        centers = transducer.element_id_to_receiver_center
        element_ids = sorted(centers)
        receivers_px = np.array([centers[i] for i in element_ids], dtype=float)
        lateral_center_px = receivers_px[:, 1].mean()
        receivers_m = np.zeros((len(receivers_px), 3))
        receivers_m[:, 0] = (receivers_px[:, 1] - lateral_center_px) * grid.dx
        if receivers_px.shape[1] == 3:
            receivers_m[:, 1] = (receivers_px[:, 2] - receivers_px[:, 2].mean()) * grid.dx
        receivers_m[:, 2] = (receivers_px[:, 0] - transducer.face_row_px) * grid.dx
        return cls(
            lateral_position_m=lateral_position_m,
            axial_position_m=axial_position_m,
            receiver_positions_m=receivers_m,
            sound_speed=float(grid.c0),
            center_frequency_hz=float(grid.f0),
            sampling_frequency_hz=1.0 / (float(grid.dt) * transducer.sampling_modulus_time),
            f_number=f_number,
            time_zero_s=face_crossing_time_s(transducer),
            tukey_alpha=tukey_alpha,
        )

    @property
    def element_positions_m(self) -> NDArray[np.float64]:
        """Return the element positions at the probe face, shape [n_elements, 3] [m]."""
        positions = self.receiver_positions_m.copy()
        positions[:, 2] = 0.0
        return positions

    @property
    def scan_points_m(self) -> NDArray[np.float64]:
        """Return the image points, shape [n_lateral * n_axial, 3], lateral-major [m]."""
        lateral, axial = np.meshgrid(
            self.lateral_position_m,
            self.axial_position_m,
            indexing="ij",
        )
        points = np.zeros((lateral.size, 3))
        points[:, 0] = lateral.ravel()
        points[:, 2] = axial.ravel()
        return points

    def plane_wave_arrival_s(
        self,
        *,
        angle_deg: float,
        elevation_angle_deg: float = 0.0,
    ) -> NDArray[np.float64]:
        """Return the arrival time of a steered plane wave at the image points [s]."""
        return plane_wave_arrival_s(
            self.scan_points_m,
            self.element_positions_m,
            angle_deg=angle_deg,
            sound_speed=self.sound_speed,
            elevation_angle_deg=elevation_angle_deg,
        )

    def focused_arrival_s(self, *, focus_m: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the arrival time of a transmit focused at ``focus_m`` [s]."""
        return focused_arrival_s(
            self.scan_points_m,
            self.element_positions_m,
            focus_m=focus_m,
            sound_speed=self.sound_speed,
        )

    def synthetic_aperture_arrival_s(self, *, element: int) -> NDArray[np.float64]:
        """Return the arrival time of the wave of one element, counted from one [s]."""
        return synthetic_aperture_arrival_s(
            self.scan_points_m,
            self.element_positions_m,
            element=element,
            sound_speed=self.sound_speed,
        )

    def user_arrival_s(self, arrival_s: NDArray[np.float64]) -> NDArray[np.float64]:
        """Check arrival times given by the caller for the image points [s]."""
        return user_arrival_s(arrival_s, n_points=len(self.scan_points_m))

    def rf_to_iq(self, rf: NDArray[np.float64]) -> NDArray[np.complex64]:
        """Convert traces to IQ with ``pymust.rf2iq``, the only processing applied.

        Parameters
        ----------
        rf : NDArray[np.float64]
            Traces of shape [n_elements, n_time] or [n_transmit, n_elements, n_time].

        Returns
        -------
        NDArray[np.complex64]
            IQ traces of shape [n_transmit, n_elements, n_time].

        """
        _, pymust, _ = _load_mach_modules()
        traces = np.asarray(rf)
        if traces.ndim == 2:
            traces = traces[np.newaxis]
        iq = pymust.rf2iq(
            traces.transpose(2, 1, 0),
            Fs=self.sampling_frequency_hz,
            Fc=self.center_frequency_hz,
        )
        return np.ascontiguousarray(iq.transpose(2, 1, 0), dtype=np.complex64)

    def beamform(
        self,
        rf: NDArray[np.float64],
        *,
        tx_arrival_s: NDArray[np.float64],
    ) -> NDArray[np.complex64]:
        """Beamform the traces into one image, summing the transmits coherently.

        Parameters
        ----------
        rf : NDArray[np.float64]
            Traces of shape [n_elements, n_time] for one transmit, or
            [n_transmit, n_elements, n_time].
        tx_arrival_s : NDArray[np.float64]
            Transmit arrival times from one of the ``*_arrival_s`` helpers, shape
            [n_points] for one transmit or [n_transmit, n_points].

        Returns
        -------
        NDArray[np.complex64]
            The IQ image of shape [n_axial, n_lateral]. Its magnitude is the envelope.

        Raises
        ------
        ValueError
            If the transmit count of ``rf`` and ``tx_arrival_s`` differ.

        """
        cupy, _, experimental = _load_mach_modules()
        arrival = np.atleast_2d(self.user_arrival_s(tx_arrival_s))
        iq = self.rf_to_iq(rf)
        if iq.shape[0] != arrival.shape[0]:
            error_msg = "rf and tx_arrival_s hold different numbers of transmits"
            logger.error(error_msg)
            raise ValueError(error_msg)
        channel_data = np.ascontiguousarray(iq[..., np.newaxis])
        flat_image = experimental.beamform(
            channel_data=cupy.asarray(channel_data),
            rx_coords_m=cupy.asarray(self.receiver_positions_m),
            scan_coords_m=cupy.asarray(self.scan_points_m),
            tx_wave_arrivals_s=cupy.asarray(arrival),
            f_number=self.f_number,
            rx_start_s=-self.time_zero_s,
            sampling_freq_hz=self.sampling_frequency_hz,
            sound_speed_m_s=self.sound_speed,
            modulation_freq_hz=self.center_frequency_hz,
            tukey_alpha=self.tukey_alpha,
        )
        image = cupy.asnumpy(flat_image)[:, 0]
        return image.reshape(len(self.lateral_position_m), len(self.axial_position_m)).T
