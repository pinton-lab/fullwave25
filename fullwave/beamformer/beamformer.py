"""Delay-and-sum beamformer."""

import numpy as np
from scipy.signal import hilbert
from tqdm import tqdm


class Beamformer:
    """Delay-and-sum beamformer for linear transducer arrays.

    This is not optimized for performance and is intended for educational
    purposes only.

    For faster implementations, consider using libraries such as
    mach beamformer: https://github.com/Forest-Neurotech/mach

    References:
    - Fullwave2 BMME890 implementation
    - https://github.com/gfpinton/fullwave_bmme890/blob/master/fullwave2_launcher_imaging_planewave.m

    """

    def __init__(
        self,
        c0: float,
        dx: float,
        dt: float,
        *,
        lateral_position_m: np.ndarray,
        axial_position_m: np.ndarray,
        num_elements: int,
        transducer_coordinates: np.ndarray,
        f_number: float = 1.0,
    ) -> None:
        """Initialize the beamformer with simulation and transducer parameters.

        Parameters
        ----------
        c0 : float
            Speed of sound in the medium (m/s).
        dx : float
            Spatial grid spacing (m).
        dt : float
            Temporal grid spacing (s).
        lateral_position_m : np.ndarray
            Lateral positions (m) where the beamformed image will be computed.
        axial_position_m : np.ndarray
            Axial positions (m) where the beamformed image will be computed.
        num_elements : int
            Number of transducer elements.
        transducer_coordinates : np.ndarray
            Coordinates of the transducer elements (m).
        f_number : float, optional
            F-number for aperture calculation, by default 1.0.

        """
        self.c0 = c0
        self.dx = dx
        self.dt = dt
        self.lateral_position_m = lateral_position_m
        self.axial_position_m = axial_position_m
        self.num_elements = num_elements
        self.transducer_coordinates = transducer_coordinates
        self.f_number = f_number

    def run(self, signals: np.ndarray) -> np.ndarray:
        """Perform delay-and-sum beamforming on the input signals.

        Parameters
        ----------
        signals : np.ndarray
            Input signals from the transducer elements.
            Shape: [n_elements, n_time] or [n_elements, n_time, n_transmit].

        Returns
        -------
        np.ndarray
            Beamformed image. Shape: [n_axial, n_lateral, n_transmit].

        """
        # signals: [n_elements, n_time, n_transmit]
        if signals.ndim == 2:
            signals = signals[:, :, np.newaxis]
        n_transmit = signals.shape[2]

        hilbert_signals = hilbert(signals, axis=1)
        beamformed_image = np.zeros(
            (len(self.axial_position_m), len(self.lateral_position_m), n_transmit),
            dtype=np.complex128,
        )

        for i_trasnsmit in range(n_transmit):
            idt_0 = np.argmax(np.abs(hilbert_signals[hilbert_signals.shape[0] // 2]))
            idps = np.empty(
                (len(self.axial_position_m), len(self.lateral_position_m)),
                dtype=object,
            )
            for i_lat, lat in tqdm(
                enumerate(self.lateral_position_m),
                desc="Lateral positions",
                total=len(self.lateral_position_m),
            ):
                for i_axial, axial in tqdm(
                    enumerate(self.axial_position_m),
                    desc="Axial positions",
                    total=len(self.axial_position_m),
                    leave=False,
                ):
                    fcen = np.array(
                        [
                            int(axial / self.dx),
                            int(lat / self.dx + np.mean(self.transducer_coordinates[:, 1])),
                        ],
                    )
                    idx = np.where(
                        np.abs(self.transducer_coordinates[:, 1] - fcen[1])
                        <= fcen[0] / self.f_number,
                    )
                    idx = np.array(idx).flatten()
                    if len(idx) == 0:
                        idps[i_axial, i_lat] = np.array([], dtype=int)
                        continue

                    dd = np.zeros(len(idx), dtype=int)

                    for i in range(len(fcen)):
                        dd += np.round(
                            (self.transducer_coordinates[idx, i] - fcen[i]) ** 2,
                        ).astype(int)
                    dd = np.sqrt(dd)

                    dd = np.round(dd / (self.dt / self.dx * self.c0)).astype(int)
                    dd = dd - dd.min()

                    idt = idt_0 + int(2 * axial / self.c0 / self.dt)

                    # idp: 1D indices into signals flattened array
                    # we convert 2D indices to 1D indices
                    idp = (signals.shape[1] * (idx)) + (idt + dd)

                    idp = idp[(idp > 0) & (idp < signals.shape[0] * signals.shape[1])]

                    idps[i_axial, i_lat] = idp

            hilbert_signals_flattened = hilbert_signals[:, :, i_trasnsmit].ravel()
            for i_lat in tqdm(range(len(self.lateral_position_m)), desc="Beamforming lateral"):
                for i_axial in tqdm(
                    range(len(self.axial_position_m)),
                    desc="Beamforming axial",
                    leave=False,
                ):
                    beamformed_image[i_axial, i_lat, i_trasnsmit] = np.sum(
                        hilbert_signals_flattened[idps[i_axial, i_lat]],
                    )
        return beamformed_image
