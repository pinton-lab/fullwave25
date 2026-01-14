import numpy as np

from fullwave.beamformer.beamformer import Beamformer


def test_beamformer_delay_and_sum():
    c0 = 1500.0  # m/s
    dx = 3.468468468468468e-05  # m
    dt = 9.009009009009009e-09  # s
    num_elements = 8
    lateral_position = np.linspace(-num_elements * dx / 2, num_elements * dx / 2, num_elements)  # m
    axial_position = np.linspace(0.0, dt * 100, 16)  # m

    transducer_coordinates_y = lateral_position / dx  # in grid units
    transducer_coordinates_y = (transducer_coordinates_y - transducer_coordinates_y.min()).astype(
        int,
    )
    transducer_coordinates_x = np.zeros_like(transducer_coordinates_y)  # at z=0
    transducer_coordinates = np.stack(
        (transducer_coordinates_x, transducer_coordinates_y),
        axis=-1,
    )  # shape: [n_elements, 2]

    beamformer = Beamformer(
        c0=c0,
        dx=dx,
        dt=dt,
        lateral_position_m=lateral_position,
        axial_position_m=axial_position,
        num_elements=num_elements,
        transducer_coordinates=transducer_coordinates,
    )

    n_time = 100
    signals = np.ones((num_elements, n_time))

    output_signal = beamformer.run(signals)
    assert output_signal.shape == (len(axial_position), len(lateral_position), 1)
