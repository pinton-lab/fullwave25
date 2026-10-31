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


def test_beamformer_array_depth_is_not_in_time_origin():
    """Envelope peak lands at the true depth when the array face is not at row 0."""
    c0 = 1540.0
    dx = 3.468468468468468e-05
    dt = 9.009009009009009e-09
    num_elements = 32
    array_row = 30
    target_row = 600
    num_samples = 6000
    pulse_sample = 100

    cols = 200 + np.arange(num_elements)
    transducer_coordinates = np.stack((np.full(num_elements, array_row), cols), axis=-1)
    focus = np.array([target_row, cols.mean()])
    distance_to_focus = np.sqrt(((transducer_coordinates - focus) ** 2).sum(axis=1))
    two_way_samples_to_target = 2 * (target_row - array_row) * dx / c0 / dt
    receive_delay_samples = (distance_to_focus - distance_to_focus.min()) * dx / c0 / dt
    echo_sample = np.round(two_way_samples_to_target + receive_delay_samples).astype(int)
    echo_sample += pulse_sample

    signals = np.zeros((num_elements, num_samples))
    sample = np.arange(num_samples)
    for i in range(num_elements):
        for center, amplitude in ((pulse_sample, 1.0), (echo_sample[i], 0.5)):
            signals[i] += (
                amplitude
                * np.exp(-(((sample - center) / 6.0) ** 2))
                * np.cos(2 * np.pi * 0.1 * (sample - center))
            )

    rows = np.arange(target_row - 60, target_row + 61, 2)
    beamformer = Beamformer(
        c0=c0,
        dx=dx,
        dt=dt,
        lateral_position_m=np.array([0.0]),
        axial_position_m=rows * dx,
        num_elements=num_elements,
        transducer_coordinates=transducer_coordinates,
    )
    envelope = np.abs(beamformer.run(signals))[:, 0, 0]
    peak_row = rows[np.argmax(envelope)]
    assert abs(peak_row - target_row) <= 2
