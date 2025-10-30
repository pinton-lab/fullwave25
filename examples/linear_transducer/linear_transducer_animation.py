"""Simple plane wave transmit example."""

from pathlib import Path

import numpy as np

import fullwave
from fullwave.utils import plot_utils, signal_process


def main() -> None:
    """Run Simple plane wave transmit example."""
    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "linear_transducer"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- define the computational grid ---
    #

    domain_size = (42.5e-3 * 1.5, 42.5e-3)  # meters
    f0 = 1e6
    c0 = 1540
    duration = domain_size[0] / c0 * 2
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0)

    #
    # --- define the acoustic medium properties ---
    #

    sound_speed_map = 1540 * np.ones((grid.nx, grid.ny))  # m/s
    density_map = 1000 * np.ones((grid.nx, grid.ny))  # kg/m^3
    alpha_coeff_map = 0.5 * np.ones((grid.nx, grid.ny))  # dB/(MHz^y cm)
    alpha_power_map = 1.0 * np.ones((grid.nx, grid.ny))  # power law exponent
    beta_map = 0.0 * np.ones((grid.nx, grid.ny))  # nonlinearity parameter

    # embed an object with different properties in the center of the medium
    obj_x_start = grid.nx // 3
    obj_x_end = 2 * grid.nx // 3
    obj_y_start = grid.ny // 3
    obj_y_end = 2 * grid.ny // 3

    sound_speed_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = 1600  # m/s
    density_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = 1100  # kg/m^3
    alpha_coeff_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = 0.75  # dB/(MHz^y cm)
    alpha_power_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = 1.1  # power law exponent
    beta_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = 0.0  # nonlinearity parameter

    medium = fullwave.Medium(
        grid=grid,
        sound_speed=sound_speed_map,
        density=density_map,
        alpha_coeff=alpha_coeff_map,
        alpha_power=alpha_power_map,
        beta=beta_map,
    )
    medium.plot(export_path=Path(work_dir / "medium.png"))

    #
    # --- define the linear transducer ---
    #

    element_layer_px = 1
    transducer_geometry = fullwave.TransducerGeometry(
        grid,
        number_elements=128,
        # -
        element_width_m=0.146484375e-3,
        # element_width_px=6,  # depends on the ppw, cfl
        # -
        element_spacing_m=0.146484375e-3,
        # element_spacing_px=6,
        # -
        element_layer_px=element_layer_px,
        # -
        # [axial, lateral]
        # position_px=(0, 0),
        position_m=(
            0,
            (42.5 - 37.5) / 2 * 1e-3,
        ),
        # -
        radius=float("inf"),
    )
    p_max = 1e5

    # input_signal = generate_signal(input_signal, transducer_geometry)
    active_source_elements = np.zeros(transducer_geometry.number_elements, dtype=bool)
    active_sensor_elements = np.zeros(transducer_geometry.number_elements, dtype=bool)
    active_source_elements[32:96] = True
    active_sensor_elements[:] = True

    n_sources = transducer_geometry.n_sources_per_element * active_source_elements.sum()

    input_signal = np.zeros((n_sources, grid.nt))
    for i_layer in range(element_layer_px):
        p0_vec = fullwave.utils.pulse.gaussian_modulated_sinusoidal_signal(
            nt=grid.nt,
            f0=f0,
            duration=duration,
            ncycles=1,
            drop_off=1,
            p0=p_max,
            i_layer=i_layer + 1,
            dt_for_layer_delay=grid.dt,
            cfl_for_layer_delay=grid.cfl,
        )
        input_signal[i_layer::element_layer_px, :] = p0_vec.copy()

    transducer = fullwave.Transducer(
        transducer_geometry=transducer_geometry,
        grid=grid,
        input_signal=input_signal,
        active_source_elements=active_source_elements,
        active_sensor_elements=active_sensor_elements,
    )
    transducer.plot_source_mask(export_path=work_dir / "source_transducer.svg")
    transducer.plot_sensor_mask(export_path=work_dir / "sensor_transducer.svg")

    # make a sensor for whole domain to make an animation
    sensor_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    sensor_mask[:, :] = True
    sensor = fullwave.Sensor(mask=sensor_mask, sampling_interval=2)
    sensor.plot(export_path=work_dir / "sensor_whole.svg")

    #
    # --- run simulation ---
    #

    # input source and sensor separately for animation
    fw_solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        source=transducer.source,
        sensor=sensor,
        run_on_memory=False,
    )
    sensor_output = fw_solver.run()

    #
    # --- visualization ---
    #

    propagation_map = signal_process.reshape_whole_sensor_to_nt_nx_ny(sensor_output, grid)
    plot_utils.plot_wave_propagation_with_map(
        propagation_map=propagation_map,
        c_map=medium.sound_speed,
        rho_map=medium.density,
        export_name=work_dir / "wave_propagation.mp4",
        figsize=(4, 6),
    )


if __name__ == "__main__":
    main()
