"""Simple plane wave transmit example."""

from pathlib import Path

import numpy as np

import fullwave
from fullwave.utils import plot_utils, signal_process
from fullwave.utils.coordinates import map_to_coords


def main() -> None:
    """Run Simple plane wave transmit example."""
    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "simple_plane_wave"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- define the computational grid ---
    #
    domain_size = (3e-2, 2e-2)  # meters
    f0 = 3e6
    c0 = 1540
    duration = domain_size[0] / c0 * 2
    grid = fullwave.Grid(
        domain_size=domain_size,
        f0=f0,
        duration=duration,
        c0=c0,
    )

    #
    # --- define the acoustic medium properties ---
    #
    # Define the base 2D medium arrays
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

    # setup the Medium instance
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
    # --- define the acoustic source ---
    #

    # initialize the pressure source mask
    p_mask = np.zeros((grid.nx, grid.ny), dtype=bool)

    # set the source location at the top rows of the grid with specified thickness
    element_thickness_px = 3
    p_mask[0:element_thickness_px, :] = True

    # define the pressure source [n_sources, nt]d
    p0 = np.zeros((p_mask.sum(), grid.nt))  # [n_sources, nt]

    # The order of p_coordinates corresponds to the order of sources in p0
    p_coordinates = map_to_coords(p_mask)

    for i_thickness in range(element_thickness_px):
        # create a gaussian-modulated sinusoidal pulse as the source signal with layer delay
        p0_vec = fullwave.utils.pulse.gaussian_modulated_sinusoidal_signal(
            nt=grid.nt,  # number of time steps
            f0=f0,  # center frequency [Hz]
            duration=duration,  # duration [s]
            ncycles=2,  # number of cycles
            drop_off=2,  # drop off factor
            p0=1e5,  # maximum amplitude [Pa]
            i_layer=i_thickness,
            dt_for_layer_delay=grid.dt,
            cfl_for_layer_delay=grid.cfl,
        )

        # assign the source signal to the corresponding layer
        n_y = p_coordinates.shape[0] // element_thickness_px
        p0[n_y * i_thickness : n_y * (i_thickness + 1), :] = p0_vec.copy()

    # setup the Source instance
    source = fullwave.Source(p0, p_mask)

    #
    # --- define the sensor ---
    #
    sensor_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    sensor_mask[:, :] = True

    # setup the Sensor instance
    sensor = fullwave.Sensor(mask=sensor_mask, sampling_modulus_time=7)

    #
    # --- run simulation ---
    #
    # setup the Solver instance
    fw_solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        run_on_memory=False,
    )
    fw_solver.summary()
    # execute the solver
    sensor_output = fw_solver.run()

    #
    # --- visualization ---
    #

    propagation_map = signal_process.reshape_whole_sensor_to_nt_nx_ny(
        sensor_output,
        grid,
    )
    p_max_plot = np.abs(propagation_map).max().item() / 4
    time_step = propagation_map.shape[0] // 3
    plot_utils.plot_array(
        propagation_map[time_step, :, :],
        aspect=propagation_map.shape[2] / propagation_map.shape[1],
        export_path=work_dir / "wave_propagation_snapshot_1.png",
        vmax=p_max_plot,
        vmin=-p_max_plot,
    )
    plot_utils.plot_wave_propagation_with_map(
        propagation_map=propagation_map,
        c_map=medium.sound_speed,
        rho_map=medium.density,
        export_name=work_dir / "wave_propagation_animation.mp4",
        vmax=p_max_plot,
        vmin=-p_max_plot,
        figsize=(4, 6),
    )


if __name__ == "__main__":
    main()
