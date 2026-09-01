"""Give the solver relaxation parameters directly, rather than an attenuation map.

`Medium` looks the relaxation parameters up from the attenuation a caller asks for.
`MediumRelaxationMaps` takes the parameters themselves, which is what a caller
needs after fitting a table of their own.

Run it with:
    uv run python examples/simple_plane_wave/relaxation_parameter_input.py
"""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave.solver.shipped_database import ShippedDatabase
from fullwave.utils import plot_utils, signal_process
from fullwave.utils.coordinates import map_to_coords
from fullwave.utils.relaxation_parameters import generate_relaxation_params


def main() -> None:
    """Run one plane wave transmit on a medium built from relaxation parameters."""
    # overwrite the logging level, DEBGUG, INFO, WARNING, ERROR
    logging.getLogger("__main__").setLevel(logging.INFO)

    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "relaxation_parameter_input"
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

    # ShippedDatabase names every table this release holds, so an example never
    # writes a file name of its own.
    n_relaxation_mechanisms = ShippedDatabase.default_mechanisms
    path_relaxation_parameters_database = ShippedDatabase.table_of(n_relaxation_mechanisms)
    alpha_coeff = np.ones_like(sound_speed_map) * 0.5  # dB/(MHz^y cm)
    alpha_power = np.ones_like(sound_speed_map) * 1.0  # power law exponent

    relaxation_param_dict = generate_relaxation_params(
        n_relaxation_mechanisms=n_relaxation_mechanisms,
        alpha_coeff=alpha_coeff,
        alpha_power=alpha_power,
        path_database=path_relaxation_parameters_database,
    )

    # setup the Medium instance
    medium = fullwave.MediumRelaxationMaps(
        grid=grid,
        sound_speed=sound_speed_map,
        density=density_map,
        beta=beta_map,
        relaxation_param_dict=relaxation_param_dict,
        n_relaxation_mechanisms=n_relaxation_mechanisms,
    )
    medium.plot(export_path=Path(work_dir / "medium.png"), plot_fw2_params=True)

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
        cuda_device_id=[0],
    )
    # fw_solver.summary()
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
