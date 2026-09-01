"""Send one plane wave from an additive source.

An additive source adds its signal to the pressure of its own nodes, so what it
radiates depends on the time step. `Source.additive` applies the scale that makes
the row radiate the pressure it is given. One row is enough, because an additive
source needs no thickness.

Run it with:
    uv run python examples/simple_plane_wave/simple_plane_wave_additive_source.py
"""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave.utils import plot_utils, signal_process
from fullwave.utils.coordinates import map_to_coords


def main() -> None:
    """Run one plane wave transmit from an additive source."""
    # overwrite the logging level, DEBUG, INFO, WARNING, ERROR
    logging.getLogger("__main__").setLevel(logging.INFO)

    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "simple_plane_wave_additive_source"
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

    # an additive source needs one row, so no layer delay is needed either
    p_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    p_mask[0, :] = True
    p_coordinates = map_to_coords(p_mask)

    p0_vec = fullwave.utils.pulse.gaussian_modulated_sinusoidal_signal(
        nt=grid.nt,  # number of time steps
        f0=f0,  # center frequency [Hz]
        duration=duration,  # duration [s]
        ncycles=2,  # number of cycles
        drop_off=2,  # drop off factor
        p0=1e5,  # the pressure the row radiates [Pa]
    )
    p0 = np.tile(p0_vec, (p_coordinates.shape[0], 1))

    # Source.additive applies the 2 c dt / dx scale, which Source(p0_additive=...) does not.
    source = fullwave.Source.additive(
        p0=p0,
        coords=p_coordinates,
        grid_shape=grid.shape,
        grid=grid,
    )

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
        run_on_memory=True,
    )
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
