"""Simple plane wave transmit example using a velocity (u0) source.

A velocity source drives the particle-velocity equation directly, as opposed to
the hard pressure source (p0) which drives the pressure equation.  For a plane
wave propagating in the depth (x) direction the relevant component is u0.

The acoustic impedance relation  p = rho · c · u  links the two formulations, so
the velocity amplitude is scaled as

    u_amp = p_amp / (ρ₀ · c₀)

Everything else (grid, medium, sensor, solver) is identical to
simple_plane_wave.py so the two examples can be run side-by-side for comparison.
"""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave.utils import plot_utils, signal_process
from fullwave.utils.coordinates import map_to_coords


def main() -> None:
    """Run simple plane wave transmit example with a velocity source."""
    logging.getLogger("__main__").setLevel(logging.INFO)

    #
    # --- working directory ---
    #
    work_dir = Path("./outputs/") / "simple_plane_wave_velocity_source"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- computational grid ---
    #
    domain_size = (3e-2, 2e-2)  # (depth, lateral) in metres
    f0 = 3e6  # centre frequency [Hz]
    c0 = 1540.0  # background sound speed [m/s]
    rho0 = 1000.0  # background density [kg/m³]
    duration = domain_size[0] / c0 * 2  # two-way travel time across depth
    grid = fullwave.Grid(
        domain_size=domain_size,
        f0=f0,
        duration=duration,
        c0=c0,
    )

    #
    # --- acoustic medium ---
    #
    sound_speed_map = c0 * np.ones((grid.nx, grid.ny))  # m/s
    density_map = rho0 * np.ones((grid.nx, grid.ny))  # kg/m³
    alpha_coeff_map = 0.5 * np.ones((grid.nx, grid.ny))  # dB/(MHz^y cm)
    alpha_power_map = 1.0 * np.ones((grid.nx, grid.ny))  # power-law exponent
    beta_map = 0.0 * np.ones((grid.nx, grid.ny))  # nonlinearity coefficient

    # embed a scatterer with different acoustic properties
    obj_x_start = grid.nx // 3
    obj_x_end = 2 * grid.nx // 3
    obj_y_start = grid.ny // 3
    obj_y_end = 2 * grid.ny // 3

    sound_speed_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = 1600
    density_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = 1100
    alpha_coeff_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = 0.75
    alpha_power_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = 1.1

    medium = fullwave.Medium(
        grid=grid,
        sound_speed=sound_speed_map,
        density=density_map,
        alpha_coeff=alpha_coeff_map,
        alpha_power=alpha_power_map,
        beta=beta_map,
    )
    medium.plot(export_path=work_dir / "medium.png")

    #
    # --- velocity source ---
    #
    # Use the u-component (depth / x direction) to drive a downward-travelling
    # plane wave.  The source occupies the top `element_thickness_px` rows of the
    # grid, matching the pressure-source layout in simple_plane_wave.py.
    #
    # Velocity amplitude is derived from the target pressure amplitude via the
    # plane-wave impedance relation:  u_amp = p_amp / (ρ₀ · c₀)
    #
    p_amp = 1e5  # target pressure amplitude [Pa]
    u_amp = p_amp / (rho0 * c0)  # corresponding velocity amplitude [m/s]

    ncycles = 2
    drop_off = 2
    # element_thickness_px = 3

    # Build the coordinate array for the velocity source layer
    # small velocity source at the center of the domain

    source_width_px_x = 2
    source_width_px_y = 2
    u_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    u_mask[
        grid.nx // 2 - source_width_px_x // 2 : grid.nx // 2 + source_width_px_x // 2,
        grid.ny // 2 - source_width_px_y // 2 : grid.ny // 2 + source_width_px_y // 2,
    ] = True

    coords_u = map_to_coords(u_mask)  # shape [n_sources_u, 2]

    # Build the u0 signal matrix [n_sources_u, nt]
    u0 = np.zeros((coords_u.shape[0], grid.nt))

    u0_vec = fullwave.utils.pulse.gaussian_modulated_sinusoidal_signal(
        nt=grid.nt,
        f0=f0,
        duration=duration,
        ncycles=ncycles,
        drop_off=drop_off,
        p0=u_amp,  # amplitude in m/s
    )
    u0[:, :] = u0_vec

    # for i_layer in range(element_thickness_px):
    #     u0_vec = fullwave.utils.pulse.gaussian_modulated_sinusoidal_signal(
    #         nt=grid.nt,
    #         f0=f0,
    #         duration=duration,
    #         ncycles=ncycles,
    #         drop_off=drop_off,
    #         p0=u_amp,  # amplitude in m/s
    #         i_layer=i_layer,
    #         dt_for_layer_delay=grid.dt,
    #         cfl_for_layer_delay=grid.cfl,
    #     )
    #     n_y = coords_u.shape[0] // element_thickness_px
    #     u0[n_y * i_layer : n_y * (i_layer + 1), :] = u0_vec

    source = fullwave.Source(
        grid_shape=grid.shape,
        u0=u0,
        coords_u=coords_u,
    )

    #
    # --- sensor ---
    #
    sensor_mask = np.ones((grid.nx, grid.ny), dtype=bool)
    sensor = fullwave.Sensor(mask=sensor_mask, sampling_modulus_time=7)

    #
    # --- solver ---
    #
    fw_solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        run_on_memory=False,
        use_exponential_attenuation=True,
        save_gpu_memory=True,
        path_fullwave_simulation_bin=Path(
            "/home/msode/workspace/lab_repos/fullwave-python-public/debug_solver_bin/fullwave2_2d_exponential_attenuation_multi_gpu",
        ),
    )
    sensor_output = fw_solver.run()

    #
    # --- visualisation ---
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
        export_path=work_dir / "wave_propagation_snapshot.png",
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
