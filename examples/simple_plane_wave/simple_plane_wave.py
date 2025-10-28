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
    sound_speed = 1540  # m/s
    density = 1000  # kg/m^3
    alpha_coeff = 0.5  # dB/(MHz^gamma * cm)
    alpha_power = 1.0  # [-]
    beta = 0.0

    sound_speed_map = sound_speed * np.ones((grid.nx, grid.ny))
    sound_speed_map[
        int(grid.nx // 2 - grid.nx * 0.1) : int(grid.nx // 2 + grid.nx * 0.1),
        int(grid.ny // 2 - grid.ny * 0.1) : int(grid.ny // 2 + grid.ny * 0.1),
    ] = 1600  # m/s

    density_map = density * np.ones((grid.nx, grid.ny))
    alpha_coeff_map = alpha_coeff * np.ones((grid.nx, grid.ny))
    alpha_power_map = alpha_power * np.ones((grid.nx, grid.ny))
    beta_map = beta * np.ones((grid.nx, grid.ny))

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

    # define where to put the pressure source [nx, ny]
    p_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    p_mask[0:1, :] = True

    # define the pressure source [n_sources, nt]
    p0_vec = fullwave.utils.pulse.gaussian_modulated_sinusoidal_signal(
        nt=grid.nt,
        f0=f0,
        duration=duration,
        ncycles=2,
        drop_off=2,
        p0=1e5,
    )
    p0 = np.zeros((p_mask.sum(), grid.nt))
    p0[:] = p0_vec

    # setup the Source instance
    source = fullwave.Source(p0, p_mask)

    #
    # --- define the sensor ---
    #
    sensor_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    sensor_mask[:, :] = True

    # setup the Sensor instance
    sensor = fullwave.Sensor(mask=sensor_mask, sampling_interval=7)

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
        pml_layer_thickness_px=grid.ppw * 3,
        n_transition_layer=grid.ppw * 3,
    )
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
    )


if __name__ == "__main__":
    main()
