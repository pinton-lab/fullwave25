"""Simple plane wave transmit example."""

from pathlib import Path

import numpy as np

import fullwave
from fullwave.utils import plot_utils


def main() -> None:
    """Run Simple plane wave transmit example."""
    #
    # define the working directory
    #

    work_dir = Path("./outputs/") / "simple_plane_wave_3d"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- define the computational grid ---
    #
    f0 = 1e6
    c0 = 1540
    wavelength = c0 / f0
    domain_size = (5 * wavelength, 5 * wavelength, 5 * wavelength)  # meters
    duration = domain_size[0] / c0 * 2
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0)

    #
    # --- define the acoustic medium properties ---
    #
    sound_speed = 1540
    density = 1000
    alpha_coeff = 0.5
    alpha_power = 1.0
    beta = 0.0

    sound_speed_map = sound_speed * np.ones((grid.nx, grid.ny, grid.nz))
    density_map = density * np.ones((grid.nx, grid.ny, grid.nz))
    alpha_coeff_map = alpha_coeff * np.ones((grid.nx, grid.ny, grid.nz))
    alpha_power_map = alpha_power * np.ones((grid.nx, grid.ny, grid.nz))
    beta_map = beta * np.ones((grid.nx, grid.ny, grid.nz))

    medium = fullwave.Medium(
        grid=grid,
        sound_speed=sound_speed_map,
        density=density_map,
        alpha_coeff=alpha_coeff_map,
        alpha_power=alpha_power_map,
        beta=beta_map,
    )

    #
    # --- define the acoustic source ---
    #

    # define where to put the pressure source [nx, ny]
    p_mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=bool)
    p_mask[0, :, :] = True

    # define the pressure source [n_sources, nt]
    p0_vec = fullwave.utils.pulse.gaussian_modulated_sinusoidal_signal(
        nt=grid.nt,
        f0=f0,
        duration=duration,
        ncycles=1,
        drop_off=1,
        p0=1e5,
    )
    p0 = np.zeros((p_mask.sum(), grid.nt))
    p0[:] = p0_vec

    source = fullwave.Source(p0, p_mask)

    #
    # --- define the sensor ---
    #
    sensor_mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=bool)
    sensor_mask[:, :] = True
    sensor = fullwave.Sensor(mask=sensor_mask, sampling_interval=1)

    #
    # --- run simulation ---
    #
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
    sensor_output = fw_solver.run()

    #
    # --- visualization ---
    #
    propagation_map = sensor_output.reshape((grid.nx, grid.ny, grid.nz, -1))
    propagation_map = propagation_map.transpose((3, 0, 1, 2))  # nt, nx, ny, nz

    plot_utils.plot_array(
        propagation_map[propagation_map.shape[0] // 2, :, :, grid.nz // 2],
        aspect=1,
        export_path=work_dir / "propagation_map_slice.png",
    )

    propagation_map_slice = propagation_map[:, :, :, grid.nz // 2]
    plot_utils.plot_wave_propagation_animation(
        propagation_map=propagation_map_slice,
        export_name=work_dir / "propagation_animation.mp4",  # requires ffmpeg
        vmax=1e5,
        vmin=-1e5,
    )

    print(sensor_output)  # noqa: T201


if __name__ == "__main__":
    main()
