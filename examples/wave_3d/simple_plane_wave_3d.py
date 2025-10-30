"""Simple plane wave transmit example."""

from pathlib import Path

import numpy as np

import fullwave
from fullwave.utils import plot_utils


def main() -> None:  # noqa: PLR0915
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
    domain_size = (10 * wavelength, 10 * wavelength, 10 * wavelength)  # meters
    duration = domain_size[0] / c0 * 2
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0)
    grid.print_info()

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

    # embed an object with different properties in the center of the medium
    obj_x_start = grid.nx // 3
    obj_x_end = 2 * grid.nx // 3

    obj_y_start = grid.ny // 4
    obj_y_end = 3 * grid.ny // 4

    obj_z_start = grid.nz // 3
    obj_z_end = 9 * grid.nz // 10

    sound_speed_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end, obj_z_start:obj_z_end] = (
        1600  # m/s
    )
    density_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end, obj_z_start:obj_z_end] = (
        1100  # kg/m^3
    )
    alpha_coeff_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end, obj_z_start:obj_z_end] = (
        0.75  # dB/(MHz^y cm)
    )
    alpha_power_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end, obj_z_start:obj_z_end] = (
        1.1  # power law exponent
    )
    beta_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end, obj_z_start:obj_z_end] = (
        0.0  # nonlinearity
    )

    medium = fullwave.Medium(
        grid=grid,
        sound_speed=sound_speed_map,
        density=density_map,
        alpha_coeff=alpha_coeff_map,
        alpha_power=alpha_power_map,
        beta=beta_map,
    )
    medium.print_info()
    medium.plot(figsize=(20, 6), export_path=Path(work_dir / "medium.png"))

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
    sensor = fullwave.Sensor(mask=sensor_mask, sampling_interval=7)

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
    )
    fw_solver.print_info()
    sensor_output = fw_solver.run()

    #
    # --- visualization ---
    #
    propagation_map = sensor_output.reshape((grid.nx, grid.ny, grid.nz, -1))
    propagation_map = propagation_map.transpose((3, 0, 1, 2))  # nt, nx, ny, nz

    plot_utils.plot_array(
        propagation_map[propagation_map.shape[0] // 2, :, :, grid.nz // 2],
        aspect=1,
        export_path=work_dir / "propagation_map_slice_x-y.png",
    )

    propagation_map_slice = propagation_map[:, :, :, grid.nz // 2]
    plot_utils.plot_wave_propagation_with_map(
        propagation_map=propagation_map_slice,
        c_map=medium.sound_speed[:, :, grid.nz // 2],
        rho_map=medium.density[:, :, grid.nz // 2],
        export_name=work_dir / "wave_propagation_x-y.mp4",  # requires ffmpeg
        vmax=1e5,
        vmin=-1e5,
        figsize=(6, 6),
    )

    plot_utils.plot_array(
        propagation_map[propagation_map.shape[0] // 2, :, grid.ny // 2, :],
        aspect=1,
        export_path=work_dir / "propagation_map_slice_x-z.png",
    )
    plot_utils.plot_wave_propagation_with_map(
        propagation_map=propagation_map[:, :, grid.ny // 2, :],
        c_map=medium.sound_speed[:, grid.ny // 2, :],
        rho_map=medium.density[:, grid.ny // 2, :],
        export_name=work_dir / "wave_propagation_x-z.mp4",  # requires ffmpeg
        vmax=1e5,
        vmin=-1e5,
        figsize=(6, 6),
    )


if __name__ == "__main__":
    main()
