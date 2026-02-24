"""Simple 3D plane wave example using sparse-grid sensor output.

Instead of recording every grid point, the sensor is configured with
mod_x=4, mod_y=4, mod_z=4 so the binary records only every 4th point in each
spatial dimension.  This reduces output data size by a factor of 64
compared to a full-domain sensor.
"""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave.utils import plot_utils
from fullwave.utils.coordinates import map_to_coords


def main() -> None:  # noqa: PLR0915
    """Run 3D simple plane wave example with sparse-grid sensor."""
    logging.getLogger("__main__").setLevel(logging.INFO)

    work_dir = Path("./outputs/") / "simple_plane_wave_sparse_grid_3d"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- grid ---
    #
    f0 = 1e6
    c0 = 1540
    wavelength = c0 / f0
    domain_size = (10 * wavelength, 10 * wavelength, 10 * wavelength)  # meters
    duration = domain_size[0] / c0 * 2
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0)
    grid.print_info()

    #
    # --- medium ---
    #
    sound_speed_map = 1540 * np.ones((grid.nx, grid.ny, grid.nz))
    density_map = 1000 * np.ones((grid.nx, grid.ny, grid.nz))
    alpha_coeff_map = 0.5 * np.ones((grid.nx, grid.ny, grid.nz))
    alpha_power_map = 1.0 * np.ones((grid.nx, grid.ny, grid.nz))
    beta_map = np.zeros((grid.nx, grid.ny, grid.nz))

    obj_x = slice(grid.nx // 3, 2 * grid.nx // 3)
    obj_y = slice(grid.ny // 4, 3 * grid.ny // 4)
    obj_z = slice(grid.nz // 3, 9 * grid.nz // 10)
    sound_speed_map[obj_x, obj_y, obj_z] = 1600
    density_map[obj_x, obj_y, obj_z] = 1100
    alpha_coeff_map[obj_x, obj_y, obj_z] = 0.75
    alpha_power_map[obj_x, obj_y, obj_z] = 1.1

    medium = fullwave.Medium(
        grid=grid,
        sound_speed=sound_speed_map,
        density=density_map,
        alpha_coeff=alpha_coeff_map,
        alpha_power=alpha_power_map,
        beta=beta_map,
    )
    medium.print_info()

    #
    # --- source: plane wave from the top ---
    #
    ncycles = 2
    drop_off = 2
    element_thickness_px = 3

    p_mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=bool)
    p_mask[:element_thickness_px, :] = True
    p_coordinates = map_to_coords(p_mask)

    p0 = np.zeros((p_mask.sum(), grid.nt))
    for i in range(element_thickness_px):
        p0_vec = fullwave.utils.pulse.gaussian_modulated_sinusoidal_signal(
            nt=grid.nt,
            f0=f0,
            duration=duration,
            ncycles=ncycles,
            drop_off=drop_off,
            p0=1e5,
            i_layer=i,
            dt_for_layer_delay=grid.dt,
            cfl_for_layer_delay=grid.cfl,
        )
        n_yz = p_coordinates.shape[0] // element_thickness_px
        p0[n_yz * i : n_yz * (i + 1), :] = p0_vec[: grid.nt]

    source = fullwave.Source(
        p0=p0,
        coords=p_coordinates,
        grid_shape=(grid.nx, grid.ny, grid.nz),
    )

    #
    # --- sparse-grid sensor ---
    #
    # Record every 4th point in x (depth), y (lateral), and z (elevational).
    # The binary generates the sensor positions automatically; no mask or
    # coordinate array is needed.  mod_z is required for 3D simulations.
    mod_x = 2
    mod_y = 2
    mod_z = 2
    sensor = fullwave.Sensor(mod_x=mod_x, mod_y=mod_y, mod_z=mod_z, sampling_modulus_time=4)

    #
    # --- solver ---
    #
    # Sparse-grid output is supported by the exponential-attenuation binary.
    fw_solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        use_exponential_attenuation=True,
    )
    fw_solver.summary()

    # sensor_output shape: [n_sparse_sensors, nt_recorded]
    # n_sparse_sensors is inferred by the solver from the binary output length.
    sensor_output = fw_solver.run()

    #
    # --- reshape and visualize ---
    #
    # Compute the subsampled spatial dimensions to reconstruct the 3D wavefield.
    nx_sparse = int(np.ceil(grid.nx / mod_x))
    ny_sparse = int(np.ceil(grid.ny / mod_y))
    nz_sparse = int(np.ceil(grid.nz / mod_z))
    nt_recorded = sensor_output.shape[1]

    # Reshape to [nt_recorded, nx_sparse, ny_sparse, nz_sparse]
    propagation_map = sensor_output.T.reshape(nt_recorded, nx_sparse, ny_sparse, nz_sparse)

    time_step = nt_recorded // 3
    p_max = np.abs(propagation_map).max() / 4

    # x-y slice at the middle z index
    plot_utils.plot_array(
        propagation_map[time_step, :, :, nz_sparse // 2],
        aspect=ny_sparse / nx_sparse,
        export_path=work_dir / "wave_propagation_sparse_snapshot_x-y.png",
        vmax=p_max,
        vmin=-p_max,
    )

    # x-z slice at the middle y index
    plot_utils.plot_array(
        propagation_map[time_step, :, ny_sparse // 2, :],
        aspect=nz_sparse / nx_sparse,
        export_path=work_dir / "wave_propagation_sparse_snapshot_x-z.png",
        vmax=p_max,
        vmin=-p_max,
    )

    print(f"Full grid size:    ({grid.nx}, {grid.ny}, {grid.nz})")
    print(
        f"Sparse grid size:  ({nx_sparse}, {ny_sparse}, {nz_sparse})"
        f"  [mod_x={mod_x}, mod_y={mod_y}, mod_z={mod_z}]",
    )
    print(f"Output shape:      {sensor_output.shape}  [n_sparse_sensors, nt_recorded]")


if __name__ == "__main__":
    main()
