"""Record every fourth grid point rather than every one.

`Sensor(mod_x=4, mod_y=4)` keeps one point in each direction out of four, which
makes the output 16 times smaller than a sensor over the whole domain.

Run it with:
    uv run python examples/simple_plane_wave/simple_plane_wave_sparse_grid.py
"""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave.utils import plot_utils
from fullwave.utils.coordinates import map_to_coords


def main() -> None:
    """Run simple plane wave example with sparse-grid sensor."""
    logging.getLogger("__main__").setLevel(logging.INFO)

    work_dir = Path("./outputs/") / "simple_plane_wave_sparse_grid"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- grid ---
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
    # --- medium ---
    #
    sound_speed_map = 1540 * np.ones((grid.nx, grid.ny))
    density_map = 1000 * np.ones((grid.nx, grid.ny))
    alpha_coeff_map = 0.5 * np.ones((grid.nx, grid.ny))
    alpha_power_map = 1.0 * np.ones((grid.nx, grid.ny))
    beta_map = np.zeros((grid.nx, grid.ny))

    obj_x = slice(grid.nx // 3, 2 * grid.nx // 3)
    obj_y = slice(grid.ny // 3, 2 * grid.ny // 3)
    sound_speed_map[obj_x, obj_y] = 1600
    density_map[obj_x, obj_y] = 1100
    alpha_coeff_map[obj_x, obj_y] = 0.75
    alpha_power_map[obj_x, obj_y] = 1.1

    medium = fullwave.Medium(
        grid=grid,
        sound_speed=sound_speed_map,
        density=density_map,
        alpha_coeff=alpha_coeff_map,
        alpha_power=alpha_power_map,
        beta=beta_map,
    )

    #
    # --- source: plane wave from the top ---
    #
    ncycles = 2
    drop_off = 2
    element_thickness_px = 3

    p_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
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
        n_y = p_coordinates.shape[0] // element_thickness_px
        p0[n_y * i : n_y * (i + 1), :] = p0_vec[: grid.nt]

    source = fullwave.Source(p0=p0, coords=p_coordinates, grid_shape=grid.shape)

    #
    # --- sparse-grid sensor ---
    #
    # Record every 4th point in x (depth) and every 4th point in y (lateral).
    # The binary generates the sensor positions automatically; no mask or
    # coordinate array is needed.
    mod_x = 4
    mod_y = 4
    sensor = fullwave.Sensor(mod_x=mod_x, mod_y=mod_y, sampling_modulus_time=7)

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
    )
    fw_solver.summary()

    # sensor_output shape: [n_sparse_sensors, nt_recorded]
    # n_sparse_sensors is inferred by the solver from the binary output length.
    sensor_output = fw_solver.run()

    #
    # --- reshape and visualize ---
    #
    # Compute the subsampled spatial dimensions to reconstruct the 2D wavefield.
    nx_sparse = int(np.ceil(grid.nx / mod_x))
    ny_sparse = int(np.ceil(grid.ny / mod_y))
    nt_recorded = sensor_output.shape[1]

    # Reshape to [nt_recorded, nx_sparse, ny_sparse]
    propagation_map = sensor_output.T.reshape(nt_recorded, nx_sparse, ny_sparse)

    p_max = np.abs(propagation_map).max() / 4
    time_step = nt_recorded // 3
    plot_utils.plot_array(
        propagation_map[time_step],
        aspect=ny_sparse / nx_sparse,
        export_path=work_dir / "wave_propagation_sparse_snapshot.png",
        vmax=p_max,
        vmin=-p_max,
    )

    print(f"Full grid size:    ({grid.nx}, {grid.ny})")
    print(f"Sparse grid size:  ({nx_sparse}, {ny_sparse})  [mod_x={mod_x}, mod_y={mod_y}]")
    print(f"Output shape:      {sensor_output.shape}  [n_sparse_sensors, nt_recorded]")


if __name__ == "__main__":
    main()
