"""Send one plane wave with the experimental fullwave2-ultra backend.

This is simple_plane_wave.py with the changes that ``Solver(use_ultra=True)`` needs:

- ``Grid(ppw=6)``, the resolution the backend is optimized for (a larger ppw runs, but
  more slowly, and gives a warning).
- ``alpha_power`` is 1 everywhere, also inside the embedded object.
- The sensor is a mask over the whole grid, not the sparse-grid mode (``mod_x``, ``mod_y``).
- The solver gets no ``save_gpu_memory``, no PML options and no ``use_exponential_attenuation``.
- ``path_fullwave_simulation_bin`` points to the fullwave2-ultra binary.

Run it with:
    uv run python examples/simple_plane_wave/simple_plane_wave_ultra.py
"""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave.utils import plot_utils, pulse, signal_process
from fullwave.utils.coordinates import map_to_coords

F0 = 3e6  # center frequency [Hz]
C0 = 1540  # background sound speed [m/s]
P0_PA = 1e5  # source amplitude [Pa]


def make_grid() -> fullwave.Grid:
    """Return the 3 cm x 2 cm grid at ppw = 6, the setting the ultra backend is fastest at."""
    domain_size = (3e-2, 2e-2)  # meters
    return fullwave.Grid(
        domain_size=domain_size,
        f0=F0,
        duration=domain_size[0] / C0 * 2,
        c0=C0,
        ppw=6,
    )


def make_medium(*, grid: fullwave.Grid) -> fullwave.Medium:
    """Return a water-like medium with a denser, faster object in the center.

    The ultra backend needs ``alpha_power == 1`` everywhere.
    """
    sound_speed_map = C0 * np.ones((grid.nx, grid.ny))  # m/s
    density_map = 1000 * np.ones((grid.nx, grid.ny))  # kg/m^3
    alpha_coeff_map = 0.5 * np.ones((grid.nx, grid.ny))  # dB/(MHz cm)
    alpha_power_map = np.ones((grid.nx, grid.ny))  # must stay 1 for ultra
    beta_map = np.zeros((grid.nx, grid.ny))  # nonlinearity parameter

    # the object occupies the central third of the grid in each direction
    obj = (
        slice(grid.nx // 3, 2 * grid.nx // 3),
        slice(grid.ny // 3, 2 * grid.ny // 3),
    )
    sound_speed_map[obj] = 1600
    density_map[obj] = 1100
    alpha_coeff_map[obj] = 0.75

    return fullwave.Medium(
        grid=grid,
        sound_speed=sound_speed_map,
        density=density_map,
        alpha_coeff=alpha_coeff_map,
        alpha_power=alpha_power_map,
        beta=beta_map,
    )


def make_source(*, grid: fullwave.Grid) -> fullwave.Source:
    """Return a one-row plane wave source at the top of the grid.

    ``source_type="soft"`` needs exactly one row.
    """
    p_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    p_mask[0, :] = True
    p_coordinates = map_to_coords(p_mask)

    pulse_signal = pulse.gaussian_modulated_sinusoidal_signal(
        nt=grid.nt,
        f0=F0,
        duration=grid.duration,
        ncycles=2,
        drop_off=2,
        p0=P0_PA,
    )
    # every source cell plays the same pulse: [n_sources, nt]
    p0 = np.tile(pulse_signal, (p_coordinates.shape[0], 1))
    return fullwave.Source(p0=p0, coords=p_coordinates, grid_shape=grid.shape)


def plot_results(
    *,
    sensor_output: np.ndarray,
    grid: fullwave.Grid,
    medium: fullwave.Medium,
    work_dir: Path,
) -> None:
    """Save a snapshot of the pressure field and an animation of the propagation."""
    propagation_map = signal_process.reshape_whole_sensor_to_nt_nx_ny(sensor_output, grid)
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


def main() -> None:
    """Run the simple plane wave example on the fullwave2-ultra backend."""
    logging.getLogger("__main__").setLevel(logging.INFO)

    path_ultra_bin = Path("/path/to/fullwave2-ultra/bin")

    work_dir = Path("./outputs/") / "simple_plane_wave_ultra"
    work_dir.mkdir(parents=True, exist_ok=True)

    grid = make_grid()
    medium = make_medium(grid=grid)
    medium.plot(export_path=work_dir / "medium.png")
    source = make_source(grid=grid)

    # record the pressure at every grid point, every 7th time step
    sensor = fullwave.Sensor(mask=np.ones(grid.shape, dtype=bool), sampling_modulus_time=7)

    solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        path_fullwave_simulation_bin=path_ultra_bin,
        run_on_memory=True,
        source_type="soft",  # the signal is added to the field, not assigned
        use_ultra=True,
    )
    sensor_output = solver.run()

    plot_results(sensor_output=sensor_output, grid=grid, medium=medium, work_dir=work_dir)


if __name__ == "__main__":
    main()
