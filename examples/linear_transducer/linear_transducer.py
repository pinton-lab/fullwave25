"""Send one plane wave from a linear array and record the element traces.

The medium holds a square of slower tissue, so the recorded traces carry its echo.

Run it with:
    uv run python examples/linear_transducer/linear_transducer.py
"""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave.utils import plot_utils


def main() -> None:
    """Run one plane wave transmit on a linear array."""
    # overwrite the logging level, DEBUG, INFO, WARNING, ERROR
    logging.getLogger("__main__").setLevel(logging.INFO)

    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "linear_transducer"
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- define the computational grid ---
    domain_size = (42.5e-3 / 2, 42.5e-3)  # meters
    f0 = 3.7e6
    c0 = 1540
    duration = domain_size[0] / c0 * 2.5
    ppw = 12
    cfl = 0.4
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0, ppw=ppw, cfl=cfl)

    # --- define the acoustic medium properties ---
    sound_speed = 1540
    density = 1000
    alpha_coeff = 0.5
    alpha_power = 1.0
    beta = 0.0

    sound_speed_map = sound_speed * np.ones((grid.nx, grid.ny))
    # put a square with different sound speed
    sound_speed_map[
        grid.nx // 2 - int(grid.nx * 0.1) : grid.nx // 2 + int(grid.nx * 0.1),
        grid.ny // 2 - int(grid.ny * 0.1) : grid.ny // 2 + int(grid.ny * 0.1),
    ] = 1400

    density_map = density * np.ones((grid.nx, grid.ny))
    alpha_coeff_map = alpha_coeff * np.ones((grid.nx, grid.ny))
    alpha_power_map = alpha_power * np.ones((grid.nx, grid.ny))
    beta_map = beta * np.ones((grid.nx, grid.ny))

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
    # --- define the linear transducer ---
    #

    # A named array places itself, so only the transmit has to be stated.
    transducer = fullwave.Transducer.l7_4(grid)

    #
    # --- define the transmit ---
    #

    # send the plane wave straight ahead
    transducer.plane_wave(angle_deg=0.0)

    transducer.plot_source_mask(work_dir / "source_transducer.png")
    transducer.plot_sensor_mask(work_dir / "sensor_transducer.png")

    # --- run simulation ---
    fw_solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        transducer=transducer,
        run_on_memory=False,
    )
    sensor_output = fw_solver.run()

    sensor_output = transducer.post_process_sensor_output(
        sensor_output,
        average_surface_signals=True,
    )

    # --- visualization ---
    plot_utils.plot_array(
        transducer.sensor.indexed_mask,
        export_path=work_dir / "transducer_sensor_index.png",
        xlim=[-10, transducer.sensor.mask.shape[1] + 10],
        ylim=[-10, transducer.sensor.mask.shape[0] + 10],
        reverse_y_axis=True,
    )

    plot_utils.plot_array(
        sensor_output.T,
        aspect=sensor_output.shape[0] / sensor_output.shape[1],
        export_path=work_dir / "rf.svg",
    )


if __name__ == "__main__":
    main()
