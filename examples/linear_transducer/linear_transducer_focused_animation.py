"""Focus a P4-1C phased array from the middle half of its aperture.

The sensor covers the whole domain, so the run writes a movie of the field.

Run it with:
    uv run python examples/linear_transducer/linear_transducer_focused_animation.py
"""

from pathlib import Path

import numpy as np

import fullwave
from fullwave.utils import plot_utils, signal_process


def main() -> None:
    """Run linear transducer with focused transmit example."""
    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "linear_transducer_focused_animation"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- define the computational grid ---
    #

    f0 = 1e6
    c0 = 1540
    width_m = 34e-3  # lateral width [m], wider than the array
    domain_size = (
        42.5e-3 * 1.5,
        width_m,
    )  # meters
    duration = domain_size[0] / c0 * 2
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0)

    #
    # --- define the phased array ---
    #

    # A named array places itself, so only the transmit has to be stated.
    transducer = fullwave.Transducer.p4_1c(grid)

    #
    # --- define the acoustic medium properties ---
    #

    sound_speed_map = 1540 * np.ones((grid.nx, grid.ny))  # m/s
    density_map = 1000 * np.ones((grid.nx, grid.ny))  # kg/m^3
    alpha_coeff_map = 0.5 * np.ones((grid.nx, grid.ny))  # dB/(MHz^y cm)
    alpha_power_map = 1.0 * np.ones((grid.nx, grid.ny))  # power law exponent
    beta_map = 0.0 * np.ones((grid.nx, grid.ny))  # nonlinearity parameter

    # embed an object with different properties in the middle third of the tissue below the face
    tissue_rows = grid.nx - transducer.face_row_px
    obj_x_start = transducer.face_row_px + tissue_rows // 3
    obj_x_end = transducer.face_row_px + 2 * tissue_rows // 3
    obj_y_start = grid.ny // 3
    obj_y_end = 2 * grid.ny // 3

    sound_speed_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = 1600  # m/s
    density_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = 1100  # kg/m^3
    alpha_coeff_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = 0.75  # dB/(MHz^y cm)
    alpha_power_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = 1.1  # power law exponent
    beta_map[obj_x_start:obj_x_end, obj_y_start:obj_y_end] = 0.0  # nonlinearity parameter

    medium = fullwave.Medium(
        grid=grid,
        beta=beta_map,
        **transducer.fill_probe_above_face(
            sound_speed=sound_speed_map,
            density=density_map,
            alpha_coeff=alpha_coeff_map,
            alpha_power=alpha_power_map,
        ),
    )
    medium.plot(export_path=Path(work_dir / "medium.png"))

    # drive the middle half of the aperture
    transducer.active_source_elements[:] = False
    transducer.active_source_elements[16:48] = True

    # focus two thirds of the way down the tissue, on the axis
    tissue_depth_m = domain_size[0] - transducer.face_depth_m
    focus_depth_m = transducer.face_depth_m + tissue_depth_m * 2 / 3
    transducer.focus(focus_m=(focus_depth_m, domain_size[1] / 2), apodization=0.5)

    transducer.plot_source_mask(export_path=work_dir / "source_transducer.svg")
    transducer.plot_sensor_mask(export_path=work_dir / "sensor_transducer.svg")

    # make a sensor for whole domain to make an animation
    sensor_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    sensor_mask[:, :] = True
    sensor = fullwave.Sensor(mask=sensor_mask, sampling_modulus_time=2)
    sensor.plot(export_path=work_dir / "sensor_whole.svg")

    #
    # --- run simulation ---
    #

    # input source and sensor separately for animation
    fw_solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        transducer=transducer,
        sensor=sensor,
        run_on_memory=False,
    )
    sensor_output = fw_solver.run()

    #
    # --- visualization ---
    #

    propagation_map = signal_process.reshape_whole_sensor_to_nt_nx_ny(sensor_output, grid)
    p_max_plot = np.abs(propagation_map).max().item() / 2

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
        export_name=work_dir / "wave_propagation.mp4",
        vmax=p_max_plot,
        vmin=-p_max_plot,
        figsize=(4, 6),
    )


if __name__ == "__main__":
    main()
