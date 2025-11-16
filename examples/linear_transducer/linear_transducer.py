"""Simple plane wave transmit example."""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave.utils import plot_utils


def main() -> None:  # noqa: PLR0915
    """Run linear transducer with focused transmit example."""
    # overwrite the logging level, DEBGUG, INFO, WARNING, ERROR
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

    element_layer_px = 3
    # --- define the linear transducer ---
    transducer_geometry = fullwave.TransducerGeometry(
        grid,
        number_elements=128,
        # -
        element_width_m=0.146484375e-3,
        # element_width_px=6,  # depends on the ppw, cfl
        # -
        element_spacing_m=0.146484375e-3,
        # element_spacing_px=6,
        # -
        element_layer_px=element_layer_px,
        # -
        # [axial, lateral]
        # position_px=(0, 0),
        position_m=(
            (42.5 - 37.5) / 2 * 1e-3,
            (42.5 - 37.5) / 2 * 1e-3,
        ),
        # -
        radius=float("inf"),
        average_surface_signals=True,
    )
    transducer = fullwave.Transducer(
        transducer_geometry=transducer_geometry,
        grid=grid,
    )
    p_max = 1e5

    angle = 0
    # length = 1000000
    length = int(grid.nx * (9 / 10))
    target_location_px = np.array(
        [
            # focus transmit
            # int(grid.nx * (9 / 10)),
            # grid.ny // 2,
            #
            # plane wave
            # 1000000,
            # grid.ny // 2,
            # plane wave with angle
            length * np.cos(np.deg2rad(angle)),
            length * np.sin(np.deg2rad(angle)) + grid.ny // 2,
        ],
        dtype=int,
    )

    active_source_elements = np.zeros(transducer_geometry.number_elements, dtype=bool)
    active_sensor_elements = np.zeros(transducer_geometry.number_elements, dtype=bool)
    active_source_elements[:] = True
    # active_source_elements[32:96] = True
    active_sensor_elements[:] = True

    input_signal = np.zeros((transducer.n_sources, grid.nt))
    dict_source_index_to_location = transducer.dict_source_index_to_location
    element_id_to_element_center = transducer.element_id_to_element_center

    delay_list = []
    for i_source_index in range(len(input_signal)):
        source_location = dict_source_index_to_location[i_source_index + 1]
        element_id = transducer.transducer_geometry.indexed_element_mask_input[*source_location]
        source_location = element_id_to_element_center[element_id]

        delay_sec = np.sqrt(np.sum((target_location_px - source_location) ** 2)) * grid.dx / c0
        delay_list.append(delay_sec)
    delay_list = np.array(delay_list)
    delay_list = delay_list.max() - delay_list
    delay_list = delay_list - delay_list.min()

    for i_source_index in range(len(input_signal)):
        delay_sec = delay_list[i_source_index]
        source_location = dict_source_index_to_location[i_source_index + 1]

        n_y = input_signal.shape[0] // element_layer_px
        i_layer = i_source_index // n_y
        element_id = transducer.transducer_geometry.indexed_element_mask_input[*source_location]
        if not active_source_elements[element_id - 1]:
            p0_vec = np.zeros(grid.nt)
        else:
            p0_vec = fullwave.utils.pulse.gaussian_modulated_sinusoidal_signal(
                nt=grid.nt,
                f0=f0,
                duration=duration,
                ncycles=2,
                drop_off=2,
                p0=p_max,
                i_layer=i_layer,
                dt_for_layer_delay=grid.dt,
                cfl_for_layer_delay=grid.cfl,
                delay_sec=delay_sec,
            )
        input_signal[i_source_index, :] = p0_vec.copy()

    transducer.set_signal(input_signal)
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
