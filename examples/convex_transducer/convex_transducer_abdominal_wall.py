"""Simple plane wave transmit example."""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave import MediumBuilder, presets
from fullwave.utils import plot_utils, signal_process


def main() -> None:  # noqa: PLR0915
    """Run convex transducer abdominal wall example."""
    # overwrite the logging level, DEBGUG, INFO, WARNING, ERROR
    logging.getLogger("__main__").setLevel(logging.INFO)

    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "convex_transducer"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- define the computational grid ---
    #

    domain_size = (4.5e-2, 6e-2)  # meters
    f0 = 3.7e6
    c0 = 1540
    duration = domain_size[0] / c0 * 1.0
    ppw = 12
    cfl = 0.4
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0, ppw=ppw, cfl=cfl)

    #
    # --- define the convex transducer ---
    #

    element_layer_px = 3
    transducer_geometry = fullwave.TransducerGeometry(
        grid,
        number_elements=128,
        # -
        element_width_m=0.0,
        # -
        element_spacing_m=0.508e-3,
        # -
        element_layer_px=element_layer_px,
        # -
        # [axial, lateral]
        position_m=(
            (42.5 - 37.5) / 2 * 1e-3,
            (42.5 - 37.5) / 2 * 1e-3,
        ),
        radius=0.04957,
        # -
    )
    p_max = 1e5

    transducer = fullwave.Transducer(
        transducer_geometry=transducer_geometry,
        grid=grid,
    )

    # make a sensor for whole domain to make an animation
    sensor_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    sensor_mask[:, :] = True
    sensor = fullwave.Sensor(mask=sensor_mask, sampling_modulus_time=2)
    sensor.plot(export_path=work_dir / "sensor_whole.svg")

    #
    # --- define focus point location ---
    #

    angle = 0
    # length = 1000000
    length = int(grid.nx * (9 / 10))
    target_location_px = np.array(
        [
            # focus transmit
            # int(grid.nx * (9 / 10)),
            # grid.ny // 2,
            #
            # plane wave (diverging wave)
            # 1000000,
            # grid.ny // 2,
            # plane wave with angle (diverging wave)
            length * np.cos(np.deg2rad(angle)),
            length * np.sin(np.deg2rad(angle)) + grid.ny // 2,
        ],
        dtype=int,
    )

    active_source_elements = np.zeros(transducer_geometry.number_elements, dtype=bool)
    # active_source_elements[95] = 1
    active_source_elements[32:96] = 1
    # active_source_elements[:] = 1

    input_signal = np.zeros((transducer.n_sources, grid.nt))
    dict_source_index_to_location = transducer.dict_source_index_to_location
    element_id_to_element_center = transducer.element_id_to_element_center

    delay_list = []
    for i_source_index in range(len(input_signal)):
        source_location = dict_source_index_to_location[i_source_index + 1]
        element_id = transducer.transducer_geometry.indexed_element_mask_input[*source_location]
        if not active_source_elements[element_id - 1]:
            delay_list.append(0)
            continue

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

    #
    # --- set the signal to transducer ---
    #

    transducer.set_signal(input_signal)

    #
    # --- define the acoustic medium properties ---
    #

    # define background
    background = presets.BackgroundDomain(
        grid=grid,
        background_property_name="liver",
    )

    # define abdominal wall
    abdominal_wall = presets.AbdominalWallDomain(
        grid=grid,
        start_depth=0,
        tranducer_surface=transducer.tranducer_surface,
    )

    # define scatterer
    scatterer = presets.ScattererDomain(
        grid=grid,
        num_scatterer=18,
        ncycles=2,
    )

    # scatterer will be applied to density directly, instead of registering as a domain
    csr = 0.035
    background.density[np.logical_not(transducer.tranducer_mask)] -= (
        scatterer.density[np.logical_not(transducer.tranducer_mask)] * csr
    )
    abdominal_wall.density -= scatterer.density * csr

    # register the domains to MediumBuilder
    mb = MediumBuilder(
        grid=grid,
    )
    mb.register_domain(background)
    mb.register_domain(abdominal_wall)

    # we can plot to see the current registered domains
    mb.plot_current_map(export_path=work_dir / "medium.png")

    # generate medium for simulation
    medium = mb.run()

    #
    # --- run simulation ---
    #

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

    propagation_map = signal_process.reshape_whole_sensor_to_nt_nx_ny(
        sensor_output,
        grid,
    )
    propagation_map = np.nan_to_num(propagation_map, 0, posinf=p_max, neginf=-p_max)

    p_max_plot = np.abs(propagation_map).max().item() / 8

    time_step = propagation_map.shape[0] // 3 * 2
    plot_utils.plot_wave_propagation_snapshot(
        propagation_map=propagation_map[time_step],
        c_map=medium.sound_speed,
        rho_map=medium.density,
        export_name=work_dir / "wave_propagation_snapshot_1.png",
        vmin=-p_max_plot,
        vmax=p_max_plot,
        turn_off_axes=True,
    )
    plot_utils.plot_wave_propagation_with_map(
        propagation_map=propagation_map,
        c_map=medium.sound_speed,
        rho_map=medium.density,
        export_name=work_dir / "wave_propagation.mp4",
        vmin=-p_max_plot,
        vmax=p_max_plot,
        figsize=(6, 4),
    )


if __name__ == "__main__":
    main()
