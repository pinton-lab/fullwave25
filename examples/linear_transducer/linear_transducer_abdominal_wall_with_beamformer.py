"""Simple plane wave transmit example."""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave import MediumBuilder, presets
from fullwave.beamformer.beamformer import Beamformer
from fullwave.utils import plot_utils


def main() -> None:  # noqa: PLR0915
    """Run linear transducer abdominal wall example."""
    # overwrite the logging level, DEBUG, INFO, WARNING, ERROR
    logging.getLogger("__main__").setLevel(logging.INFO)

    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "linear_transducer"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- define the computational grid ---
    #

    domain_size = (6e-2, 6e-2)  # [axial, lateral] meters
    f0 = 2e6
    c0 = 1540
    duration = domain_size[0] / c0 * 2.5
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0)

    #
    # --- define the linear transducer ---
    #

    element_layer_px = 3
    transducer_geometry = fullwave.TransducerGeometry(
        grid,
        number_elements=128,
        # -
        element_width_m=0.146484375e-3,
        # -
        element_spacing_m=0.146484375e-3,
        # -
        element_layer_px=element_layer_px,
        # -
        # [axial, lateral]
        position_m=(
            0,
            (60 - 37.4) / 2 * 1e-3,
        ),
        # -
        radius=float("inf"),
    )
    transducer = fullwave.Transducer(
        transducer_geometry=transducer_geometry,
        grid=grid,
    )
    p_max = 1e5

    angle = 0
    length = 1000000
    # length = int(grid.nx * (9 / 10))
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

    # make a sensor for whole domain to make an animation
    sensor_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    sensor_mask[:, :] = True
    sensor = fullwave.Sensor(mask=sensor_mask, sampling_modulus_time=1)
    sensor.plot(export_path=work_dir / "sensor_whole.svg")

    #
    # --- define the acoustic medium properties ---
    #

    # define background
    background_property_name = "liver"
    background = presets.BackgroundDomain(
        grid=grid,
        background_property_name=background_property_name,
    )
    # define abdominal wall
    abdominal_wall = presets.AbdominalWallDomain(
        grid=grid,
    )

    # define scatterer

    rng = np.random.default_rng(seed=42)

    csr = 0.03
    ratio_scatterer_to_total_grid = 0.38

    scatterer, _ = fullwave.utils.generate_scatterer(
        grid=grid,
        # ratio_scatterer_to_total_grid=0.38,
        ratio_scatterer_to_total_grid=ratio_scatterer_to_total_grid,
        scatter_value_std=csr / 2,
        rng=rng,
    )

    background.density *= scatterer
    abdominal_wall.density *= scatterer
    background.beta = np.zeros_like(background.beta)
    abdominal_wall.beta = np.zeros_like(abdominal_wall.beta)

    # register the domains to MediumBuilder
    mb = MediumBuilder(
        grid=grid,
    )
    mb.register_domain(background)
    mb.register_domain(abdominal_wall)
    # mb.register_domain(simple_domain_1)
    # mb.register_domain(simple_domain_2)

    # we can plot to see the current registered domains

    # generate medium for simulation
    medium = mb.run()

    medium.plot(export_path=work_dir / "medium.svg")
    #
    # --- run simulation ---
    #

    # input source and sensor separately for animation
    fw_solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        transducer=transducer,
        # sensor=sensor,
        run_on_memory=False,
    )
    sensor_output = fw_solver.run()

    sensor_output = transducer.post_process_sensor_output(
        sensor_output,
        average_surface_signals=False,
    )

    num_elements = 128

    lateral_position = np.arange(
        # -transducer_geometry.transducer_width_m / 2,
        # transducer_geometry.transducer_width_m / 2,
        -domain_size[1] / 2,
        domain_size[1] / 2,
        grid.wavelength / 8,
    )
    axial_position = np.arange(domain_size[0] * 1 / 100, domain_size[0], grid.wavelength / 8)

    element_id_to_element_center = transducer.element_id_to_element_center
    transducer_coordinates = np.array(
        [element_id_to_element_center[i] for i in range(1, num_elements + 1)],
    )

    beamformer = Beamformer(
        c0=grid.c0,
        dx=grid.dx,
        dt=grid.dt,
        lateral_position_m=lateral_position,
        axial_position_m=axial_position,
        num_elements=num_elements,
        transducer_coordinates=transducer_coordinates,
        f_number=1.0,
    )
    beamformed_image = beamformer.run(sensor_output)

    # --- visualization ---

    plot_utils.plot_array(
        20 * np.log10((np.abs(beamformed_image) + 1e-20) / np.abs(beamformed_image).max()),
        vmin=-30,
        vmax=0,
        cmap="gray",
        extent=[
            lateral_position[0] * 1e3,
            lateral_position[-1] * 1e3,
            axial_position[-1] * 1e3,
            axial_position[0] * 1e3,
        ],
        xlabel="Lateral position (mm)",
        ylabel="Axial position (mm)",
        colorbar=True,
    )
    print()


if __name__ == "__main__":
    main()
