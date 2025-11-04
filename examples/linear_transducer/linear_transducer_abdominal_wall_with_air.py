"""Simple plane wave transmit example."""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave import MediumBuilder, presets
from fullwave.constants import MaterialProperties
from fullwave.utils import plot_utils, signal_process


def main() -> None:  # noqa: PLR0915
    """Run linear transducer abdominal wall with air example."""
    # overwrite the logging level, DEBGUG, INFO, WARNING, ERROR
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
    f0 = 1e6
    c0 = 1540
    duration = domain_size[0] / c0 * 1.2
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

    # make a sensor for whole domain to make an animation
    sensor_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    sensor_mask[:, :] = True
    sensor = fullwave.Sensor(mask=sensor_mask, sampling_modulus_time=2)
    sensor.plot(export_path=work_dir / "sensor_whole.svg")

    #
    # --- define the acoustic medium properties ---
    #

    # define background
    background_property_name = "liver"
    material_properties = MaterialProperties()
    background = presets.BackgroundDomain(
        grid=grid,
        background_property_name=background_property_name,
    )
    # define abdominal wall
    abdominal_wall = presets.AbdominalWallDomain(
        grid=grid,
    )

    geometry = np.zeros((grid.nx, grid.ny))
    air_location = np.array(
        [
            [round(grid.nx // 3 * 2 - grid.nx * 0.1), round(grid.nx // 3 * 2 + grid.nx * 0.1)],
            [round(grid.ny // 2 - grid.ny * 0.2), round(grid.ny // 2 + grid.ny * 0.2)],
        ],
    )
    geometry[
        air_location[0][0] : air_location[0][1],
        air_location[1][0] : air_location[1][1],
    ] = 1
    sound_speed = getattr(material_properties, background_property_name)["sound_speed"]
    density = getattr(material_properties, background_property_name)["density"]
    alpha_coeff = getattr(material_properties, background_property_name)["alpha_coeff"]
    alpha_power = getattr(material_properties, background_property_name)["alpha_power"]
    beta = getattr(material_properties, background_property_name)["beta"]
    air_map = np.zeros((grid.nx, grid.ny), dtype=bool)

    rng = np.random.default_rng()
    random_location = rng.random((1000, 2))
    for loc in random_location:
        # x_idx = int(grid.nx // 2 - grid.nx * 0.1) + int(loc[0] * grid.nx * 0.4)
        # y_idx = int(grid.ny // 2 - grid.ny * 0.2) + int(loc[1] * grid.ny * 0.4)
        x_idx = air_location[0][0] + int(loc[0] * (air_location[0][1] - air_location[0][0]))
        y_idx = air_location[1][0] + int(loc[1] * (air_location[1][1] - air_location[1][0]))
        air_map[x_idx, y_idx] = True

    maps = {
        "sound_speed": sound_speed * geometry,
        "density": density * geometry,
        "alpha_coeff": alpha_coeff * geometry,
        "alpha_power": alpha_power * geometry,
        "beta": beta * geometry,
        "air": air_map,
    }
    air_domain = presets.SimpleDomain(
        grid=grid,
        name="air",
        geometry=geometry,
        maps=maps,
    )

    # define scatterer
    scatterer = presets.ScattererDomain(
        grid=grid,
        num_scatterer=18,
        ncycles=2,
    )

    # scatterer will be applied to density directly, instead of registering as a domain
    csr = 0.035
    background.density -= scatterer.density * csr
    abdominal_wall.density -= scatterer.density * csr
    air_domain.density -= scatterer.density * csr

    # register the domains to MediumBuilder
    mb = MediumBuilder(
        grid=grid,
    )
    mb.register_domain(background)
    mb.register_domain(abdominal_wall)
    mb.register_domain(air_domain)
    # mb.register_domain(simple_domain_1)
    # mb.register_domain(simple_domain_2)

    # we can plot to see the current registered domains
    mb.plot_current_map(export_path=work_dir / "medium.png")

    # generate medium for simulation
    medium = mb.run()

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

    propagation_map = signal_process.reshape_whole_sensor_to_nt_nx_ny(
        sensor_output,
        grid,
    )
    propagation_map = np.nan_to_num(propagation_map, 0, posinf=p_max, neginf=-p_max)

    p_max_plot = np.abs(propagation_map).max().item()

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
        export_name=work_dir / "wave_propagation_animation.mp4",
        vmin=-p_max_plot,
        vmax=p_max_plot,
        figsize=(6, 6),
    )

    # maximum intensity projection
    plot_utils.plot_array(
        np.max(np.abs(propagation_map**2), axis=0),
        aspect=propagation_map.shape[2] / propagation_map.shape[1],
        export_path=work_dir / "wave_propagation_mip.png",
    )


if __name__ == "__main__":
    main()
