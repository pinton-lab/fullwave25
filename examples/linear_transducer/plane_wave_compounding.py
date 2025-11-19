"""Plane wave compounding example."""

import logging
from pathlib import Path

import numpy as np
from tqdm import tqdm

import fullwave
from fullwave import MediumBuilder, presets
from fullwave.utils import plot_utils, signal_process


def make_angled_input_signal(
    angle_deg: float,
    length: int,
    grid: fullwave.Grid,
    transducer_geometry: fullwave.TransducerGeometry,
    transducer: fullwave.Transducer,
    element_layer_px: int,
    p_max: float = 1e5,
) -> np.ndarray:
    """Generate an angled input signal for plane wave transmission.

    Parameters
    ----------
    angle_deg : float
        Angle of the plane wave in degrees.
    length : int
        Length parameter for target location calculation.
    grid : fullwave.Grid
        Computational grid for the simulation.
    transducer_geometry : fullwave.TransducerGeometry
        Geometry of the transducer.
    transducer : fullwave.Transducer
        Transducer object containing source information.
    element_layer_px : int
        Number of pixels per element layer.
    p_max : float, optional
        Maximum pressure amplitude (default: 1e5).

    Returns
    -------
    np.ndarray
        Input signal array of shape (n_sources, nt).

    """
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
            length * np.cos(np.deg2rad(angle_deg)),
            length * np.sin(np.deg2rad(angle_deg)) + grid.ny // 2,
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

        delay_sec = np.sqrt(np.sum((target_location_px - source_location) ** 2)) * grid.dx / grid.c0
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
                f0=grid.f0,
                duration=grid.duration,
                ncycles=2,
                drop_off=2,
                p0=p_max,
                i_layer=i_layer,
                dt_for_layer_delay=grid.dt,
                cfl_for_layer_delay=grid.cfl,
                delay_sec=delay_sec,
            )
        input_signal[i_source_index, :] = p0_vec.copy()
    return input_signal


def main() -> None:
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
    f0 = 1e6
    c0 = 1540
    duration = domain_size[0] / c0 * 1.2
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0)

    #
    # --- define the linear transducer ---
    #
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

    scatterer, _ = fullwave.utils.generate_scatterer(
        grid=grid,
        ratio_scatterer_to_total_grid=0.38,
        scatter_value_std=0.035,
        rng=rng,
    )

    background.density *= scatterer
    abdominal_wall.density *= scatterer

    # register the domains to MediumBuilder
    mb = MediumBuilder(
        grid=grid,
    )
    mb.register_domain(background)
    mb.register_domain(abdominal_wall)

    # we can plot to see the current registered domains
    mb.plot_current_map(export_path=work_dir / "medium.svg")

    # generate medium for simulation
    medium = mb.run()

    #
    # --- run simulation ---
    #

    angles = [-15, -5, 0, 5, 15]  # degrees

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
    length = 1000000

    sensor_output_list = []
    for i_angle, angle in tqdm(enumerate(angles), total=len(angles)):
        input_signal = make_angled_input_signal(
            angle_deg=angle,
            length=length,
            grid=grid,
            transducer_geometry=transducer_geometry,
            transducer=transducer,
            element_layer_px=element_layer_px,
        )
        transducer.set_signal(input_signal)
        fw_solver = fullwave.Solver(
            work_dir=work_dir,
            grid=grid,
            medium=medium,
            transducer=transducer,
            sensor=sensor,
            run_on_memory=False,
        )
        if i_angle == 0:
            sensor_output = fw_solver.run(
                is_static_map=True,
                recalculate_pml=True,
            )
            sensor_output_list.append(sensor_output)
        else:
            sensor_output = fw_solver.run(
                simulation_dir_name=f"txrx_{i_angle}",
                is_static_map=True,
                recalculate_pml=False,  # Reuse PML from first run
            )
            sensor_output_list.append(sensor_output)

    #
    # --- visualization ---
    #
    for sensor_output, angle in zip(sensor_output_list, angles, strict=False):
        propagation_map = signal_process.reshape_whole_sensor_to_nt_nx_ny(
            sensor_output,
            grid,
        )
        propagation_map = np.nan_to_num(propagation_map, 0, posinf=p_max, neginf=-p_max)

        p_max_plot = np.abs(propagation_map).max().item() / 4

        time_step = propagation_map.shape[0] // 50 * 37
        plot_utils.plot_wave_propagation_snapshot(
            propagation_map=propagation_map[time_step],
            c_map=medium.sound_speed,
            rho_map=medium.density,
            export_name=work_dir / f"wave_propagation_snapshot_1_angle={angle:04d}.svg",
            vmin=-p_max_plot,
            vmax=p_max_plot,
            turn_off_axes=True,
            figsize=(6, 6),
        )

        plot_utils.plot_wave_propagation_with_map(
            propagation_map=propagation_map,
            c_map=medium.sound_speed,
            rho_map=medium.density,
            export_name=work_dir / f"wave_propagation_angle={angle:04d}.mp4",
            vmin=-p_max_plot,
            vmax=p_max_plot,
            figsize=(6, 6),
        )

        # maximum intensity projection
        plot_utils.plot_array(
            np.max(np.abs(propagation_map**2), axis=0),
            aspect=propagation_map.shape[2] / propagation_map.shape[1],
            export_path=work_dir / f"wave_propagation_mip_angle={angle:04d}.png",
        )


if __name__ == "__main__":
    main()
