"""Simple plane wave transmit example."""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave import MediumBuilder, presets
from fullwave.beamformer.beamformer import Beamformer
from fullwave.utils import plot_utils


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
    f0 = 2e6
    c0 = 1540
    duration = domain_size[0] / c0 * 2.5
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0)

    #
    # --- define the linear transducer ---
    #

    # A named array places itself, so only the transmit has to be stated.
    transducer = fullwave.Transducer.l7_4(grid)

    transducer.plane_wave()

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
