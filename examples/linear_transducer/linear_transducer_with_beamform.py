"""Send one plane wave, then beamform the element traces into an image.

The medium holds seven point targets and a speckle pattern, so the image shows both.

Run it with:
    uv run python examples/linear_transducer/linear_transducer_with_beamform.py
"""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave.beamformer.beamformer import Beamformer
from fullwave.utils import plot_utils


def main() -> None:
    """Run one plane wave transmit and beamform the result."""
    # overwrite the logging level, DEBUG, INFO, WARNING, ERROR
    logging.getLogger("__main__").setLevel(logging.INFO)

    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "linear_transducer_with_beamform"
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- define the computational grid ---
    domain_size = (42.5e-3, 42.5e-3)  # meters
    f0 = 2e6
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
    # put point targetes
    point_target_locations_m = [
        (10e-3, domain_size[1] / 2),
        (15e-3, domain_size[1] / 2),
        (20e-3, domain_size[1] / 2),
        (25e-3, domain_size[1] / 2),
        (30e-3, domain_size[1] / 2),
        (35e-3, domain_size[1] / 2),
        (40e-3, domain_size[1] / 2),
    ]
    for loc in point_target_locations_m:
        ix = int(loc[0] / grid.dx)
        iy = int(loc[1] / grid.dx)
        sound_speed_map[ix, iy] = sound_speed * 0.6

    density_map = density * np.ones((grid.nx, grid.ny))

    alpha_coeff_map = alpha_coeff * np.ones((grid.nx, grid.ny))
    alpha_power_map = alpha_power * np.ones((grid.nx, grid.ny))
    beta_map = beta * np.ones((grid.nx, grid.ny))

    rng = np.random.default_rng(seed=42)
    scatterer, _ = fullwave.utils.generate_scatterer(
        grid=grid,
        ratio_scatterer_to_total_grid=0.38,
        scatter_value_std=0.035 / 2,
        rng=rng,
    )

    density_map *= scatterer

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

    transducer.plot_source_mask(work_dir / "source_transducer.svg")
    transducer.plot_sensor_mask(work_dir / "sensor_transducer.svg")

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

    num_elements = transducer.transducer_geometry.number_elements
    aperture_m = transducer.transducer_geometry.transducer_width_m

    lateral_position = np.arange(
        -aperture_m / 2,
        aperture_m / 2,
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
        export_path=work_dir / "beamformed_image.png",
    )


if __name__ == "__main__":
    main()
