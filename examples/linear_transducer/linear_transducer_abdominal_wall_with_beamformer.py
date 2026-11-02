"""Send one plane wave through an abdominal wall, then beamform the traces with mach.

The beamformer runs on the GPU.

Run it with:
    uv run python examples/linear_transducer/linear_transducer_abdominal_wall_with_beamformer.py
"""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave import MediumBuilder, presets
from fullwave.beamformer import MachBeamformer, face_crossing_time_s, receiver_positions_m
from fullwave.utils import plot_utils


def main() -> None:
    """Run linear transducer abdominal wall example."""
    # overwrite the logging level, DEBUG, INFO, WARNING, ERROR
    logging.getLogger("__main__").setLevel(logging.INFO)

    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "linear_transducer_abdominal_wall_with_beamformer"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- define the computational grid ---
    #

    f0 = 2e6
    c0 = 1540
    width_m = 45e-3  # lateral width [m], wider than the array
    domain_size = (
        6e-2,
        width_m,
    )  # [axial, lateral] meters
    duration = domain_size[0] / c0 * 2.5
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0)

    #
    # --- define the linear transducer ---
    #

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
    # define abdominal wall, whose first layer starts at the probe face
    abdominal_wall = presets.AbdominalWallDomain(
        grid=grid,
        start_depth=transducer.face_depth_m,
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
    # register the probe fill last so it overrides the other domains
    mb.register_domain(transducer.probe_fill_domain())

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
        run_on_memory=True,
    )
    sensor_output = fw_solver.run()

    sensor_output = transducer.post_process_sensor_output(
        sensor_output,
        average_surface_signals=False,
    )

    lateral_position_m = np.arange(
        -domain_size[1] / 2,
        domain_size[1] / 2,
        grid.wavelength / 8,
    )
    # depths below the probe face
    image_start_m = 5 * grid.wavelength
    image_end_m = domain_size[0] - transducer.face_depth_m
    axial_position_m = np.arange(image_start_m, image_end_m, grid.wavelength / 8)

    beamformer = MachBeamformer(
        lateral_position_m=lateral_position_m,
        axial_position_m=axial_position_m,
        receiver_positions_m=receiver_positions_m(transducer),
        sound_speed=grid.c0,
        center_frequency_hz=grid.f0,
        sampling_frequency_hz=1 / (grid.dt * transducer.sampling_modulus_time),
        f_number=2.0,
        time_zero_s=face_crossing_time_s(transducer),
        tukey_alpha=0.5,
    )
    plane_wave_arrival_s = beamformer.plane_wave_arrival_s(angle_deg=0.0)
    beamformed_image = beamformer.beamform(sensor_output, tx_arrival_s=plane_wave_arrival_s)

    # --- visualization ---

    plot_utils.plot_array(
        20 * np.log10((np.abs(beamformed_image) + 1e-20) / np.abs(beamformed_image).max()),
        vmin=-30,
        vmax=0,
        cmap="gray",
        extent=[
            lateral_position_m[0] * 1e3,
            lateral_position_m[-1] * 1e3,
            axial_position_m[-1] * 1e3,
            axial_position_m[0] * 1e3,
        ],
        xlabel="Lateral position (mm)",
        ylabel="Depth (mm)",
        colorbar=True,
        export_path=work_dir / "beamformed_image.png",
    )
    print()


if __name__ == "__main__":
    main()
