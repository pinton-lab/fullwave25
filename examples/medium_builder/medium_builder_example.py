"""Build a medium out of registered domains, then transmit through it.

`MediumBuilder` paints one domain at a time, so a phantom is stated as a list of
regions rather than as five raw arrays.

Run it with:
    uv run python examples/medium_builder/medium_builder_example.py
"""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave import MediumBuilder, presets
from fullwave.utils import plot_utils, signal_process


def main() -> None:
    """Build a medium from domains and run one focused transmit."""
    # overwrite the logging level, DEBUG, INFO, WARNING, ERROR
    logging.getLogger("__main__").setLevel(logging.INFO)

    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "medium_builder_example"
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- define the computational grid ---

    domain_size = (42.5e-3 * 1.5, 42.5e-3)  # [axial, lateral] meters
    f0 = 1e6
    c0 = 1540
    duration = domain_size[0] / c0 * 2.5
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0)

    # --- define the acoustic medium properties ---

    sound_speed = 1540
    density = 1100
    alpha_coeff = 0.6
    alpha_power = 1.2
    beta = 0.0

    # define background
    background = presets.BackgroundDomain(
        grid=grid,
        background_property_name=None,
    )

    # define simple domain 1
    geometry = np.zeros((grid.nx, grid.ny), dtype=int)
    geometry[
        round(grid.nx // 2 - grid.nx * 0.1) : round(grid.nx // 2 + grid.nx * 0.1),
        round(grid.ny // 2 - grid.ny * 0.1) : round(grid.ny // 2 + grid.ny * 0.1),
    ] = 1
    maps = {
        "sound_speed": sound_speed * geometry,
        "density": density * geometry,
        "alpha_coeff": alpha_coeff * geometry,
        "alpha_power": alpha_power * geometry,
        "beta": beta * geometry,
    }
    simple_domain_1 = presets.SimpleDomain(
        grid=grid,
        name="simple1",
        geometry=geometry,
        maps=maps,
    )

    # define simple domain 2
    geometry_2 = np.zeros((grid.nx, grid.ny), dtype=int)
    geometry_2[
        round(grid.nx // 4 - grid.nx * 0.1) : round(grid.nx // 4 + grid.nx * 0.1),
        round(grid.ny // 4 - grid.ny * 0.1) : round(grid.ny // 4 + grid.ny * 0.1),
    ] = 1
    maps_2 = {
        "sound_speed": sound_speed * geometry_2 * 0.9,
        "density": density * geometry_2 * 0.9,
        "alpha_coeff": 1.8 * geometry_2,
        "alpha_power": 1.8 * geometry_2,
        "beta": beta * geometry_2 * 0.9,
    }
    simple_domain_2 = presets.SimpleDomain(
        grid=grid,
        name="simple2",
        geometry=geometry_2,
        maps=maps_2,
    )

    # define scatterer
    scatterer = presets.ScattererDomain(
        grid=grid,
        num_scatterer=18,
        ncycles=2,
    )

    # scatterer will be applied to density directly, instead of registering as a domain
    csr = 0.035
    background.density += scatterer.density * csr
    # simple_domain_1.density = simple_domain_1.density - scatterer.density * csr
    simple_domain_2.density += scatterer.density * csr

    # register the domains to MediumBuilder
    mb = MediumBuilder(
        grid=grid,
    )
    mb.register_domain(background)
    mb.register_domain(simple_domain_1)
    mb.register_domain(simple_domain_2)

    # we can plot to see the current registered domains
    mb.plot_current_map(export_path=work_dir / "medium.png")

    # generate medium for simulation
    medium = mb.run()

    # --- define the linear transducer ---

    # A named array places itself, so only the transmit has to be stated.
    transducer = fullwave.Transducer.l7_4(grid)

    # --- define the transmit ---

    # focus nine tenths of the way down the domain, on the axis
    transducer.focus(focus_m=(domain_size[0] * 9 / 10, domain_size[1] / 2))

    # make a sensor for whole domain to make an animation
    sensor_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    sensor_mask[:, :] = True
    sensor = fullwave.Sensor(mask=sensor_mask, sampling_modulus_time=7)
    sensor.plot(export_path=work_dir / "sensor_whole.svg")

    # --- run simulation ---

    # input source and sensor separately for animation
    fw_solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        source=transducer.source,
        sensor=sensor,
        run_on_memory=False,
    )
    sensor_output = fw_solver.run()

    # --- visualization ---

    propagation_map = signal_process.reshape_whole_sensor_to_nt_nx_ny(sensor_output, grid)

    plot_utils.plot_wave_propagation_with_map(
        propagation_map=propagation_map,
        c_map=medium.sound_speed,
        rho_map=medium.density,
        export_name=work_dir / "wave_propagation.mp4",
    )


if __name__ == "__main__":
    main()
