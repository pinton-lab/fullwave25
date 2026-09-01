"""Build an abdominal wall with `MediumBuilder`, then transmit through it.

The wall is a preset domain, so the phantom needs two registered domains and no
raw maps.

Run it with:
    uv run python examples/medium_builder/medium_builder_abdominal_example.py
"""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave import MediumBuilder, presets
from fullwave.utils import plot_utils, signal_process


def main() -> None:
    """Build an abdominal wall from domains and run one focused transmit."""
    # overwrite the logging level, DEBUG, INFO, WARNING, ERROR
    logging.getLogger("__main__").setLevel(logging.INFO)

    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "medium_builder_abdominal_example"
    work_dir.mkdir(parents=True, exist_ok=True)

    # --- define the computational grid ---

    domain_size = (42.5e-3, 42.5e-3)  # [axial, lateral] meters
    f0 = 2e6
    c0 = 1540
    duration = domain_size[0] / c0 * 2.5
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0)

    # --- define the acoustic medium properties ---

    # define background
    background = presets.BackgroundDomain(
        grid=grid,
        background_property_name="liver",
    )
    # define abdominal wall
    abdominal_wall = presets.AbdominalWallDomain(
        grid=grid,
        start_depth=0,
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

    # register the domains to MediumBuilder
    mb = MediumBuilder(
        grid=grid,
    )
    mb.register_domain(background)
    mb.register_domain(abdominal_wall)

    # we can plot to see the current registered domains
    mb.plot_current_map(export_path=Path(work_dir / "medium.png"))

    # generate medium for simulation
    medium = mb.run()

    # --- define the linear transducer ---
    # A named array places itself, so only the transmit has to be stated.
    transducer = fullwave.Transducer.l7_4(grid)

    # --- define the transmit ---

    # focus nine tenths of the way down the domain, on the axis
    transducer.focus(focus_m=(domain_size[0] * 9 / 10, domain_size[1] / 2))

    sensor_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    sensor_mask[:, :] = True
    sensor = fullwave.Sensor(mask=sensor_mask, sampling_modulus_time=2)
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
    propagation_map = np.nan_to_num(propagation_map, 0)

    plot_utils.plot_wave_propagation_with_map(
        propagation_map=propagation_map,
        c_map=medium.sound_speed,
        rho_map=medium.density,
        export_name=work_dir / "wave_propagation.mp4",
        vmax=1e5,
        vmin=-1e5,
    )
    plot_utils.plot_array(
        propagation_map[500, :, :],
        aspect=propagation_map.shape[2] / propagation_map.shape[1],
        export_path=work_dir / "wave_propagation_snapshot.svg",
        clear_all=True,
    )


if __name__ == "__main__":
    main()
