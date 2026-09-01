"""Focus an L7-4 linear array through an abdominal wall.

The sensor covers the whole domain, so the run writes a movie and a maximum
intensity projection of the field.

Run it with:
    uv run python examples/linear_transducer/linear_transducer_abdominal_wall.py
"""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave import MediumBuilder, presets
from fullwave.utils import plot_utils, signal_process


def main() -> None:
    """Run linear transducer abdominal wall example."""
    # overwrite the logging level, DEBUG, INFO, WARNING, ERROR
    logging.getLogger("__main__").setLevel(logging.DEBUG)

    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "linear_transducer_abdominal_wall"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- define the computational grid ---
    #

    domain_size = (6e-2, 6e-2)  # [axial, lateral] meters
    f0 = 1e6
    c0 = 1540
    duration = domain_size[0] / c0 * 2.3
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0)

    #
    # --- define the linear transducer ---
    #

    # A named array places itself, so only the transmit has to be stated.
    transducer = fullwave.Transducer.l7_4(grid)

    # focus nine tenths of the way down the domain, on the axis
    transducer.focus(focus_m=(domain_size[0] * 9 / 10, domain_size[1] / 2))

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

    csr = 0.05
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

    # register the domains to MediumBuilder
    mb = MediumBuilder(
        grid=grid,
        n_jobs=1,
    )
    mb.register_domain(background)
    mb.register_domain(abdominal_wall)
    # mb.register_domain(simple_domain_1)
    # mb.register_domain(simple_domain_2)

    # we can plot to see the current registered domains
    mb.plot_current_map(export_path=work_dir / "medium.svg")

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
    pressure = transducer.pulse.pressure
    propagation_map = np.nan_to_num(propagation_map, 0, posinf=pressure, neginf=-pressure)

    p_max_plot = np.abs(propagation_map).max().item() / 8

    time_step = propagation_map.shape[0] // 50 * 37
    plot_utils.plot_wave_propagation_snapshot(
        propagation_map=propagation_map[time_step],
        c_map=medium.sound_speed,
        rho_map=medium.density,
        export_name=work_dir / "wave_propagation_snapshot_1.svg",
        vmin=-p_max_plot,
        vmax=p_max_plot,
        turn_off_axes=True,
        figsize=(6, 6),
    )

    plot_utils.plot_wave_propagation_with_map(
        propagation_map=propagation_map,
        c_map=medium.sound_speed,
        rho_map=medium.density,
        export_name=work_dir / "wave_propagation.mp4",
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
