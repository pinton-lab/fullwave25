"""Focus a linear array through an abdominal wall onto a region holding air.

Air reflects almost everything, so the movie shows the shadow behind the region.

Run it with:
    uv run python examples/linear_transducer/linear_transducer_abdominal_wall_with_air.py
"""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave import MediumBuilder, presets
from fullwave.constants import MaterialProperties
from fullwave.utils import plot_utils, signal_process


def main() -> None:
    """Run one focused transmit through an abdominal wall and a region of air."""
    # overwrite the logging level, DEBUG, INFO, WARNING, ERROR
    logging.getLogger("__main__").setLevel(logging.INFO)

    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "linear_transducer_abdominal_wall_with_air"
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

    # A named array places itself, so only the transmit has to be stated.
    transducer = fullwave.Transducer.l7_4(grid)

    #
    # --- define the transmit ---
    #

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

    rng = np.random.default_rng(seed=42)
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
    pressure = transducer.pulse.pressure
    propagation_map = np.nan_to_num(propagation_map, 0, posinf=pressure, neginf=-pressure)

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
