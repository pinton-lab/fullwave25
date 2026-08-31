"""Simple plane wave transmit example."""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave import MediumBuilder, presets
from fullwave.utils import plot_utils, signal_process


def main() -> None:
    """Run convex transducer abdominal wall example."""
    # overwrite the logging level, DEBUG, INFO, WARNING, ERROR
    logging.getLogger("__main__").setLevel(logging.INFO)

    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "convex_transducer"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- define the computational grid ---
    #

    domain_size = (4.5e-2, 7e-2)  # meters
    f0 = 3.7e6
    c0 = 1540
    duration = domain_size[0] / c0 * 1.0
    ppw = 12
    cfl = 0.4
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0, ppw=ppw, cfl=cfl)

    #
    # --- define the convex transducer ---
    #

    # A named array places itself, so only the transmit has to be stated.
    sampling_modulus_time = 7
    transducer = fullwave.Transducer.c5_2v(grid, sampling_modulus_time=sampling_modulus_time)
    air_map = transducer.make_suraface_reflective_with_air()

    # make a sensor for whole domain to make an animation
    sensor_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    sensor_mask[:, :] = True
    sensor = fullwave.Sensor(mask=sensor_mask, sampling_modulus_time=2)
    sensor.plot(export_path=work_dir / "sensor_whole.svg")

    #
    # --- define the transmit ---
    #

    # focus nine tenths of the way down the domain, on the axis
    transducer.focus(focus_m=(domain_size[0] * 9 / 10, domain_size[1] / 2))

    #
    # --- define the acoustic medium properties ---
    #

    # define background
    background = presets.BackgroundDomain(
        grid=grid,
        background_property_name="liver",
    )

    # define abdominal wall
    abdominal_wall = presets.AbdominalWallDomain(
        grid=grid,
        start_depth=0,
        transducer_surface=transducer.transducer_surface,
    )

    # define scatterer
    scatterer = presets.ScattererDomain(
        grid=grid,
        num_scatterer=18,
        ncycles=2,
    )

    # scatterer will be applied to density directly, instead of registering as a domain
    csr = 0.035
    background.density[np.logical_not(transducer.transducer_mask)] -= (
        scatterer.density[np.logical_not(transducer.transducer_mask)] * csr
    )
    abdominal_wall.density -= scatterer.density * csr

    # register the domains to MediumBuilder
    mb = MediumBuilder(
        grid=grid,
    )
    mb.register_domain(background)
    mb.register_domain(abdominal_wall)

    # we can plot to see the current registered domains
    mb.plot_current_map(export_path=work_dir / "medium.png")

    # generate medium for simulation
    medium = mb.run()
    # the transducer face reflects, so its air is added to the medium's own
    medium = fullwave.Medium(
        grid=grid,
        sound_speed=medium.sound_speed,
        density=medium.density,
        alpha_coeff=medium.alpha_coeff,
        alpha_power=medium.alpha_power,
        beta=medium.beta,
        air_map=np.logical_or(medium.air_map, air_map).astype(int),
    )

    #
    # --- run simulation ---
    #

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

    time_step = propagation_map.shape[0] // 3 * 2
    plot_utils.plot_wave_propagation_snapshot(
        propagation_map=propagation_map[time_step],
        c_map=medium.sound_speed,
        rho_map=medium.density,
        export_name=work_dir / "wave_propagation_snapshot_1.png",
        vmin=-p_max_plot,
        vmax=p_max_plot,
        turn_off_axes=True,
        # extent=(-domain_size[1] * 1e3 / 2, domain_size[1] * 1e3 / 2, domain_size[0] * 1e3, 0),
        # ylabel="Depth (mm)",
        # xlabel="Lateral position (mm)",
    )
    plot_utils.plot_wave_propagation_with_map(
        propagation_map=propagation_map,
        c_map=medium.sound_speed,
        rho_map=medium.density,
        export_name=work_dir / "wave_propagation.mp4",
        vmin=-p_max_plot,
        vmax=p_max_plot,
        figsize=(4, 3.5),
        extent=(-domain_size[1] * 1e3 / 2, domain_size[1] * 1e3 / 2, domain_size[0] * 1e3, 0),
        ylabel="Depth (mm)",
        xlabel="Lateral position (mm)",
    )
    print()


if __name__ == "__main__":
    main()
