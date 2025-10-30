"""Simple plane wave transmit example."""

from pathlib import Path

import numpy as np

import fullwave
from fullwave.medium_builder import MediumBuilder, presets
from fullwave.solver.utils import load_dat_and_reshape
from fullwave.utils import plot_utils, signal_process


def main() -> None:  # noqa: PLR0915
    """Run Simple plane wave transmit example."""
    #
    # define the working directory
    #

    work_dir = Path("./outputs/") / "medium_builder_example_3d"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- define the computational grid ---
    #

    domain_size = (2.1e-2 / 10, 2.1e-2 / 10, 2.1e-2 / 10)  # meters
    f0 = 7.81e6 / 8
    c0 = 1540
    duration = domain_size[0] / c0 * 2.5
    ppw = 12
    cfl = 0.4
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0, ppw=ppw, cfl=cfl)

    #
    # --- define the acoustic medium properties ---
    #

    sound_speed = 1540
    density = 1100
    alpha_coeff = 0.5
    alpha_power = 1.0
    beta = 0.0

    # define background
    background = presets.BackgroundDomain(
        grid=grid,
        background_property_name=None,
    )

    # define simple domain 1
    geometry = np.zeros((grid.nx, grid.ny, grid.nz))
    geometry[
        round(grid.nx // 2 - grid.nx * 0.1) : round(grid.nx // 2 + grid.nx * 0.1),
        round(grid.ny // 2 - grid.ny * 0.1) : round(grid.ny // 2 + grid.ny * 0.1),
        round(grid.nz // 2 - grid.nz * 0.1) : round(grid.nz // 2 + grid.nz * 0.1),
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
    geometry_2 = np.zeros((grid.nx, grid.ny, grid.nz))
    geometry_2[
        round(grid.nx // 4 - grid.nx * 0.1) : round(grid.nx // 4 + grid.nx * 0.1),
        round(grid.ny // 4 - grid.ny * 0.1) : round(grid.ny // 4 + grid.ny * 0.1),
        round(grid.nz // 4 - grid.nz * 0.1) : round(grid.nz // 4 + grid.nz * 0.1),
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
    background.density -= scatterer.density * csr
    simple_domain_1.density -= scatterer.density * csr
    simple_domain_2.density -= scatterer.density * csr

    # register the domains to MediumBuilder
    mb = MediumBuilder(
        grid=grid,
        n_relaxation_mechanisms=2,
    )
    mb.register_domain(background)
    mb.register_domain(simple_domain_1)
    mb.register_domain(simple_domain_2)

    # generate medium for simulation
    medium = mb.run()

    medium.plot(export_path=work_dir / "medium.png")

    #
    # --- define the acoustic source ---
    #

    # define where to put the pressure source [nx, ny, nz]
    p_mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=bool)
    element_thickness_px = 3
    p_mask[0:element_thickness_px, :] = True

    # define the pressure source [n_sources, nt]d
    p0 = np.zeros((p_mask.sum(), grid.nt))  # [n_sources, nt]

    # The order of p_coordinates corresponds to the order of sources in p0
    # p_coordinates = map_to_coords(p_mask)

    for i_thickness in range(element_thickness_px):
        # create a gaussian-modulated sinusoidal pulse as the source signal with layer delay
        p0_vec = fullwave.utils.pulse.gaussian_modulated_sinusoidal_signal(
            nt=grid.nt,  # number of time steps
            f0=f0,  # center frequency [Hz]
            duration=duration,  # duration [s]
            ncycles=2,  # number of cycles
            drop_off=2,  # drop off factor
            p0=1e5,  # maximum amplitude [Pa]
            i_layer=i_thickness,
            dt_for_layer_delay=grid.dt,
            cfl_for_layer_delay=grid.cfl,
        )

        # assign the source signal to the corresponding layer
        p0[grid.ny * grid.nz * i_thickness : grid.ny * grid.nz * (i_thickness + 1), :] = (
            p0_vec.copy()
        )

    source = fullwave.Source(p0, p_mask)

    #
    # --- define the sensor ---
    #

    sensor_mask = np.zeros((grid.nx, grid.ny, grid.nz), dtype=bool)
    sensor_mask[:, :] = True
    sensor = fullwave.Sensor(mask=sensor_mask)

    #
    # --- run simulation ---
    #

    fw_solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        run_on_memory=False,
    )
    result_path: Path = fw_solver.run(load_results=False)
    sensor_output = load_dat_and_reshape(
        dat_file_path=result_path,
        n_sensors=sensor.n_sensors,
    )

    #
    # --- visualization ---
    #

    propagation_map = signal_process.reshape_whole_sensor_to_nt_nx_ny_nz(sensor_output, grid)
    np.save(work_dir / "propagation_map.npy", propagation_map)

    time_step = propagation_map.shape[0] // 2
    plot_utils.plot_array(
        propagation_map[time_step, :, :, grid.nz // 2],
        aspect=1,
        export_path=work_dir / "propagation_map.png",
    )


if __name__ == "__main__":
    main()
