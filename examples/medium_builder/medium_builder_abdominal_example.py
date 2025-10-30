"""Simple plane wave transmit example."""

from pathlib import Path

import numpy as np

import fullwave
from fullwave.medium_builder import MediumBuilder, presets
from fullwave.utils import plot_utils, signal_process


def main() -> None:
    """Run Simple plane wave transmit example."""
    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "medium_builder"
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
    element_layer_px = 1
    transducer_geometry = fullwave.TransducerGeometry(
        grid,
        number_elements=128,
        # -
        element_width_m=0.146484375e-3,
        # element_width_px=6,  # depends on the ppw, cfl
        # -
        element_spacing_m=0.146484375e-3,
        # element_spacing_px=6,
        # -
        element_layer_px=element_layer_px,
        # -
        # [axial, lateral]
        # position_px=(0, 0),
        position_m=(
            # (42.5 - 37.5) / 2 * 1e-3,
            0,
            (42.5 - 37.5) / 2 * 1e-3,
        ),
        # -
        radius=float("inf"),
    )
    p_max = 1e5
    # input_signal = generate_signal(input_signal, transducer_geometry)
    active_source_elements = np.zeros(transducer_geometry.number_elements, dtype=bool)
    active_sensor_elements = np.zeros(transducer_geometry.number_elements, dtype=bool)
    active_source_elements[32:96] = True
    # active_sensor_elements[32:96] = True
    # active_source_elements[:] = True
    active_sensor_elements[:] = True

    n_sources = transducer_geometry.n_sources_per_element * active_source_elements.sum()

    input_signal = np.zeros((n_sources, grid.nt))
    for i_layer in range(element_layer_px):
        p0_vec = fullwave.utils.pulse.gaussian_modulated_sinusoidal_signal(
            nt=grid.nt,
            f0=f0,
            duration=duration,
            ncycles=2,
            drop_off=2,
            p0=p_max,
            i_layer=i_layer + 1,
            dt_for_layer_delay=grid.dt,
            cfl_for_layer_delay=grid.cfl,
        )
        input_signal[i_layer::element_layer_px, :] = p0_vec.copy()

    transducer = fullwave.Transducer(
        transducer_geometry=transducer_geometry,
        grid=grid,
        input_signal=input_signal,
        active_source_elements=active_source_elements,
        active_sensor_elements=active_sensor_elements,
    )
    transducer.plot_source_mask(export_path=work_dir / "source_transducer.png")
    transducer.plot_sensor_mask(export_path=work_dir / "sensor_transducer.png")

    # make a sensor for whole domain to make an animation
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
