"""Acquire a full synthetic aperture with a C5-2V curved array.

One element transmits and every element records, once for each transmit.

Run it with:
    uv run python examples/convex_transducer/convex_transducer_fsa.py
"""

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

import fullwave
from fullwave import MediumBuilder, presets
from fullwave.utils import plot_utils

logger = logging.getLogger("__main__." + __name__)


def db_scale(value: NDArray[np.float32], v_max: float = 1e5) -> NDArray[np.float32]:
    """Convert values to decibel scale.

    Parameters
    ----------
    value : NDArray[np.float32]
        Input values.
    v_max : float
        Maximum reference value.

    Returns
    -------
    NDArray[np.float32]
        Values in decibel scale.

    """
    return 20 * np.log10(np.abs(value) / v_max + 1e-10)


def main() -> None:
    """Run convex transducer abdominal wall example."""
    # overwrite the logging level, DEBGUG, INFO, WARNING, ERROR
    logging.getLogger("__main__").setLevel(logging.INFO)

    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "convex_transducer_fsa"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- define the computational grid ---
    #

    domain_size = (4e-2, 7e-2)  # meters
    f0 = 3.7e6
    c0 = 1540
    # duration = domain_size[0] / c0 * 2.3
    duration = domain_size[0] / c0 * 1.2
    ppw = 12
    cfl = 0.45
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0, ppw=ppw, cfl=cfl)

    #
    # --- define the convex transducer ---
    #

    sampling_modulus_time = 7
    transducer = fullwave.Transducer.c5_2v(
        grid,
        sampling_modulus_time=sampling_modulus_time,
    )
    air_map = transducer.make_suraface_reflective_with_air()

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
    rng = np.random.default_rng(seed=42)
    scatterer, _ = fullwave.utils.generate_scatterer(
        grid=grid,
        ratio_scatterer_to_total_grid=0.38,
        scatter_value_std=0.035,
        rng=rng,
    )

    background.density *= scatterer
    abdominal_wall.density *= scatterer

    # register the domains to MediumBuilder
    mb = MediumBuilder(
        grid=grid,
    )
    mb.register_domain(background)
    mb.register_domain(abdominal_wall)

    # we can plot to see the current registered domains

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
    medium.plot(export_path=work_dir / "medium.png")

    active_source_element_id_list = list(range(128))
    active_source_element_id_list = active_source_element_id_list[::16]  # for faster demo

    sensor_output_list = []
    for i_active_source_element_id, active_source_elements_id in tqdm(
        enumerate(active_source_element_id_list),
        desc="FSA simulation",
        total=len(active_source_element_id_list),
    ):
        # for active_source_elements_id in tqdm([63]):

        #
        # --- set the signal to transducer ---
        #
        # one element transmits, every element records
        transducer.synthetic_aperture(active_source_elements_id + 1)

        #
        # --- run simulation ---
        #
        fw_solver = fullwave.Solver(
            work_dir=work_dir,
            grid=grid,
            medium=medium,
            transducer=transducer,
            run_on_memory=False,
        )
        if i_active_source_element_id == 0:
            sensor_output = fw_solver.run(
                is_static_map=True,
                recalculate_pml=True,
            )
            sensor_output_list.append(sensor_output)
        else:
            sensor_output = fw_solver.run(
                simulation_dir_name=f"txrx_{i_active_source_element_id}",
                is_static_map=True,
                recalculate_pml=False,  # Reuse PML from first run
            )
            sensor_output_list.append(sensor_output)

    #
    # --- visualization ---
    #
    for sensor_output, active_source_element_id in zip(
        sensor_output_list,
        active_source_element_id_list,
        strict=False,
    ):
        sorted_sensor_output = transducer.post_process_sensor_output(sensor_output)
        np.save(
            work_dir / f"sensor_output_{active_source_element_id}.npy",
            arr=sorted_sensor_output,
        )

        plot_utils.plot_array(
            # np.power(np.abs(sorted_sensor_output.T), 1 / 3),
            db_scale(sorted_sensor_output.T, v_max=transducer.pulse.pressure),
            aspect=0.1,
            vmax=0,
            vmin=-80,
            export_path=work_dir / f"sensor_output_amplitude_{active_source_element_id}.png",
        )


if __name__ == "__main__":
    main()
