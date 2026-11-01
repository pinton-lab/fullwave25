"""Acquire a full synthetic aperture with an L7-4 linear array and beamform it.

One element transmits and every element records, once for each of the 128
transmits.

Run it with:
    uv run python examples/linear_transducer/full_synthetic_aperture.py
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import fullwave
from fullwave.beamformer import MachBeamformer
from fullwave.utils import plot_utils


def convert_to_db(signal: np.ndarray) -> np.ndarray:
    """Convert signal to decibel scale.

    Parameters
    ----------
    signal : np.ndarray
        Input signal array.

    Returns
    -------
    np.ndarray
        Signal in decibel scale.

    """
    return 20 * np.log10(np.abs(signal) / np.max(np.abs(signal)))


def make_echoic_targets(
    scatterer: np.ndarray,
    grid: fullwave.Grid,
    target_radius_m: float,
    target_spacing_m: float,
    start_offset_x_m: float,
    start_offset_y_m: float,
    n_targets_axial: int,
    n_targets_lateral: int,
    *,
    face_depth_m: float = 0.0,
    centered: bool = True,
) -> np.ndarray:
    """Place echoic circles in the scatter map.

    Parameters
    ----------
    scatterer : np.ndarray
        Scatterer map to modify.
    grid : fullwave.Grid
        Computational grid.
    target_radius_m : float
        Radius of each target in meters.
    target_spacing_m : float
        Spacing between targets in meters.
    start_offset_x_m : float
        Starting offset in the axial direction in meters.
    start_offset_y_m : float
        Starting offset in the lateral direction in meters.
    n_targets_axial : int
        Number of targets in the axial direction.
    n_targets_lateral : int
        Number of targets in the lateral direction.
    face_depth_m : float, optional
        Depth of the probe face in meters. Centered targets sit in the tissue below it.
    centered : bool, optional
        Whether to center the targets in the tissue below the probe face (default: True).

    Returns
    -------
    np.ndarray
        Modified scatterer map with echoic targets.

    """
    # place echoic circles in the scatter map
    # each circle has a target_radius_m and is spaced by target_spacing_m
    if centered:
        total_axial_length = (n_targets_axial - 1) * target_spacing_m
        total_lateral_length = (n_targets_lateral - 1) * target_spacing_m
        tissue_depth_m = grid.shape[0] * grid.dx - face_depth_m
        start_offset_x_m = face_depth_m + (tissue_depth_m - total_axial_length) / 2
        start_offset_y_m = (grid.shape[1] * grid.dx - total_lateral_length) / 2

    axial_positions = np.linspace(
        start_offset_x_m,
        start_offset_x_m + (n_targets_axial - 1) * target_spacing_m,
        n_targets_axial,
        endpoint=True,
    )
    lateral_positions = np.linspace(
        start_offset_y_m,
        start_offset_y_m + (n_targets_lateral - 1) * target_spacing_m,
        n_targets_lateral,
        endpoint=True,
    )

    echoic_target_max = 4
    echoic_target_min = 2

    hypo_echoic_target_max = 0.6
    hypo_echoic_target_min = 0.4

    anechoic_target_max = 0.0
    anechoic_target_min = 0.0
    hyper_echoic_ratio_array = np.linspace(
        echoic_target_max,
        echoic_target_min,
        n_targets_axial,
    )
    hypo_echoic_ratio_array = np.linspace(
        hypo_echoic_target_max,
        hypo_echoic_target_min,
        n_targets_axial,
    )
    anechoic_ratio_array = np.linspace(
        anechoic_target_max,
        anechoic_target_min,
        n_targets_axial,
    )
    echoic_ratio_array = np.zeros((n_targets_axial, n_targets_lateral))
    for i in range(n_targets_axial):
        echoic_ratio_array[i, 0] = anechoic_ratio_array[i]
        echoic_ratio_array[i, 1] = hypo_echoic_ratio_array[i]
        echoic_ratio_array[i, 2] = hyper_echoic_ratio_array[i]

    for i_axial, axial_pos in enumerate(axial_positions):
        for i_lateral, lateral_pos in enumerate(lateral_positions):
            axial_idx = int(axial_pos / grid.dx)
            lateral_idx = int(lateral_pos / grid.dx)
            (
                rr,
                cc,
            ) = np.ogrid[
                -axial_idx : scatterer.shape[0] - axial_idx,
                -lateral_idx : scatterer.shape[1] - lateral_idx,
            ]
            circle_mask = rr**2 + cc**2 <= (target_radius_m / grid.dx) ** 2

            scatterer -= 1.0
            scatterer[circle_mask] *= echoic_ratio_array[i_axial, i_lateral]
            scatterer += 1.0

    return scatterer


def main() -> None:
    """Run linear trransducer full synthetic aperture simulation and beamforming."""
    # overwrite the logging level, DEBUG, INFO, WARNING, ERROR
    # logging.getLogger("__main__").setLevel(logging.DEBUG)
    logging.getLogger("__main__").setLevel(logging.INFO)

    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "full_synthetic_aperture"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- define the computational grid ---
    #

    f0 = 2e6
    c0 = 1540
    ppw = 12
    width_m = 45e-3  # lateral width [m], wider than the array
    domain_size = (
        4.5e-2,
        width_m,
    )  # [axial, lateral] meters
    duration = domain_size[0] / c0 * 2.3
    cfl = 0.4

    grid = fullwave.Grid(domain_size, f0, duration, c0=c0, ppw=ppw, cfl=cfl)

    #
    # --- define the acoustic medium properties ---
    #
    sound_speed_map = np.ones(grid.shape) * c0
    density_map = np.ones(grid.shape) * 1000
    alpha_coeff_map = np.ones(grid.shape) * 0.5  # dB/(MHz^y cm)
    alpha_power_map = np.ones(grid.shape) * 1.1  # y
    beta_map = np.ones(grid.shape) * 0  # nonlinearity parameter
    air_map = np.zeros(grid.shape)  # no air bubbles

    sampling_interval = 7
    transducer = fullwave.Transducer.l7_4(
        grid,
        sampling_modulus_time=sampling_interval,
    )
    transducer_width_m = transducer.transducer_geometry.transducer_width_m

    # define scatterer

    rng = np.random.default_rng(seed=42)

    scatterer, _ = fullwave.utils.generate_scatterer(
        grid=grid,
        ratio_scatterer_to_total_grid=0.38,
        scatter_value_std=0.035,
        rng=rng,
    )

    scatterer = make_echoic_targets(
        scatterer=scatterer,
        grid=grid,
        target_radius_m=5e-3,
        target_spacing_m=15e-3,
        start_offset_x_m=10e-3,
        start_offset_y_m=10e-3,
        n_targets_axial=3,
        n_targets_lateral=3,
        face_depth_m=transducer.face_depth_m,
        centered=True,
    )
    plot_utils.plot_array(scatterer, export_path=work_dir / "scatterer.png")
    # scatterer modulates the density map.
    # scatterer values are centered around 1.0 with small variations.
    density_map *= scatterer

    medium = fullwave.Medium(
        grid,
        beta=beta_map,
        air_map=air_map,
        **transducer.fill_probe_above_face(
            sound_speed=sound_speed_map,
            density=density_map,
            alpha_coeff=alpha_coeff_map,
            alpha_power=alpha_power_map,
        ),
        n_jobs=1,
    )
    medium.plot(export_path=work_dir / "medium.svg")
    #
    # --- define the linear transducer ---
    #

    active_source_element_id_list = list(range(128))
    # active_source_element_id_list = active_source_element_id_list[32::32]  # for faster demo

    #
    # --- run simulation ---
    #

    sensor_output_list = []
    for i_active_source_element_id, active_source_element_id in tqdm(
        enumerate(active_source_element_id_list),
        total=len(active_source_element_id_list),
    ):
        # one element transmits, every element records
        transducer.synthetic_aperture(active_source_element_id + 1)
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
        else:
            sensor_output = fw_solver.run(
                simulation_dir_name=f"txrx_{i_active_source_element_id}",
                is_static_map=True,
                recalculate_pml=False,  # Reuse PML from first run
            )
        sensor_output = fw_solver.transducer.post_process_sensor_output(
            sensor_output,
            average_surface_signals=True,
        )
        sensor_output_list.append(sensor_output)
        # if remove_dir is True:
        #     shutil.rmtree(work_dir / f"txrx_{i_angle}")
    #
    # --- beamform with mach ---
    #

    # -- define the beamforming parameters --

    transducer_width_m = transducer.transducer_geometry.transducer_width_m

    b_mode_x = np.linspace(-transducer_width_m / 2, transducer_width_m / 2, num=256, endpoint=True)
    # depth below the probe face, which is the first medium row
    b_mode_z = np.linspace(
        5e-3,
        domain_size[0] - transducer.face_depth_m,
        num=512,
        endpoint=True,
    )

    beamformer = MachBeamformer.from_transducer(
        transducer,
        lateral_position_m=b_mode_x,
        axial_position_m=b_mode_z,
        f_number=2.0,
    )
    # element ids count from zero here, and from one in the beamformer
    tx_arrival_s = np.stack(
        [
            beamformer.synthetic_aperture_arrival_s(element=element_id + 1)
            for element_id in active_source_element_id_list
        ],
        axis=0,
    )
    sensor_output = np.stack(sensor_output_list, axis=0)
    iq_data = beamformer.rf_to_iq(sensor_output)
    # [n_axial, n_lateral] to [n_lateral, n_axial]
    b_mode = beamformer.beamform(sensor_output, tx_arrival_s=tx_arrival_s).T

    # plot the IQ data of the second transmit
    plt.close("all")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(
        convert_to_db(iq_data[1]).T,
        vmin=-100,
        vmax=0,
        aspect=0.3,
        cmap="gray",
    )
    ax.set_xlabel("Element index")
    ax.set_ylabel("Time samples")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Amplitude [dB]")
    plt.tight_layout()
    plt.savefig(work_dir / "iq_data.png", dpi=300)

    #
    # --- visualization ---
    #

    plot_utils.plot_array(
        convert_to_db(b_mode).T,
        vmin=-30,
        vmax=0,
        aspect=1,
        cmap="gray",
        extent=[
            b_mode_x[0] * 1e3,
            b_mode_x[-1] * 1e3,
            b_mode_z[-1] * 1e3,
            b_mode_z[0] * 1e3,
        ],
        xlabel="Lateral position (mm)",
        ylabel="Axial position (mm)",
        colorbar=True,
        export_path=work_dir / "beamformed_image.svg",
    )
    plot_utils.plot_array(
        convert_to_db(b_mode).T,
        vmin=-30,
        vmax=0,
        aspect=1,
        cmap="gray",
        extent=[
            b_mode_x[0] * 1e3,
            b_mode_x[-1] * 1e3,
            b_mode_z[-1] * 1e3,
            b_mode_z[0] * 1e3,
        ],
        xlabel="Lateral position (mm)",
        ylabel="Axial position (mm)",
        colorbar=True,
        export_path=work_dir / "beamformed_image.png",
    )
    np.savez(
        work_dir / "simulation_data.npz",
        b_mode=b_mode,
        b_mode_x=b_mode_x,
        b_mode_z=b_mode_z,
        iq_data=iq_data,
    )


if __name__ == "__main__":
    main()
