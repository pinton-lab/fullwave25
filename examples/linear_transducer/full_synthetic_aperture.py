"""Full synthetic aperture example."""

import logging
from pathlib import Path

import cupy
import matplotlib.pyplot as plt
import numpy as np
import pymust
from einops import rearrange
from mach import experimental, wavefront
from mach.io.must import (
    linear_probe_positions,
    scan_grid,
)
from tqdm import tqdm

import fullwave
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
    centered : bool, optional
        Whether to center the targets in the grid (default: True).

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
        start_offset_x_m = (grid.shape[0] * grid.dx - total_axial_length) / 2
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
    work_dir = Path("./outputs/") / "linear_transducer_fsa_1"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- define the computational grid ---
    #

    domain_size = (4.5e-2, 4.5e-2)  # [axial, lateral] meters
    f0 = 2e6
    c0 = 1540
    duration = domain_size[0] / c0 * 2.3
    ppw = 12
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
        centered=True,
    )
    plot_utils.plot_array(scatterer)
    # scatterer modulates the density map.
    # scatterer values are centered around 1.0 with small variations.
    density_map *= scatterer

    medium = fullwave.Medium(
        grid,
        sound_speed=sound_speed_map,
        density=density_map,
        alpha_coeff=alpha_coeff_map,
        alpha_power=alpha_power_map,
        beta=beta_map,
        air_map=air_map,
        n_jobs=1,
    )
    medium.plot(export_path=work_dir / "medium.svg")
    #
    # --- define the linear transducer ---
    #

    active_source_element_id_list = list(range(128))
    # active_source_element_id_list = active_source_element_id_list[32::32]  # for faster demo

    sampling_interval = 7
    transducer = fullwave.Transducer.l7_4(grid, sampling_modulus_time=sampling_interval)
    transducer_width_m = transducer.transducer_geometry.transducer_width_m

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

    params = {
        "fs": 1 / grid.dt / sampling_interval,  # Sampling frequency (Hz)
        "fc": f0,  # Center frequency (Hz)
        "Nelements": 128,  # Number of array elements
        "pitch": transducer_width_m / 128,  # Element pitch (meters)
        "c": c0,  # Speed of sound in medium (m/s)
        # "fnumber": 1.0,
        "fnumber": 2.0,
        "t0": 0,  # Start time of data acquisition (seconds)
    }

    element_positions = linear_probe_positions(params["Nelements"], params["pitch"])
    b_mode_x = np.linspace(-transducer_width_m / 2, transducer_width_m / 2, num=256, endpoint=True)
    b_mode_y = np.array([0.0])  # 2D imaging (single y-plane)
    b_mode_z = np.linspace(5e-3, domain_size[0], num=512, endpoint=True)

    # make grid points for beamforming
    grid_points = scan_grid(b_mode_x, b_mode_y, b_mode_z)

    # -- compute transmit wavefront arrival times --
    wavefront_arrivals_s = [
        wavefront.spherical(
            origin_m=element_positions[i_elem],
            points_m=grid_points,
            focus_m=element_positions[i_elem],  # FSA: focus at each element position
        )
        / params["c"]
        for i_elem in active_source_element_id_list
    ]
    for i in range(len(wavefront_arrivals_s)):
        wavefront_arrivals_s[i] -= wavefront_arrivals_s[i].min()
    wavefront_arrivals_s = np.stack(wavefront_arrivals_s, axis=0)

    # --- post-process and beamform ---

    sensor_output = np.stack(sensor_output_list, axis=0)
    sensor_output = sensor_output.transpose(2, 1, 0)

    # Convert RF data to IQ data
    iq_data = pymust.rf2iq(sensor_output, Fs=params["fs"], Fc=params["fc"])

    # Convert the IQ data to flattened grid points for beamforming
    iq_data = iq_data[:, :, :, np.newaxis]
    iq_data = np.ascontiguousarray(
        rearrange(
            iq_data,
            "samples elements transmit frames -> transmit elements samples frames",
        ),
        # iq_data,
        dtype=np.complex64,
    )

    # run beamforming on GPU.
    b_mode = experimental.beamform(
        channel_data=cupy.asarray(iq_data),  # IQ data from all elements
        rx_coords_m=cupy.asarray(element_positions),  # Array element positions
        scan_coords_m=cupy.asarray(grid_points),  # Imaging grid coordinates
        tx_wave_arrivals_s=cupy.asarray(
            wavefront_arrivals_s,
        ),  # Transmit arrival times (s)
        f_number=float(params["fnumber"]),  # Dynamic focusing f-number
        rx_start_s=float(params["t0"]),  # Data acquisition start time
        sampling_freq_hz=float(params["fs"]),  # Sampling frequency
        sound_speed_m_s=float(params["c"]),  # Medium sound speed
        modulation_freq_hz=float(params["fc"]),  # Demodulation frequency
        tukey_alpha=0.5,  # No additional windowing
    )

    # convert cupy to numpy
    b_mode = cupy.asnumpy(b_mode)
    # reshape to 2D image
    b_mode = b_mode.reshape(len(b_mode_x), len(b_mode_z))

    # plot convert_to_db(iq_data[0, :, :, 0]).T,
    plt.close("all")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(
        convert_to_db(iq_data[1, :, :, 0]).T,
        vmin=-100,
        vmax=0,
        aspect=0.3,
        cmap="gray",
    )
    ax.set_xlabel("Element index")
    ax.set_ylabel("Time samples")
    cbar = fig.colorbar(im, ax=ax, orientation="vertical", pad=0.02)
    cbar.set_label("Amplitude [dB]")
    # plt.subplots_adjust(left=0.05, right=1.08, top=0.95, bottom=0.15)
    plt.tight_layout()
    # plt.savefig(work_dir / "iq_data.svg", dpi=300)
    # plt.savefig("./temp/temp.png", dpi=300)
    plt.savefig("./temp/iq_fsa.png", dpi=300)

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
