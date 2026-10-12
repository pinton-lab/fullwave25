"""Send several steered plane waves as one transmit batch and compound them into one image.

This is plane_wave_compounding.py with the loop of one `Solver.run` per angle replaced by
one `Solver.run_batch` over all angles. The medium, the transducer and the beamforming
are the same.

A batch needs a 2D batch solver binary, whose file name contains "batch". Until the
batch binaries are in a binary release, pass its path:

    uv run python examples/linear_transducer/plane_wave_compounding_batch.py \
        --batch-binary /path/to/fullwave2_2d_n_relax_multi_gpu_batch

or set FULLWAVE25_BATCH_BINARY to that path.
"""

import argparse
import logging
import os
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
from plane_wave_compounding import convert_to_db, make_echoic_targets

import fullwave
from fullwave.utils import plot_utils
from fullwave.utils.signal_filter import apply_filter


def parse_arguments() -> argparse.Namespace:
    """Read the path of the batch binary from the command line or the environment.

    Returns
    -------
    argparse.Namespace
        The arguments, with ``batch_binary`` as a Path.

    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--batch-binary",
        type=Path,
        default=os.environ.get("FULLWAVE25_BATCH_BINARY"),
        help="the 2D batch solver binary (default: $FULLWAVE25_BATCH_BINARY)",
    )
    arguments = parser.parse_args()
    if arguments.batch_binary is None:
        parser.error("pass --batch-binary or set FULLWAVE25_BATCH_BINARY")
    return arguments


def main() -> None:
    """Run the plane wave compounding example as one transmit batch."""
    arguments = parse_arguments()
    # overwrite the logging level, DEBUG, INFO, WARNING, ERROR
    logging.getLogger("__main__").setLevel(logging.INFO)

    #
    # define the working directory
    #
    work_dir = Path("./outputs/") / "plane_wave_compounding_batch"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- define the computational grid ---
    #

    # The calibrated L7-4 setup: 5.208 MHz, 12 points per wavelength, Courant 0.4.
    domain_size = (4.5e-2, 4.5e-2)  # [axial, lateral] meters
    f0 = 5.208e6
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
        scatter_value_std=0.02 / 2,
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
    plot_utils.plot_array(scatterer, export_path=work_dir / "scatterer.png")
    # scatterer modulates the density map.
    # scatterer values are centered around 1.0 with small variations.
    density_map *= scatterer

    #
    # --- define the linear transducer and its probe stack ---
    #

    sampling_interval = 7
    transducer = fullwave.Transducer.l7_4(
        grid,
        face_depth_m=fullwave.TransducerStack.backing_thickness_m,
        sampling_modulus_time=sampling_interval,
    )
    transducer_width_m = transducer.transducer_geometry.transducer_width_m
    transducer.apply_transducer_stack(
        sound_speed_map,
        density_map,
        alpha_coeff_map,
        alpha_power_map,
        beta_map,
        scatterer=scatterer,
        rng=rng,
    )

    medium = fullwave.Medium(
        grid,
        sound_speed=sound_speed_map,
        density=density_map,
        alpha_coeff=alpha_coeff_map,
        alpha_power=alpha_power_map,
        beta=beta_map,
        air_map=air_map,
    )
    medium.plot(export_path=work_dir / "medium.svg")

    #
    # --- run simulation ---
    #

    angles = [-10, -5, 0, 5, 10]  # degrees

    #
    # --- run all transmits as one batch ---
    #
    # One source per angle: plane_wave() only changes the signals of the same source
    # points, so the transmits can share one simulation.
    sources = []
    for angle in angles:
        transducer.plane_wave(angle_deg=angle, apodization=0.5)
        sources.append(transducer.source)

    fw_solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        transducer=transducer,
        path_fullwave_simulation_bin=arguments.batch_binary,
    )
    # [angles, sensor points, time samples]; at most 32 transmits run in one simulation.
    batch_output = fw_solver.run_batch(sources, simulation_dir_name="plane_waves")

    use_filter = True
    sensor_output_list = []
    for transmit_output in batch_output:
        sensor_output = transmit_output
        if use_filter:
            sensor_output = apply_filter(
                sensor_output,
                dt=grid.dt * transducer.sampling_modulus_time,  # effective dt after downsampling
                f_low_hz=0.05e6,
                use_gpu=True,
            )
        sensor_output = fw_solver.transducer.post_process_sensor_output(
            sensor_output,
            average_surface_signals=True,
        )
        sensor_output_list.append(sensor_output)

    #
    # --- b-mode imaging parameters ---
    #
    params = {
        "fs": 1 / grid.dt / sampling_interval,  # Sampling frequency (Hz)
        "fc": f0,  # Center frequency (Hz)
        "Nelements": 128,  # Number of array elements
        "pitch": transducer_width_m / 128,  # Element pitch (meters)
        "c": c0,  # Speed of sound in medium (m/s)
        "fnumber": 1.0,
        "t0": 0,  # Start time of data acquisition (seconds)
    }
    element_positions = linear_probe_positions(params["Nelements"], params["pitch"])
    b_mode_x = np.linspace(-transducer_width_m / 2, transducer_width_m / 2, num=256, endpoint=True)
    b_mode_y = np.array([0.0])  # 2D imaging (single y-plane)
    b_mode_z = np.linspace(5e-3, domain_size[0], num=512, endpoint=True)

    grid_points = scan_grid(b_mode_x, b_mode_y, b_mode_z)

    origin_m = np.array([0, 0, 0])
    wavefront_arrivals_s = []
    for angle in angles:
        direction = np.array([np.sin(np.radians(angle)), 0, np.cos(np.radians(angle))])
        at_scan_points_s = wavefront.plane(origin_m, grid_points, direction) / params["c"]
        at_aperture_s = wavefront.plane(origin_m, element_positions, direction) / params["c"]
        wavefront_arrivals_s.append(at_scan_points_s - at_aperture_s.min())
    wavefront_arrivals_s = np.stack(wavefront_arrivals_s, axis=0)

    # -- beamforming ---
    sensor_output = np.stack(sensor_output_list, axis=0)
    sensor_output = sensor_output.transpose(2, 1, 0)

    iq_data = pymust.rf2iq(sensor_output, Fs=params["fs"], Fc=params["fc"])

    # Convert to flattened grid points for beamforming
    iq_data = iq_data[:, :, :, np.newaxis]
    iq_data = np.ascontiguousarray(
        rearrange(
            iq_data,
            "samples elements transmit frames -> transmit elements samples frames",
        ),
        # iq_data,
        dtype=np.complex64,
    )

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
    b_mode = b_mode.reshape(len(b_mode_x), len(b_mode_z))

    # plot convert_to_db(iq_data[0, :, :, 0]).T,
    plt.close("all")
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    im = ax.imshow(
        convert_to_db(iq_data[0, :, :, 0]).T,
        vmin=-70,
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
        export_path=work_dir / "b_mode.png",
    )
    np.savez(
        work_dir / "simulation_data.npz",
        b_mode=b_mode,
        b_mode_x=b_mode_x,
        b_mode_z=b_mode_z,
        iq_data=iq_data,
    )
    print()


if __name__ == "__main__":
    main()
