"""Send a plane wave through an imaging phantom and image the phantom with it.

The combined phantom holds point targets and contrast targets on a scattering background.
One plane wave straight ahead is recorded over the whole domain for a movie.
The same plane wave is beamformed into a B-mode image.

Run it with:
    uv run python examples/linear_transducer/plane_wave_phantom.py
"""

import logging
from pathlib import Path

import numpy as np

import fullwave
from fullwave.beamformer import MachBeamformer, face_crossing_time_s, receiver_positions_m
from fullwave.constants import MaterialProperties
from fullwave.utils import plot_utils, signal_process

F0_HZ = 5.208e6
MAX_MOVIE_RECORD_BYTES = 4e9
MAX_MOVIE_FRAMES = 200
DURATION_IN_DEPTH_TRANSITS = 2.3
MOVIE_DURATION_IN_DEPTH_TRANSITS = 1.3
MOVIE_COLOR_LIMIT_FRACTION = 1 / 20
BYTES_PER_RECORDED_VALUE = 8
PLANE_WAVE_ANGLE_DEG = 0
DOMAIN_SIZE_M = (5e-2, 4.5e-2)  # [axial, lateral]


def convert_to_db(signal: np.ndarray) -> np.ndarray:
    """Convert a signal to decibels relative to its maximum.

    Parameters
    ----------
    signal : np.ndarray
        Input signal array.

    Returns
    -------
    np.ndarray
        Signal in decibels.

    """
    return 20 * np.log10(np.abs(signal) / np.max(np.abs(signal)))


def movie_time_modulus(grid: fullwave.Grid) -> int:
    """Return the time sampling modulus that keeps the whole-domain record small.

    Parameters
    ----------
    grid : fullwave.Grid
        Computational grid.

    Returns
    -------
    int
        Every ``modulus``-th time step is recorded.

    """
    frames_in_budget = int(MAX_MOVIE_RECORD_BYTES / (grid.nx * grid.ny * BYTES_PER_RECORDED_VALUE))
    n_frames = min(MAX_MOVIE_FRAMES, frames_in_budget)
    return int(np.ceil(grid.nt / n_frames))


def contrast_target_circles_mm(
    phantom: fullwave.presets.CombinedPhantomDomain,
) -> list[tuple[float, float, float]]:
    """Return the outline of every contrast target of the phantom.

    Parameters
    ----------
    phantom : fullwave.presets.CombinedPhantomDomain
        Phantom whose targets are marked.

    Returns
    -------
    list[tuple[float, float, float]]
        (lateral position, depth, radius) in millimeters.

    """
    return [
        (lateral_m * 1e3, depth_m * 1e3, radius_m * 1e3)
        for lateral_m, depth_m, radius_m in phantom.contrast_target_circles_in_grid_m()
    ]


def point_targets_mm(
    phantom: fullwave.presets.CombinedPhantomDomain,
) -> list[tuple[float, float]]:
    """Return the position of every point target of the phantom.

    Parameters
    ----------
    phantom : fullwave.presets.CombinedPhantomDomain
        Phantom whose targets are marked.

    Returns
    -------
    list[tuple[float, float]]
        (lateral position, depth) in millimeters.

    """
    return [
        (lateral_m * 1e3, depth_m * 1e3)
        for lateral_m, depth_m in phantom.point_target_centers_in_grid_m()
    ]


def write_movie(
    *,
    grid: fullwave.Grid,
    medium: fullwave.Medium,
    transducer: fullwave.Transducer,
    circles_mm: list[tuple[float, float, float]],
    points_mm: list[tuple[float, float]],
    work_dir: Path,
) -> None:
    """Record one plane wave over the whole domain and write the movie.

    Parameters
    ----------
    grid : fullwave.Grid
        Computational grid.
    medium : fullwave.Medium
        Medium of the simulation.
    transducer : fullwave.Transducer
        Transducer that sends the plane wave.
    circles_mm : list[tuple[float, float, float]]
        (lateral position, depth, radius) of the circles drawn on every frame.
    points_mm : list[tuple[float, float]]
        (lateral position, depth) of the dots drawn on every frame.
    work_dir : Path
        Directory of the movie and of the snapshot.

    """
    transducer.plane_wave(angle_deg=PLANE_WAVE_ANGLE_DEG, apodization=0.5)
    sensor = fullwave.Sensor(
        mask=np.ones(grid.shape, dtype=bool),
        sampling_modulus_time=movie_time_modulus(grid),
    )
    solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        transducer=transducer,
        sensor=sensor,
        run_on_memory=True,
    )
    sensor_output = solver.run()
    propagation_map = signal_process.reshape_whole_sensor_to_nt_nx_ny(sensor_output, grid)
    n_movie_frames = int(
        propagation_map.shape[0] * MOVIE_DURATION_IN_DEPTH_TRANSITS / DURATION_IN_DEPTH_TRANSITS,
    )
    propagation_map = propagation_map[:n_movie_frames]
    p_max = np.abs(propagation_map).max().item() * MOVIE_COLOR_LIMIT_FRACTION

    plot_utils.plot_array(
        propagation_map[propagation_map.shape[0] // 2],
        vmin=-p_max,
        vmax=p_max,
        export_path=work_dir / "wave_propagation_snapshot.png",
        **plot_utils.mm_axes_kwargs(grid),
    )
    plot_utils.plot_wave_propagation_with_map(
        propagation_map=propagation_map,
        c_map=medium.sound_speed,
        rho_map=medium.density,
        export_name=work_dir / "wave_propagation.mp4",
        vmin=-p_max,
        vmax=p_max,
        figsize=(6, 6),
        circles=circles_mm,
        points=points_mm,
        mark_color="black",
        cmap="seismic",
        show_medium=False,
        **plot_utils.mm_axes_kwargs(grid),
    )


def write_b_mode(
    *,
    grid: fullwave.Grid,
    medium: fullwave.Medium,
    transducer: fullwave.Transducer,
    work_dir: Path,
) -> None:
    """Send the plane wave, beamform the echoes and write the B-mode.

    Parameters
    ----------
    grid : fullwave.Grid
        Computational grid.
    medium : fullwave.Medium
        Medium of the simulation.
    transducer : fullwave.Transducer
        Transducer that sends the plane wave and receives the echoes.
    work_dir : Path
        Directory of the B-mode image.

    """
    transducer.plane_wave(angle_deg=PLANE_WAVE_ANGLE_DEG, apodization=0.5)
    solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        transducer=transducer,
        run_on_memory=True,
    )
    sensor_output = transducer.post_process_sensor_output(
        solver.run(),
        average_surface_signals=True,
    )

    transducer_width_m = transducer.transducer_geometry.transducer_width_m
    lateral_position_m = np.linspace(-transducer_width_m / 2, transducer_width_m / 2, num=256)
    axial_position_m = np.linspace(
        5e-3,
        grid.domain_size[0] - transducer.face_depth_m,
        num=512,
    )
    beamformer = MachBeamformer(
        lateral_position_m=lateral_position_m,
        axial_position_m=axial_position_m,
        receiver_positions_m=receiver_positions_m(transducer),
        sound_speed=grid.c0,
        center_frequency_hz=grid.f0,
        sampling_frequency_hz=1 / (grid.dt * transducer.sampling_modulus_time),
        f_number=1.0,
        time_zero_s=face_crossing_time_s(transducer),
        tukey_alpha=0.5,
    )
    tx_arrival_s = beamformer.plane_wave_arrival_s(angle_deg=PLANE_WAVE_ANGLE_DEG)
    b_mode_db = convert_to_db(beamformer.beamform(sensor_output, tx_arrival_s=tx_arrival_s))
    plot_utils.plot_array(
        b_mode_db,
        vmin=-40,
        vmax=0,
        aspect=1,
        cmap="gray",
        extent=[
            lateral_position_m[0] * 1e3,
            lateral_position_m[-1] * 1e3,
            axial_position_m[-1] * 1e3,
            axial_position_m[0] * 1e3,
        ],
        xlabel="Lateral position (mm)",
        ylabel="Depth below the probe face (mm)",
        colorbar=True,
        export_path=work_dir / "b_mode.png",
    )


def main(*, f0_hz: float = F0_HZ, work_dir: Path = Path("./outputs/plane_wave_phantom")) -> None:
    """Run the plane wave movie and the B-mode of the combined phantom.

    Parameters
    ----------
    f0_hz : float, optional
        Center frequency of the transmit [Hz].
    work_dir : Path, optional
        Directory of the images and the movie.

    """
    logging.getLogger("__main__").setLevel(logging.INFO)
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- define the computational grid and the transducer ---
    #

    c0 = 1540
    duration = DOMAIN_SIZE_M[0] / c0 * DURATION_IN_DEPTH_TRANSITS
    grid = fullwave.Grid(DOMAIN_SIZE_M, f0_hz, duration, c0=c0, ppw=12, cfl=0.4)
    transducer = fullwave.Transducer.l7_4(grid, sampling_modulus_time=7)

    #
    # --- define the phantom, which starts at the probe face ---
    #

    liver = MaterialProperties.liver
    phantom = fullwave.presets.CombinedPhantomDomain(
        grid=grid,
        use_center=True,
        use_px_based_point_target=False,
        shift_px=[transducer.face_row_px, 0],
        base_sound_speed=liver["sound_speed"],
        base_density=liver["density"],
        base_alpha_coeff=liver["alpha_coeff"],
        base_alpha_power=liver["alpha_power"],
        base_scatterer_ratio=0.02,
        rng=np.random.default_rng(seed=42),
    )
    plot_utils.plot_array(
        phantom.phantom_map,
        export_path=work_dir / "phantom_map.png",
        **plot_utils.mm_axes_kwargs(grid),
    )
    medium = fullwave.Medium(
        grid,
        beta=phantom.beta,
        air_map=phantom.air,
        **transducer.fill_probe_above_face(
            sound_speed=phantom.sound_speed,
            density=phantom.density,
            alpha_coeff=phantom.alpha_coeff,
            alpha_power=phantom.alpha_power,
        ),
    )

    #
    # --- run the simulations ---
    #

    write_movie(
        grid=grid,
        medium=medium,
        transducer=transducer,
        circles_mm=contrast_target_circles_mm(phantom),
        points_mm=point_targets_mm(phantom),
        work_dir=work_dir,
    )
    write_b_mode(grid=grid, medium=medium, transducer=transducer, work_dir=work_dir)


if __name__ == "__main__":
    main()
