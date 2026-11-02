"""Run a transmit sequence as batches with Solver.run_batch.

A batch runs several transmits on the same medium in one simulation, which is faster
than one `Solver.run` per transmit. The transmits of a batch share the medium, the sensor
and the source positions, and differ only in their signals. The transmits of one
transducer (plane waves, focused lines, synthetic aperture elements) satisfy this.

This example:

1. builds a small speckle medium and an L7-4 transducer,
2. runs 5 plane waves as one batch,
3. runs 16 focused lines, with at most 8 transmits in one simulation,
4. checks that a transmit of the batch equals the same transmit run alone,
5. plots the element traces of one plane wave and one focused line in dB, where the
   echoes of 3 point targets show as hyperbolas.

run_batch downloads the batch solver binary on first use, like the other binaries.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fullwave


def main() -> None:
    """Run the transmit batch example."""
    logging.getLogger("__main__").setLevel(logging.INFO)

    work_dir = Path("./outputs/") / "transmit_batch_basics"
    work_dir.mkdir(parents=True, exist_ok=True)

    #
    # --- define the computational grid ---
    #
    # 20 mm deep and 45 mm wide, wider than the L7-4 aperture.
    f0 = 5.208e6
    c0 = 1540
    width_m = 45e-3  # lateral width [m], wider than the array
    domain_size = (
        2.0e-2,
        width_m,
    )  # [axial, lateral] meters
    duration = domain_size[0] / c0 * 2.3  # the two-way time, with a margin
    grid = fullwave.Grid(domain_size, f0, duration, c0=c0, ppw=8, cfl=0.4)

    #
    # --- define the transducer ---
    #
    receive_every = 4  # time steps between two recorded samples
    transducer = fullwave.Transducer.l7_4(grid, sampling_modulus_time=receive_every)

    #
    # --- define the acoustic medium: soft tissue with speckle and 3 point targets ---
    #
    rng = np.random.default_rng(seed=42)
    scatterer, _ = fullwave.utils.generate_scatterer(
        grid=grid,
        ratio_scatterer_to_total_grid=0.38,
        scatter_value_std=0.01,
        rng=rng,
    )
    density = np.ones(grid.shape) * 1000 * scatterer
    axial_m, lateral_m = np.meshgrid(
        np.arange(grid.shape[0]) * grid.dx,
        np.arange(grid.shape[1]) * grid.dx - domain_size[1] / 2,
        indexing="ij",
    )
    for depth_m in (8e-3, 12e-3, 16e-3):  # below the probe face
        target = (axial_m - transducer.face_depth_m - depth_m) ** 2 + lateral_m**2 <= (0.2e-3) ** 2
        density[target] = 2000  # a strong reflector of 0.2 mm radius

    medium = fullwave.Medium(
        grid,
        beta=np.zeros(grid.shape),
        **transducer.fill_probe_above_face(
            sound_speed=np.ones(grid.shape) * c0,
            density=density,
            alpha_coeff=np.ones(grid.shape) * 0.5,  # dB/(MHz^y cm)
            alpha_power=np.ones(grid.shape) * 1.1,
        ),
    )

    #
    # --- build one source per transmit ---
    #
    # Each transmit call (plane_wave, focus, synthetic_aperture ...) only changes the
    # signals of the same source points, and transducer.source returns a new Source
    # for the transmit set last.
    angles = [-10, -5, 0, 5, 10]  # degrees
    plane_waves = []
    for angle in angles:
        transducer.plane_wave(angle_deg=angle)
        plane_waves.append(transducer.source)

    aperture_m = transducer.transducer_geometry.transducer_width_m
    focus_depth_m = transducer.face_depth_m + 15e-3  # 15 mm below the probe face
    # focus_m is absolute, so the lines are centered on the middle of the grid
    offsets_m = np.linspace(-aperture_m / 4, aperture_m / 4, num=16)
    laterals_m = domain_size[1] / 2 + offsets_m
    focused_lines = []
    for lateral_m in laterals_m:
        transducer.focus(focus_m=(focus_depth_m, lateral_m))
        focused_lines.append(transducer.source)

    #
    # --- the solver: the same as for one transmit ---
    #
    # The Solver's own source (here the transducer's last transmit) sets the source
    # positions; run_batch takes the signals of each transmit from its list.
    solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        transducer=transducer,
    )

    # All 5 plane waves in one simulation. The result is [transmits, sensors, time].
    plane_output = solver.run_batch(plane_waves, simulation_dir_name="plane_waves")
    print(f"plane waves: {plane_output.shape} [transmits, sensor points, time samples]")

    # run_batch runs all 16 focused lines, with at most 8 transmits in one simulation
    # (here 2 simulations of 8). Without max_members_per_run, it puts at most 32 transmits
    # in one simulation, or fewer when the GPU memory estimate does not fit them, so 128
    # transmits run as 4 simulations of 32.
    focus_output = solver.run_batch(
        focused_lines,
        simulation_dir_name="focused_lines",
        max_members_per_run=8,
    )
    print(f"focused lines: {focus_output.shape}")

    #
    # --- a transmit of the batch equals the same transmit run alone ---
    #
    transducer.plane_wave(angle_deg=angles[2])
    single = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        transducer=transducer,
    ).run(simulation_dir_name="plane_wave_alone")
    identical = np.array_equal(single, plane_output[2])
    print(f"plane wave {angles[2]} deg: batch output equals the single run: {identical}")

    #
    # --- plot the element traces of one plane wave and one focused line, in dB ---
    #
    plane_traces = transducer.post_process_sensor_output(
        plane_output[2],
        average_surface_signals=True,
    )
    focus_traces = transducer.post_process_sensor_output(
        focus_output[8],
        average_surface_signals=True,
    )
    time_us = np.arange(plane_traces.shape[1]) * grid.dt * receive_every * 1e6
    figure, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True, layout="constrained")
    for axis, traces, title in (
        (axes[0], plane_traces, f"plane wave {angles[2]} deg"),
        (axes[1], focus_traces, f"focused line {offsets_m[8] * 1e3:.1f} mm off axis"),
    ):
        level_db = 20 * np.log10(np.abs(traces) / np.abs(traces).max() + 1e-12)
        image = axis.imshow(
            level_db.T,
            aspect="auto",
            cmap="gray",
            vmin=-80,
            vmax=0,
            extent=[0, traces.shape[0], time_us[-1], time_us[0]],
        )
        axis.set_title(title)
        axis.set_xlabel("element [-]")
    axes[0].set_ylabel("time [us]")
    figure.colorbar(image, ax=axes, label="level [dB]")
    figure.savefig(work_dir / "element_traces.png", dpi=150)
    print(f"wrote {work_dir / 'element_traces.png'}")


if __name__ == "__main__":
    main()
