"""Demonstrate the built-in high-pass filter inside solver.run().

Based on examples/linear_transducer/plane_wave_compounding.py.

This script runs a single 0-degree plane wave transmission through a medium
with echoic targets, then shows the effect of passing
``highpass_cutoff_mhz=0.5`` to ``solver.run()``:

  * Left panel   - raw sensor traces (PML low-frequency drift visible)
  * Right panel  - high-pass filtered traces (drift removed)
  * Bottom panel - amplitude spectra of a single element before/after

Run with:
    uv run python examples/signal_filter/plane_wave_with_filter.py
"""

import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import fullwave
from fullwave.utils.signal_filter import apply_filter

logging.getLogger("__main__").setLevel(logging.INFO)


# ---------------------------------------------------------------------------
# Medium helpers
# ---------------------------------------------------------------------------


def _make_echoic_targets(
    scatterer: np.ndarray,
    grid: fullwave.Grid,
    target_radius_m: float,
    target_spacing_m: float,
    n_targets_axial: int,
    n_targets_lateral: int,
) -> np.ndarray:
    """Place a grid of hypo/hyper/anechoic circles in the scatterer map."""
    total_axial = (n_targets_axial - 1) * target_spacing_m
    total_lateral = (n_targets_lateral - 1) * target_spacing_m
    x0 = (grid.shape[0] * grid.dx - total_axial) / 2
    y0 = (grid.shape[1] * grid.dx - total_lateral) / 2

    axial_pos = np.linspace(x0, x0 + total_axial, n_targets_axial)
    lateral_pos = np.linspace(y0, y0 + total_lateral, n_targets_lateral)

    # columns: anechoic, hypo-echoic, hyper-echoic
    ratio = np.array(
        [
            [0.0, 0.5, 3.0],
            [0.0, 0.5, 3.0],
            [0.0, 0.5, 3.0],
        ],
    )

    for i_ax, xp in enumerate(axial_pos):
        for i_lat, yp in enumerate(lateral_pos):
            xi = int(xp / grid.dx)
            yi = int(yp / grid.dx)
            rr, cc = np.ogrid[
                -xi : scatterer.shape[0] - xi,
                -yi : scatterer.shape[1] - yi,
            ]
            mask = rr**2 + cc**2 <= (target_radius_m / grid.dx) ** 2
            scatterer -= 1.0
            scatterer[mask] *= ratio[i_ax, i_lat]
            scatterer += 1.0

    return scatterer


def _make_input_signal(
    grid: fullwave.Grid,
    transducer: fullwave.Transducer,
    element_layer_px: int,
    p_max: float = 1e5,
) -> np.ndarray:
    """Build a 0-degree plane wave input signal (no steering delay)."""
    input_signal = np.zeros((transducer.n_sources, grid.nt))
    for i in range(len(input_signal)):
        n_y = input_signal.shape[0] // element_layer_px
        i_layer = i // n_y
        input_signal[i] = fullwave.utils.pulse.gaussian_modulated_sinusoidal_signal(
            nt=grid.nt,
            f0=grid.f0,
            duration=grid.duration,
            ncycles=2,
            drop_off=2,
            p0=p_max,
            i_layer=i_layer,
            dt_for_layer_delay=grid.dt,
            cfl_for_layer_delay=grid.cfl,
            delay_sec=0.0,
        )
    return input_signal


# ---------------------------------------------------------------------------
# Simulation sub-steps
# ---------------------------------------------------------------------------


def _build_medium(
    grid: fullwave.Grid,
    c0: float,
) -> fullwave.Medium:
    """Build the acoustic medium with echoic scattering targets."""
    rng = np.random.default_rng(42)
    scatterer, _ = fullwave.utils.generate_scatterer(
        grid=grid,
        ratio_scatterer_to_total_grid=0.38,
        scatter_value_std=0.02 / 2,
        rng=rng,
    )
    scatterer = _make_echoic_targets(
        scatterer,
        grid,
        target_radius_m=5e-3,
        target_spacing_m=15e-3,
        n_targets_axial=3,
        n_targets_lateral=3,
    )
    return fullwave.Medium(
        grid,
        sound_speed=np.ones(grid.shape) * c0,
        density=np.ones(grid.shape) * 1000 * scatterer,
        alpha_coeff=np.ones(grid.shape) * 0.5,
        alpha_power=np.ones(grid.shape) * 1.1,
        beta=np.zeros(grid.shape),
        air_map=np.zeros(grid.shape),
    )


def _build_transducer(
    grid: fullwave.Grid,
    domain_size: tuple[float, float],
) -> tuple[fullwave.Transducer, int]:
    """Build the 128-element linear transducer and return it with element_layer_px."""
    element_layer_px = 4
    transducer_width_m = 38e-3
    transducer_geometry = fullwave.TransducerGeometry(
        grid,
        number_elements=128,
        element_width_m=0.298e-3 - 0.048e-3,
        element_spacing_m=0.048e-3,
        element_layer_px=element_layer_px,
        position_m=(0, (domain_size[1] - transducer_width_m) / 2),
        radius=float("inf"),
    )
    transducer = fullwave.Transducer(
        transducer_geometry=transducer_geometry,
        grid=grid,
        sampling_modulus_time=7,
    )
    transducer.set_signal(_make_input_signal(grid, transducer, element_layer_px))
    return transducer, element_layer_px


def _run_simulation(
    work_dir: Path,
    grid: fullwave.Grid,
    medium: fullwave.Medium,
    transducer: fullwave.Transducer,
) -> tuple[np.ndarray, np.ndarray]:
    """Run the solver and return (raw_output, hp_filtered_output)."""
    solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        transducer=transducer,
    )
    raw_output = solver.run(simulation_dir_name="txrx_raw", is_static_map=True)
    raw_output = transducer.post_process_sensor_output(raw_output, average_surface_signals=True)
    shutil.rmtree(work_dir / "txrx_raw")

    # Equivalent to passing highpass_cutoff_mhz=0.5 directly to solver.run()
    dt_rec = grid.dt * transducer.sampling_modulus_time
    filtered_output = apply_filter(raw_output, dt=dt_rec, f_low_hz=0.5e6, use_gpu=False)
    return raw_output, filtered_output


def _plot_results(
    work_dir: Path,
    grid: fullwave.Grid,
    transducer: fullwave.Transducer,
    raw_output: np.ndarray,
    filtered_output: np.ndarray,
    f0: float,
) -> None:
    """Plot time traces and amplitude spectra for three representative elements."""
    dt_rec = grid.dt * transducer.sampling_modulus_time
    n_t_rec = raw_output.shape[1]
    t_us = np.arange(n_t_rec) * dt_rec * 1e6
    freqs_mhz = np.fft.rfftfreq(n_t_rec, d=dt_rec) / 1e6

    def _db(sig: np.ndarray) -> np.ndarray:
        amp = np.abs(np.fft.rfft(sig))
        return 20 * np.log10(np.maximum(amp / (amp.max() + 1e-12), 1e-5))

    n_elem = raw_output.shape[0]
    elem_indices = [n_elem // 4, n_elem // 2, 3 * n_elem // 4]

    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    fig.suptitle("Plane wave — sensor output before / after HP filter (0.5 MHz)", fontsize=12)

    for row, idx in enumerate(elem_indices):
        ax_t, ax_f = axes[row, 0], axes[row, 1]
        raw_trace, filt_trace = raw_output[idx], filtered_output[idx]

        ax_t.plot(t_us, raw_trace, color="tab:gray", lw=0.7, alpha=0.7, label="Raw")
        ax_t.plot(t_us, filt_trace, color="tab:orange", lw=0.9, label="HP 0.5 MHz")
        ax_t.set_ylabel("Pressure")
        ax_t.set_title(f"element {idx}")
        ax_t.legend(fontsize=8)
        ax_t.set_xlim(t_us[0], t_us[-1])
        if row == 2:
            ax_t.set_xlabel("Time (µs)")

        ax_f.plot(freqs_mhz, _db(raw_trace), color="tab:gray", lw=0.7, alpha=0.7, label="Raw")
        ax_f.plot(freqs_mhz, _db(filt_trace), color="tab:orange", lw=0.9, label="HP 0.5 MHz")
        ax_f.axvline(f0 / 1e6, color="steelblue", lw=0.8, ls="--", label=f"f0={f0 / 1e6:.0f} MHz")
        ax_f.set_xlim(0, f0 / 1e6 * 3)
        ax_f.set_ylim(-80, 5)
        ax_f.set_ylabel("Amplitude (dB)")
        ax_f.legend(fontsize=8)
        if row == 2:
            ax_f.set_xlabel("Frequency (MHz)")

    axes[0, 0].set_title(f"Time traces — {axes[0, 0].get_title()}", fontsize=10)
    axes[0, 1].set_title(f"Amplitude spectra — {axes[0, 1].get_title()}", fontsize=10)
    plt.tight_layout()
    out_fig = work_dir / "sensor_before_after_filter.png"
    plt.savefig(out_fig, dpi=150)
    print(f"Saved figure to {out_fig}")


def _print_summary(raw_output: np.ndarray, filtered_output: np.ndarray) -> None:
    """Print DC drift statistics before and after filtering."""
    drift_raw = raw_output.mean(axis=1)
    drift_filt = filtered_output.mean(axis=1)
    print(
        f"\nDC drift (mean across elements):"
        f"\n  Raw      - mean={drift_raw.mean():.4f}, std={drift_raw.std():.4f}"
        f"\n  Filtered - mean={drift_filt.mean():.6f}, std={drift_filt.std():.6f}",
    )
    print(
        "\nTip: pass highpass_cutoff_mhz=0.5 directly to solver.run() to apply"
        " the same filter automatically before the result is returned.",
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run plane wave simulation and compare raw vs high-pass filtered sensor output."""
    work_dir = Path("./outputs/plane_wave_with_filter")
    work_dir.mkdir(parents=True, exist_ok=True)

    domain_size = (4.5e-2, 4.5e-2)  # [axial, lateral] m
    f0 = 2e6
    c0 = 1540

    grid = fullwave.Grid(domain_size, f0, domain_size[0] / c0 * 2.3, c0=c0, ppw=12, cfl=0.4)
    medium = _build_medium(grid, c0)
    transducer, _ = _build_transducer(grid, domain_size)
    raw_output, filtered_output = _run_simulation(work_dir, grid, medium, transducer)
    _plot_results(work_dir, grid, transducer, raw_output, filtered_output, f0)
    _print_summary(raw_output, filtered_output)


if __name__ == "__main__":
    main()
