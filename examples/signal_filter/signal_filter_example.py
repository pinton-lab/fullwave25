"""Show what the signal filter does, on a trace made by hand.

The example needs no accelerator, because it runs no simulation. It builds a
trace that carries a slow drift, a 3 MHz pulse and broadband noise. A high pass
filter at 0.5 MHz removes the drift, and a band pass filter from 1 to 5 MHz keeps
the pulse alone. It plots each trace and its amplitude spectrum.

Run it with:
    uv run python examples/signal_filter/signal_filter_example.py
"""

import matplotlib.pyplot as plt
import numpy as np

from fullwave.utils.signal_filter import apply_filter


def main() -> None:
    """Build a synthetic sensor trace and demonstrate high-pass and band-pass filtering."""
    # ---------------------------------------------------------------------------
    # Simulation-like parameters
    # ---------------------------------------------------------------------------
    f0 = 3e6  # center frequency, Hz
    dt = 1 / (f0 * 20)  # ~20 samples per period
    n_t = 2048
    t = np.arange(n_t) * dt  # time axis, seconds

    # ---------------------------------------------------------------------------
    # Synthetic signal: DC drift + 3 MHz pulse + broadband noise
    # ---------------------------------------------------------------------------
    rng = np.random.default_rng(42)

    # Slow PML drift: linear ramp (typical artifact)
    drift = np.linspace(0, 2.0, n_t)

    # Short Gaussian-windowed 3 MHz pulse arriving at t = 3 µs
    t_pulse = 3e-6
    sigma = 0.5e-6
    envelope = np.exp(-0.5 * ((t - t_pulse) / sigma) ** 2)
    pulse = envelope * np.sin(2 * np.pi * f0 * t)

    # White noise floor at -30 dB relative to pulse
    noise = rng.standard_normal(n_t) * 0.03

    raw = drift + pulse + noise  # shape [n_t]

    # Wrap as [n_sensors, n_t] (one channel)
    data = raw[np.newaxis, :]

    # ---------------------------------------------------------------------------
    # Apply filters
    # ---------------------------------------------------------------------------
    hp_filtered = apply_filter(data, dt, f_low_hz=0.5e6, use_gpu=False)
    bp_filtered = apply_filter(data, dt, f_low_hz=1e6, f_high_hz=5e6, use_gpu=False)

    # ---------------------------------------------------------------------------
    # Amplitude spectrum helper
    # ---------------------------------------------------------------------------
    freqs_mhz = np.fft.rfftfreq(n_t, d=dt) / 1e6

    def spectrum_db(sig: np.ndarray) -> np.ndarray:
        """Return amplitude spectrum in dB (normalised to peak, floored at -100 dB)."""
        amp = np.abs(np.fft.rfft(sig))
        amp_norm = amp / (amp.max() + 1e-12)
        return 20 * np.log10(np.maximum(amp_norm, 1e-5))  # floor at -100 dB

    # ---------------------------------------------------------------------------
    # Plot — 2 rows (time / spectrum) x 3 columns (raw / high-pass / band-pass)
    # Each filtered column overlays the raw signal so before/after is clear.
    # ---------------------------------------------------------------------------
    t_us = t * 1e6  # µs for display

    fig, axes = plt.subplots(2, 3, figsize=(14, 7))
    fig.suptitle("Built-in signal filter — example", fontsize=13)

    filters = [
        ("Raw (drift + pulse + noise)", data[0], None, "tab:blue"),
        ("High-pass 0.5 MHz", hp_filtered[0], data[0], "tab:orange"),
        ("Band-pass 1-5 MHz", bp_filtered[0], data[0], "tab:green"),
    ]

    for col, (label, sig, before, color) in enumerate(filters):
        ax_t = axes[0, col]
        ax_f = axes[1, col]

        # --- time trace ---
        if before is not None:
            ax_t.plot(t_us, before, color="tab:gray", lw=0.6, alpha=0.5, label="Before")
        ax_t.plot(t_us, sig, color=color, lw=0.9, label="After" if before is not None else label)
        ax_t.set_title(label, fontsize=10)
        ax_t.set_xlabel("Time (µs)")
        ax_t.set_ylabel("Amplitude")
        ax_t.set_xlim(t_us[0], t_us[-1])
        if before is not None:
            ax_t.legend(fontsize=8)

        # --- amplitude spectrum ---
        if before is not None:
            ax_f.plot(
                freqs_mhz,
                spectrum_db(before),
                color="tab:gray",
                lw=0.6,
                alpha=0.5,
                label="Before",
            )
        ax_f.plot(
            freqs_mhz,
            spectrum_db(sig),
            color=color,
            lw=0.9,
            label="After" if before is not None else label,
        )
        ax_f.set_xlabel("Frequency (MHz)")
        ax_f.set_ylabel("Amplitude (dB)")
        ax_f.set_xlim(0, 10)
        ax_f.set_ylim(-80, 5)
        ax_f.axvline(f0 / 1e6, color="gray", lw=0.8, ls="--", label=f"f₀ = {f0 / 1e6:.0f} MHz")
        ax_f.legend(fontsize=8)

    plt.tight_layout()
    out_path = "signal_filter_example.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
