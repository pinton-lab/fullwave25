"""Measure the attenuation and the phase velocity of a uniform phantom.

The solver marches one uniform medium and the record is read at two depths. The
ratio of the two amplitudes at the center frequency gives the attenuation, and the
delay between them gives the phase velocity. Both are compared with the power law
the medium was asked for, and with the phase velocity that power law implies
through the Kramers-Kronig relation.

It sweeps a grid of attenuation coefficients and exponents over a band of
frequencies. It builds the grid, the medium, the source, the sensor and the solver
in view, and it needs nothing outside this package.

The domain is stated in wavelengths, so every frequency sees the same number of
cycles. A weakly attenuating cell is given a longer path, because a short path
develops no measurable loss.

Run it with:
    uv run python examples/attenuation_dispersion/attenuation_dispersion.py
"""

import json
import logging
import math
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import gausspulse

import fullwave
from fullwave.utils.coordinates import map_to_coords
from fullwave.utils.relaxation_parameters import RelaxationParametersGenerator


class Case:
    """Every number this example uses, with its unit."""

    # --- what to measure ---
    attenuation_coefficients = (0.0022, 0.5, 1.0)  # dB/(MHz^y cm)
    attenuation_exponents = (0.5, 1.0001, 1.5, 1.999)  # -
    low_hertz = 0.1e6  # Hz
    high_hertz = 20.0e6  # Hz
    frequency_count = 8  # -

    # --- the medium ---
    sound_speed = 1540.0  # m/s
    density = 1000.0  # kg/m^3
    beta = 0.0  # -
    relaxation_mechanisms = 4  # -
    reference_hertz = 5.0e6  # Hz, where the sound speed is quoted

    # --- the numerics ---
    points_per_wavelength = 16  # -
    courant_number = 0.2  # -
    axial_wavelengths = 10.0  # -, the short domain
    lateral_wavelengths = 50.0  # -, the short domain
    deep_axial_wavelengths = 40.0  # -, a lossless cell takes this
    deep_lateral_wavelengths = 200.0  # -
    target_loss_decibels = 0.2  # dB, what the path is sized to develop
    largest_axial_wavelengths = 40.0  # -
    shortest_lengthened_wavelengths = 12.5  # -
    record_ratio = 1.5  # -, the record over the one way transit
    pml_transition_wavelengths = 2.0  # -
    pml_wavelengths = 4.0  # -
    spatial_order = 8  # -

    # --- the pulse and the two planes ---
    source_pressure = 1.0e5  # Pa, the peak of the pulse
    source_bandwidth = 2.0  # -, the fractional bandwidth of the pulse, at the 6 dB points
    near_fraction = 0.05  # -, of the axial extent
    far_fraction = 0.80  # -, of the axial extent
    window_periods = 8.0  # -, the half width of the arrival window

    # --- how a figure is drawn ---
    # A dispersion panel of a weakly attenuating cell spans a fraction of a meter
    # per second. Left to autoscale it fills the panel and reads as a large error.
    smallest_phase_velocity_span = 2.0  # m/s

    # --- constants ---
    decibels_per_neper = 8.685889638065035  # -
    six_decibel_cycles = 0.881  # -, what `gausspulse` spans at a bandwidth of one
    lead_multiple = 2.0  # -, the Gaussian tails run past the 6 dB points
    shortest_lead_cycles = 1.0  # -

    # --- where the output goes ---
    work_root = Path("./outputs/attenuation_dispersion")


def frequencies() -> list[float]:
    """Return the evaluation frequencies, on a logarithmic grid, lowest first.

    Returns
    -------
    list[float]
        The frequencies, in hertz, inclusive of both ends.
    """
    ratio = (Case.high_hertz / Case.low_hertz) ** (1.0 / (Case.frequency_count - 1))
    held = [Case.low_hertz * ratio**index for index in range(Case.frequency_count)]
    held[0], held[-1] = Case.low_hertz, Case.high_hertz
    return held


def refused_cells() -> dict[tuple[float, float], list[str]]:
    """Return the cells the shipped table refuses, against the reason for each.

    The calibration writes one record beside its table. A refused cell carries no
    fitted parameters, so a run of it measures the calibration and not the solver.

    Returns
    -------
    dict[tuple[float, float], list[str]]
        The coefficient and the exponent of each refused cell, against its reasons.
    """
    generator = RelaxationParametersGenerator(n_relaxation_mechanisms=Case.relaxation_mechanisms)
    return generator.invalid_reasons()


def nepers_per_metre(coefficient: float, exponent: float, hertz: float) -> float:
    """Return the power law attenuation in nepers per meter.

    Parameters
    ----------
    coefficient : float
        The power law coefficient, in dB/(MHz^y cm).
    exponent : float
        The power law exponent.
    hertz : float
        The center frequency of the pulse.

    Returns
    -------
    float
        The attenuation, in nepers per meter.
    """
    decibels_per_metre = coefficient * 100.0 * (hertz / 1e6) ** exponent
    return decibels_per_metre / Case.decibels_per_neper


def kramers_kronig_phase_velocity(coefficient: float, exponent: float, hertz: float) -> float:
    """Return the phase velocity a causal power law implies, in meters per second.

    The relation is `1 / c(w) = 1 / c(w0) + tan(pi y / 2) * (a(w) / w - a(w0) / w0)`.
    Each attenuation is divided by its own angular frequency, which removes the zero
    that cancels the pole of the tangent at an exponent of one.

    Parameters
    ----------
    coefficient : float
        The power law coefficient, in dB/(MHz^y cm).
    exponent : float
        The power law exponent.
    hertz : float
        The center frequency of the pulse.

    Returns
    -------
    float
        The phase velocity, in meters per second.
    """
    if coefficient == 0.0 or hertz <= 0.0:
        return Case.sound_speed
    if exponent % 2 == 1:
        slope = nepers_per_metre(coefficient, 1.0, Case.reference_hertz) / (
            2.0 * math.pi * Case.reference_hertz
        )
        inverse = 1.0 / Case.sound_speed - (2.0 / math.pi) * slope * math.log(
            hertz / Case.reference_hertz
        )
        return 1.0 / inverse if inverse else Case.sound_speed
    tangent = math.tan(exponent * math.pi / 2.0)
    at_frequency = nepers_per_metre(coefficient, exponent, hertz) / (2.0 * math.pi * hertz)
    at_reference = nepers_per_metre(coefficient, exponent, Case.reference_hertz) / (
        2.0 * math.pi * Case.reference_hertz
    )
    inverse = 1.0 / Case.sound_speed + tangent * (at_frequency - at_reference)
    return 1.0 / inverse if inverse else Case.sound_speed


def extents_for(coefficient: float, exponent: float, hertz: float) -> tuple[float, float]:
    """Return the axial and the lateral extent one cell needs, in wavelengths.

    The path is sized to develop the target loss. A lossless cell has no such path,
    so it takes the deep domain. A path shorter than the short domain gains nothing,
    so it takes the short one. The lateral extent keeps the ratio of the short pair,
    because a longer path at a fixed width lets the edge wave reach the center line.

    Parameters
    ----------
    coefficient : float
        The power law coefficient, in dB/(MHz^y cm).
    exponent : float
        The power law exponent.
    hertz : float
        The center frequency of the pulse.

    Returns
    -------
    tuple[float, float]
        The axial and the lateral extent, in wavelengths.
    """
    wavelength = Case.sound_speed / hertz
    each = nepers_per_metre(coefficient, exponent, hertz) * wavelength * Case.decibels_per_neper
    if each <= 0.0:
        return Case.deep_axial_wavelengths, Case.deep_lateral_wavelengths
    wanted = Case.target_loss_decibels / each
    if wanted < Case.shortest_lengthened_wavelengths:
        return Case.axial_wavelengths, Case.lateral_wavelengths
    axial = min(wanted, Case.largest_axial_wavelengths)
    return axial, axial * Case.lateral_wavelengths / Case.axial_wavelengths


def lead_seconds(hertz: float) -> float:
    """Return how long the record runs before the peak of the pulse.

    A record that opens too late cuts the leading half of the pulse away, which keeps
    the peak and removes energy, so the amplitude at the center frequency falls.

    Parameters
    ----------
    hertz : float
        The center frequency of the pulse.

    Returns
    -------
    float
        The lead, in seconds.
    """
    cycles = max(
        Case.shortest_lead_cycles,
        Case.lead_multiple * Case.six_decibel_cycles / Case.source_bandwidth,
    )
    return cycles / hertz


def pulse_trace(samples: int, time_step: float, hertz: float):
    """Return the pulse of the record, scaled so its peak is the wanted pressure.

    Parameters
    ----------
    samples : int
        How many time samples the record holds.
    time_step : float
        The sampling interval, in seconds.
    hertz : float
        The center frequency of the pulse.

    Returns
    -------
    NDArray[np.float64]
        The pulse trace.
    """
    time = np.arange(samples) * time_step - lead_seconds(hertz)
    trace = gausspulse(time, fc=hertz, bw=Case.source_bandwidth)
    spectrum = np.fft.rfft(trace)
    spectrum[0] = 0.0
    trace = np.fft.irfft(spectrum, n=trace.size)
    peak = float(np.abs(trace).max())
    return trace * (Case.source_pressure / peak) if peak else trace


def single_bin_dft(trace, hertz: float, time_step: float) -> complex:
    """Return the complex amplitude of one frequency in one trace.

    The rotation is built at exactly that frequency, so the frequency does not have
    to land on a bin of the record.

    Parameters
    ----------
    trace : NDArray[np.float64]
        The recorded samples, already windowed.
    hertz : float
        The frequency to read at.
    time_step : float
        The sampling interval, in seconds.

    Returns
    -------
    complex
        The complex amplitude at that frequency.
    """
    samples = np.arange(len(trace))
    return complex(np.sum(np.asarray(trace) * np.exp(-2j * math.pi * hertz * samples * time_step)))


def windowed_trace(trace, arrival: int, hertz: float, time_step: float):
    """Return the trace with everything outside the arrival window set to zero.

    Parameters
    ----------
    trace : NDArray[np.float64]
        The recorded samples of one plane.
    arrival : int
        The sample the window opens around.
    hertz : float
        The center frequency of the pulse.
    time_step : float
        The sampling interval, in seconds.

    Returns
    -------
    NDArray[np.float64]
        A trace of the same length, zero outside the window.
    """
    values = np.zeros_like(np.asarray(trace, dtype=float))
    half = round(Case.window_periods / hertz / time_step)
    low, high = max(0, arrival - half), min(values.size, arrival + half)
    values[low:high] = np.asarray(trace, dtype=float)[low:high]
    return values


def delay_between(near, far, hertz: float, time_step: float) -> float:
    """Return the delay in seconds between two planes, at one frequency.

    A path of several wavelengths makes the measured phase ambiguous by whole cycles.
    A cross correlation cannot wrap, so it selects the cycle count, and the phase then
    supplies the precision.

    Parameters
    ----------
    near : NDArray[np.float64]
        The windowed trace at the near plane.
    far : NDArray[np.float64]
        The windowed trace at the far plane.
    hertz : float
        The center frequency of the pulse.
    time_step : float
        The sampling interval, in seconds.

    Returns
    -------
    float
        The delay, in seconds.
    """
    near_phasor = single_bin_dft(near, hertz, time_step)
    far_phasor = single_bin_dft(far, hertz, time_step)
    if near_phasor == 0 or far_phasor == 0:
        return math.nan
    wrapped = np.angle(near_phasor / far_phasor) / (2.0 * math.pi * hertz)
    length = 1 << (2 * max(len(near), len(far)) - 1).bit_length()
    spectrum = np.fft.rfft(far, length) * np.conj(np.fft.rfft(near, length))
    lag = int(np.argmax(np.fft.irfft(spectrum, length)))
    arrival = (lag - length if lag > length // 2 else lag) * time_step
    return float(wrapped + round((arrival - wrapped) * hertz) / hertz)


def measured(coefficient: float, exponent: float, hertz: float, work_dir: Path) -> dict:
    """Run ONE simulation and read the attenuation and the phase velocity.

    Parameters
    ----------
    coefficient : float
        The power law coefficient, in dB/(MHz^y cm).
    exponent : float
        The power law exponent.
    hertz : float
        The center frequency of the pulse.
    work_dir : Path
        Where the solver writes its scratch files.

    Returns
    -------
    dict
        The attenuation in nepers per meter, the phase velocity and the path.
    """
    # --- define the computational grid ---
    axial, lateral = extents_for(coefficient, exponent, hertz)
    wavelength = Case.sound_speed / hertz
    grid = fullwave.Grid(
        domain_size=(axial * wavelength, lateral * wavelength),
        f0=hertz,
        c0=Case.sound_speed,
        ppw=Case.points_per_wavelength,
        cfl=Case.courant_number,
        duration=Case.record_ratio * axial * wavelength / Case.sound_speed,
    )

    # --- define the acoustic medium properties ---
    shape = (grid.nx, grid.ny)
    medium = fullwave.Medium(
        grid=grid,
        sound_speed=Case.sound_speed * np.ones(shape),  # m/s
        density=Case.density * np.ones(shape),  # kg/m^3
        alpha_coeff=coefficient * np.ones(shape),  # dB/(MHz^y cm)
        alpha_power=exponent * np.ones(shape),  # -
        beta=Case.beta * np.ones(shape),  # -
        n_relaxation_mechanisms=Case.relaxation_mechanisms,
        use_isotropic_relaxation=True,
    )

    # --- define the acoustic source ---
    # One row, because a soft source adds its signal to the pressure of its own nodes.
    source_mask = np.zeros(shape, dtype=bool)
    source_mask[0, :] = True
    coords = map_to_coords(source_mask)
    pulse = pulse_trace(grid.nt, grid.dt, hertz)
    source = fullwave.Source(
        p0=np.tile(pulse, (coords.shape[0], 1)),
        coords=coords,
        grid_shape=grid.shape,
    )

    # --- define the sensor ---
    # One column down the center line. Row `i` of the record is the trace at
    # depth `i * grid.dx`.
    sensor_mask = np.zeros(shape, dtype=bool)
    sensor_mask[:, grid.ny // 2] = True
    sensor = fullwave.Sensor(mask=sensor_mask)

    # --- run simulation ---
    fw_solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        use_pml=True,
        n_transition_layer=round(Case.pml_transition_wavelengths * Case.points_per_wavelength),
        pml_layer_thickness_px=round(Case.pml_wavelengths * Case.points_per_wavelength),
        source_type="soft",
        m_spatial_order=Case.spatial_order,
        run_on_memory=True,
    )
    recorded = np.asarray(fw_solver.run())

    # --- read the two planes ---
    near_row = round(grid.nx * Case.near_fraction)
    far_row = round(grid.nx * Case.far_fraction)
    path_metres = (far_row - near_row) * grid.dx
    if not np.isfinite(recorded).all():
        return {"attenuation": math.nan, "phase_velocity": math.nan, "path_metres": path_metres}

    lead = lead_seconds(hertz)
    near = windowed_trace(
        recorded[near_row],
        round((lead + near_row * grid.dx / Case.sound_speed) / grid.dt),
        hertz,
        grid.dt,
    )
    far = windowed_trace(
        recorded[far_row],
        round((lead + far_row * grid.dx / Case.sound_speed) / grid.dt),
        hertz,
        grid.dt,
    )
    near_amplitude = abs(single_bin_dft(near, hertz, grid.dt))
    far_amplitude = abs(single_bin_dft(far, hertz, grid.dt))
    delay = delay_between(near, far, hertz, grid.dt)
    return {
        "attenuation": math.log(near_amplitude / far_amplitude) / path_metres,
        "phase_velocity": path_metres / delay if delay else math.nan,
        "path_metres": path_metres,
    }


def draw(rows: list[dict], quantity: str, export_path: Path) -> None:
    """Draw one measured quantity of every cell against frequency, beside its reference.

    Parameters
    ----------
    rows : list[dict]
        Every row of the sweep.
    quantity : str
        The column to draw, either `attenuation` or `phase_velocity`.
    export_path : Path
        Where the figure is written.

    Returns
    -------
    None
        None.
    """
    # A panel title rounds the exponent to one decimal, so the table's own 1.0001
    # and 1.999 read as 1.0 and 2.0. Every number behind the panel is the exact one.
    labels = {
        "attenuation": "attenuation [Np/m]",
        "phase_velocity": "phase velocity [m/s]",
    }
    references = {
        "attenuation": nepers_per_metre,
        "phase_velocity": kramers_kronig_phase_velocity,
    }
    coefficients = Case.attenuation_coefficients
    exponents = Case.attenuation_exponents
    figure, panels = plt.subplots(
        len(coefficients),
        len(exponents),
        figsize=(4.0 * len(exponents), 3.2 * len(coefficients)),
        squeeze=False,
    )
    dense = np.logspace(math.log10(Case.low_hertz), math.log10(Case.high_hertz), 200)
    for row, coefficient in enumerate(coefficients):
        for column, exponent in enumerate(exponents):
            axes = panels[row][column]
            found = sorted(
                (
                    one
                    for one in rows
                    if one["alpha_coeff"] == coefficient and one["alpha_power"] == exponent
                ),
                key=lambda one: one["hertz"],
            )
            axes.plot(
                dense / 1e6,
                [references[quantity](coefficient, exponent, one) for one in dense],
                "-",
                color="#d62728",
                label="power law",
            )
            if found:
                axes.plot(
                    [one["hertz"] / 1e6 for one in found],
                    [one[quantity] for one in found],
                    "o--",
                    color="#000000",
                    label="fullwave",
                )
            else:
                axes.text(0.5, 0.5, "not measured", ha="center", transform=axes.transAxes)
            axes.set_title(f"alpha0 {coefficient}, y {exponent:.1f}")
            axes.set_xlabel("frequency [MHz]")
            axes.set_ylabel(labels[quantity])
            if quantity == "phase_velocity":
                low, high = axes.get_ylim()
                if high - low < Case.smallest_phase_velocity_span:
                    middle = 0.5 * (low + high)
                    half = 0.5 * Case.smallest_phase_velocity_span
                    axes.set_ylim(middle - half, middle + half)
            axes.grid(visible=True, alpha=0.3)
            axes.legend(fontsize=7)
    figure.tight_layout()
    figure.savefig(export_path, dpi=150)
    plt.close(figure)


def main() -> None:
    """Sweep every cell over the band, write each case, then draw and report."""
    logging.getLogger("__main__").setLevel(logging.INFO)
    Case.work_root.mkdir(parents=True, exist_ok=True)
    records = Case.work_root / "cases"
    records.mkdir(parents=True, exist_ok=True)

    refused = refused_cells()
    rows = []
    for coefficient in Case.attenuation_coefficients:
        for exponent in Case.attenuation_exponents:
            if (coefficient, exponent) in refused:
                why = ", ".join(refused[(coefficient, exponent)])
                print(
                    f"alpha0 {coefficient:g}, y {exponent:g} is refused by the table: {why}",
                    flush=True,
                )
                continue
            for hertz in frequencies():
                stem = f"alpha{coefficient:g}_y{exponent:g}_{hertz / 1e6:.4f}MHz"
                path = records / f"{stem}.json"
                if path.exists():
                    rows.append(json.loads(path.read_text(encoding="utf-8")))
                    continue
                reading = measured(coefficient, exponent, hertz, Case.work_root / "work" / stem)
                reading.update(
                    {
                        "alpha_coeff": coefficient,
                        "alpha_power": exponent,
                        "hertz": hertz,
                        "reference_attenuation": nepers_per_metre(coefficient, exponent, hertz),
                        "reference_phase_velocity": kramers_kronig_phase_velocity(
                            coefficient, exponent, hertz
                        ),
                    }
                )
                path.write_text(json.dumps(reading, indent=1), encoding="utf-8")
                rows.append(reading)
                print(
                    f"  {stem:<34}{reading['attenuation']:>14.6f} Np/m against "
                    f"{reading['reference_attenuation']:>12.6f} Np/m",
                    flush=True,
                )

    header = (
        f"{'alpha0 [dB/(MHz^y cm)]':>24}{'y [-]':>9}{'freq [MHz]':>13}"
        f"{'fullwave [Np/m]':>18}{'power law [Np/m]':>19}{'fullwave over law [-]':>24}"
        f"{'speed error [m/s]':>20}"
    )
    print(header)
    print("-" * len(header))
    for one in rows:
        ratio = (
            one["attenuation"] / one["reference_attenuation"]
            if one["reference_attenuation"]
            else math.nan
        )
        print(
            f"{one['alpha_coeff']:>24}{one['alpha_power']:>9}{one['hertz'] / 1e6:>13.4f}"
            f"{one['attenuation']:>18.6f}{one['reference_attenuation']:>19.6f}"
            f"{ratio:>24.4f}"
            f"{one['phase_velocity'] - one['reference_phase_velocity']:>20.4f}",
        )

    print()
    for quantity, name in (("attenuation", "attenuation"), ("phase_velocity", "dispersion")):
        export_path = Case.work_root / f"{name}.png"
        draw(rows, quantity, export_path)
        print(f"the {name} figure is {export_path}")


if __name__ == "__main__":
    main()
