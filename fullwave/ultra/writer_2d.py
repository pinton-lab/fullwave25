"""Write the 2D run directory of ``.dat`` files that a fullwave2-ultra 2D batch binary reads."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import dat_io, stencil
from .medium_to_aexp import aexp_from_amap, extend_map
from .time_axis import time_axis

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

SPONGE_WAVELENGTHS = 6


def write_run_directory_2d(
    outdir: str | Path,
    *,
    c0: float,
    omega0: float,
    dur: float,
    ppw: float,
    cfl: float,
    maps: dict[str, NDArray],
    incoords: NDArray,
    outcoords: NDArray,
    n_tic: int,
    mod_t: int,
    nbdy: int | None = None,
    air_coords: NDArray | None = None,
    additive_incoords: NDArray | None = None,
    n_tic_additive: int | None = None,
    m_stencil: int = 8,
) -> dict[str, float | int]:
    """Write every input file except the source signals and return the realized scalars.

    Parameters
    ----------
    outdir : str | Path
        The run directory.
    c0, omega0, dur, ppw, cfl : float
        Sound speed, angular frequency, duration, points per wavelength and Courant number.
    maps : dict[str, NDArray]
        Interior maps ``cmap``, ``rmap``, ``nmap`` and ``amap`` [dB/(MHz cm)].
    incoords, outcoords : NDArray
        0-based interior ``(N, 2)`` source and sensor cells.
    n_tic, mod_t : int
        Source length in steps and sensor decimation in time.
    nbdy : int | None
        Sponge width in cells; None gives ``ppw * SPONGE_WAVELENGTHS``.
    air_coords : NDArray | None
        0-based interior ``(N, 2)`` cells held at zero pressure on every step.
    additive_incoords : NDArray | None
        0-based interior ``(N, 2)`` cells of the additive source; None writes no additive input.
    n_tic_additive : int | None
        Additive signal length in steps; needed with ``additive_incoords``.
    m_stencil : int
        Stencil half-width.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nbdy = int(ppw * SPONGE_WAVELENGTHS) if nbdy is None else nbdy
    dx, dt, n_t = time_axis(c0=c0, omega0=omega0, dur=dur, ppw=ppw, cfl=cfl)
    mext = nbdy + m_stencil

    c = extend_map(maps["cmap"], mext)
    rho = extend_map(maps["rmap"], mext)
    beta = extend_map(maps["nmap"], mext)
    aexp = aexp_from_amap(extend_map(maps["amap"], mext), c0=c0, omega0=omega0, dt=dt, nbdy=nbdy)
    n_xe, n_ye = c.shape
    n_zones = stencil.n_zones(sound_speed=c)

    source_cells = np.asarray(incoords)[:, :2].astype(np.int64) + mext
    sensor_cells = np.asarray(outcoords)[:, :2].astype(np.int64) + mext
    air_cells = np.empty((0, 2), dtype=np.int64)
    if air_coords is not None:
        air_cells = np.asarray(air_coords)[:, :2].astype(np.int64) + mext

    dat_io.write_map(outdir / "c.dat", c)
    dat_io.write_map(outdir / "rho.dat", rho)
    dat_io.write_map(outdir / "K.dat", c**2 * rho)
    dat_io.write_map(outdir / "beta.dat", beta)
    dat_io.write_map(outdir / "Aexp.dat", aexp)
    dat_io.write_dcmap(outdir / "dcmap.dat", stencil.zone_index_map(sound_speed=c), n_zones=n_zones)
    dat_io.write_table(outdir / "dmap.dat", stencil.zone_table(sound_speed=c, dt=dt, dx=dx))
    dat_io.write_table(outdir / "d.dat", stencil.stencil_table(cfl=cfl))
    dat_io.write_coords_2d(outdir / "icc.dat", *source_cells.T)
    dat_io.write_coords_2d(outdir / "outc.dat", *sensor_cells.T)
    if len(air_cells):
        dat_io.write_coords_2d(outdir / "icczero.dat", *air_cells.T)
        _write_air_hold_window(outdir, n_t=n_t)
    dat_io.write_int(outdir / "nTzero.dat", n_t)
    _write_additive_cells(outdir, cells=additive_incoords, n_tic=n_tic_additive, mext=mext)
    integers = {
        "nX": n_xe,
        "nY": n_ye,
        "nT": n_t,
        "ncoords": len(source_cells),
        "ncoordsout": len(sensor_cells),
        "nTic": n_tic,
        "modT": mod_t,
        "ndmap": n_zones,
        "ncoordszero": len(air_cells),
    }
    for name, value in integers.items():
        dat_io.write_int(outdir / f"{name}.dat", value)
    for name, value in {"dX": dx, "dY": dx, "dT": dt, "c0": c0}.items():
        dat_io.write_float(outdir / f"{name}.dat", value)

    return {
        "nXe": n_xe,
        "nYe": n_ye,
        "nT": n_t,
        "dX": dx,
        "dY": dx,
        "dT": dt,
        "ndmap": n_zones,
        "ncoords": len(source_cells),
        "ncoordsout": len(sensor_cells),
        "ncoordszero": len(air_cells),
        "nTzero": n_t,
        "mext": mext,
    }


def _write_air_hold_window(outdir: Path, *, n_t: int) -> None:
    """Hold the air cells for the whole run in binaries that read ``nzero0.dat``/``nzero1.dat``."""
    dat_io.write_int(outdir / "nzero0.dat", 0)
    dat_io.write_int(outdir / "nzero1.dat", n_t)


def _write_additive_cells(
    outdir: Path, *, cells: NDArray | None, n_tic: int | None, mext: int
) -> None:
    """Write ``icc_add.dat``, ``ncoords_add.dat`` and ``nTic_add.dat``, or disable the channel.

    A reused run directory keeps no stale additive cells: ``ncoords_add.dat`` is set to 0.
    """
    count_file = outdir / "ncoords_add.dat"
    if cells is None:
        if count_file.exists():
            dat_io.write_int(count_file, 0)
        return
    if n_tic is None:
        msg = "n_tic_additive is required with additive_incoords"
        raise ValueError(msg)
    shifted = np.asarray(cells)[:, :2].astype(np.int64) + mext
    dat_io.write_coords_2d(outdir / "icc_add.dat", *shifted.T)
    dat_io.write_int(count_file, len(shifted))
    dat_io.write_int(outdir / "nTic_add.dat", n_tic)


def write_sources_2d(
    outdir: str | Path,
    *,
    signals: Sequence[NDArray],
    n_t: int,
    additive_signals: Sequence[NDArray] | None = None,
    hold: bool = False,
) -> None:
    """Write the per-member source signals.

    The source cells run free after their signal ends, or stay at zero pressure when ``hold``.

    Parameters
    ----------
    outdir : str | Path
        The run directory.
    signals : Sequence[NDArray]
        One ``(ncoords, n_tic)`` signal per batch member.
    additive_signals : Sequence[NDArray] | None
        One ``(ncoords_add, n_tic_additive)`` signal per batch member, added to the pressure.
    n_t : int
        Number of time steps.
    hold : bool
        True holds the source cells at zero from the end of the signal to the end of the run,
        by the window ``nszero0.dat`` = n_tic and ``nszero1.dat`` = n_t.
    """
    outdir = Path(outdir)
    dat_io.write_icmat(outdir / "icmat.dat", list(signals))
    if additive_signals is not None:
        dat_io.write_icmat(outdir / "icmat_add.dat", list(additive_signals))
    dat_io.write_int(outdir / "nsims.dat", len(signals))
    dat_io.write_int(outdir / "soft_source.dat", 0)
    if hold:
        dat_io.write_int(outdir / "nszero0.dat", signals[0].shape[1])
        dat_io.write_int(outdir / "nszero1.dat", n_t)
    else:
        dat_io.write_int(outdir / "nszero0.dat", n_t)
