"""Write the 3D run directory of ``.dat`` files that a fullwave2-ultra 3D binary reads."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from . import dat_io, dat_io_3d
from .medium_to_aexp import aexp_from_amap, extend_map_faces, face_margins
from .time_axis import time_axis

if TYPE_CHECKING:
    from numpy.typing import NDArray

SPONGE_WAVELENGTHS = 4


def frozen_shell_3d(*, m_stencil: int = 8) -> int:
    """Return the frozen shell of the 3D binary: its outer cells that are never updated."""
    return m_stencil


def write_run_directory_3d(
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
    m_stencil: int = 8,
    air_coords: NDArray | None = None,
    additive_incoords: NDArray | None = None,
    n_tic_additive: int | None = None,
    back_faces: frozenset[str] = frozenset(),
) -> dict[str, float | int | dict[str, int]]:
    """Write every input file except the source signals and return the realized scalars.

    A face named in ``back_faces`` is padded by the frozen shell (outermost) and then the
    sponge of ``nbdy`` cells, which ends on the cell next to the grid; all other faces are
    padded by ``mext`` cells.

    With no hard source cells, ``icc.dat`` is not written: the additive source carries the
    whole source.

    Parameters
    ----------
    outdir : str | Path
        The run directory.
    c0, omega0, dur, ppw, cfl : float
        Sound speed, angular frequency, duration, points per wavelength and Courant number.
    maps : dict[str, NDArray]
        Interior volumes ``cmap``, ``rmap``, ``nmap`` and ``amap`` [dB/(MHz cm)].
    incoords, outcoords : NDArray
        0-based interior ``(N, 3)`` source and sensor cells.
    n_tic, mod_t : int
        Source length in steps and sensor decimation in time.
    nbdy : int | None
        Sponge width in cells; None gives ``ppw * SPONGE_WAVELENGTHS``.
    m_stencil : int
        Stencil half-width.
    air_coords : NDArray | None
        0-based interior ``(N, 3)`` cells held at zero pressure on every step.
    additive_incoords : NDArray | None
        0-based interior ``(N, 3)`` cells of the additive source, all different; None writes
        no additive input.
    n_tic_additive : int | None
        Additive signal length in steps; needed with ``additive_incoords``.
    back_faces : frozenset[str]
        Faces ``x_low`` ... ``z_high`` behind a held source on the grid edge. The default empty
        set pads every face by ``mext``.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    nbdy = int(ppw * SPONGE_WAVELENGTHS) if nbdy is None else nbdy
    dx, dt, n_t = time_axis(c0=c0, omega0=omega0, dur=dur, ppw=ppw, cfl=cfl)
    mext = nbdy + m_stencil
    shell = frozen_shell_3d(m_stencil=m_stencil)
    margins = face_margins(ndim=3, mext=mext, shell=shell, nbdy=nbdy, back_faces=back_faces)
    offset = np.array([margins["x_low"], margins["y_low"], margins["z_low"]], dtype=np.int64)

    c = extend_map_faces(maps["cmap"], margins)
    rho = extend_map_faces(maps["rmap"], margins)
    beta = extend_map_faces(maps["nmap"], margins)
    aexp = aexp_from_amap(
        extend_map_faces(maps["amap"], margins),
        c0=c0,
        omega0=omega0,
        dt=dt,
        nbdy=nbdy,
        back_faces=back_faces,
        frozen_cells=shell,
    )
    n_xe, n_ye, n_ze = c.shape

    source_cells = np.asarray(incoords)[:, :3].astype(np.int64) + offset
    sensor_cells = np.asarray(outcoords)[:, :3].astype(np.int64) + offset

    air_cells = np.empty((0, 3), dtype=np.int64)
    if air_coords is not None:
        air_cells = np.asarray(air_coords)[:, :3].astype(np.int64) + offset

    dat_io.write_map(outdir / "c.dat", c)
    dat_io.write_map(outdir / "rho.dat", rho)
    dat_io.write_map(outdir / "K.dat", c**2 * rho)
    dat_io.write_map(outdir / "beta.dat", beta)
    dat_io.write_map(outdir / "Aexp.dat", aexp)
    if len(source_cells):
        dat_io_3d.write_coords_3d(outdir / "icc.dat", *source_cells.T)
    else:
        (outdir / "icc.dat").unlink(missing_ok=True)
    dat_io_3d.write_coords_3d(outdir / "outc.dat", *sensor_cells.T)
    additive_cells = _write_additive_cells(
        outdir, cells=additive_incoords, n_tic=n_tic_additive, offset=offset
    )
    if len(air_cells):
        dat_io_3d.write_coords_3d(outdir / "icczero.dat", *air_cells.T)
        dat_io.write_int(outdir / "ncoordszero.dat", len(air_cells))
        dat_io.write_int(outdir / "nzero0.dat", 0)
        dat_io.write_int(outdir / "nzero1.dat", n_t)
    integers = {
        "nX": n_xe,
        "nY": n_ye,
        "nZ": n_ze,
        "nT": n_t,
        "ncoords": len(source_cells),
        "ncoordsout": len(sensor_cells),
        "nTic": n_tic,
        "modT": mod_t,
        "ncoords_add": len(additive_cells),
    }
    for name, value in integers.items():
        dat_io.write_int(outdir / f"{name}.dat", value)
    for name, value in {"dX": dx, "dY": dx, "dZ": dx, "dT": dt, "c0": c0}.items():
        dat_io.write_float(outdir / f"{name}.dat", value)

    return {
        "nXe": n_xe,
        "nYe": n_ye,
        "nZe": n_ze,
        "nT": n_t,
        "dX": dx,
        "dY": dx,
        "dZ": dx,
        "dT": dt,
        "ncoords": len(source_cells),
        "ncoordsout": len(sensor_cells),
        "ncoordszero": len(air_cells),
        "ncoords_add": len(additive_cells),
        "mext": mext,
        "margins": margins,
    }


def _write_additive_cells(
    outdir: Path, *, cells: NDArray | None, n_tic: int | None, offset: NDArray
) -> NDArray:
    """Write ``icc_add.dat`` and ``nTic_add.dat`` and return the shifted cells (none if disabled).

    Raises
    ------
    ValueError
        When ``n_tic`` is missing or two additive cells are the same cell.
    """
    if cells is None:
        return np.empty((0, 3), dtype=np.int64)
    if n_tic is None:
        msg = "n_tic_additive is required with additive_incoords"
        raise ValueError(msg)
    shifted = np.asarray(cells)[:, :3].astype(np.int64) + offset
    if len(np.unique(shifted, axis=0)) != len(shifted):
        msg = "the additive source cells must all be different: one value is added per cell"
        raise ValueError(msg)
    dat_io_3d.write_coords_3d(outdir / "icc_add.dat", *shifted.T)
    dat_io.write_int(outdir / "nTic_add.dat", n_tic)
    return shifted


def write_sources_3d(
    outdir: str | Path,
    *,
    signal: NDArray | None,
    additive_signal: NDArray | None,
    n_t: int,
    hold: bool = False,
) -> None:
    """Write ``icmat.dat`` and ``icmat_add.dat`` (the latter time-major); skip a missing signal.

    Parameters
    ----------
    outdir : str | Path
        The run directory.
    signal : NDArray | None
        The ``(ncoords, n_tic)`` hard signal; None when there are no hard cells.
    additive_signal : NDArray | None
        The ``(ncoords_add, n_tic_additive)`` signal added to the pressure.
    n_t : int
        Number of time steps.
    hold : bool
        True holds the hard cells at zero pressure from the end of the signal to the end of the
        run, by the window ``nszero0.dat`` = n_tic and ``nszero1.dat`` = n_t. False writes no
        window, which releases the cells when the signal ends.
    """
    outdir = Path(outdir)
    if hold and signal is not None:
        dat_io.write_int(outdir / "nszero0.dat", signal.shape[1])
        dat_io.write_int(outdir / "nszero1.dat", n_t)
    else:
        (outdir / "nszero0.dat").unlink(missing_ok=True)
        (outdir / "nszero1.dat").unlink(missing_ok=True)
    if signal is None:
        (outdir / "icmat.dat").unlink(missing_ok=True)
    else:
        dat_io.write_icmat(outdir / "icmat.dat", [signal])
    if additive_signal is not None:
        dat_io_3d.write_icmat_add_time_major(outdir / "icmat_add.dat", additive_signal)
