"""The time axis that the fullwave2-ultra writers derive from the grid parameters."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

    import fullwave


def _round_half_away_from_zero(x: float) -> float:
    return float(np.sign(x) * np.floor(np.abs(x) + 0.5))


def time_axis(
    *,
    c0: float,
    omega0: float,
    dur: float,
    ppw: float,
    cfl: float,
) -> tuple[float, float, int]:
    """Return the cell edge [m], the time step [s] and the number of steps."""
    wavelength = c0 / omega0 * 2 * np.pi
    dx = wavelength / ppw
    dt = dx / c0 * cfl
    n_t = int(_round_half_away_from_zero(dur * c0 / wavelength * ppw / cfl))
    return dx, dt, n_t


def delay_additive_signal(signal: NDArray) -> NDArray:
    """Return the additive signal delayed by one step, as ``(ncoords_add, n_tic + 1)``.

    The binary adds sample ``n`` to the field before the update of step ``n``. The native solver
    adds it after that update, so one leading zero puts each sample on the same step.
    """
    return np.pad(signal, ((0, 0), (1, 0)))


def check_time_axis(grid: fullwave.Grid) -> int:
    """Return the number of steps after checking that the writer's time axis is the grid's.

    Raises
    ------
    ValueError
        When the number of steps or the time step differs.
    """
    _, dt, n_t = time_axis(
        c0=grid.c0, omega0=grid.omega, dur=grid.duration, ppw=grid.ppw, cfl=grid.cfl
    )
    if n_t != grid.nt or not np.isclose(dt, grid.dt, rtol=1e-9):
        msg = (
            f"the writer's time axis (nT {n_t}, dT {dt}) differs from the grid's "
            f"(nt {grid.nt}, dt {grid.dt}); change the duration by a fraction of a step"
        )
        raise ValueError(msg)
    return n_t
