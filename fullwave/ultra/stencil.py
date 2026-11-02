"""Optimized staggered finite-difference stencil tables of the 2D binary (Tan and Huang 2014).

The coefficients are polynomials in the local Courant ratio ``r = c * dt / dx``.
The math is float64; the writer casts to float32.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray

# Polynomial in r per (row, column) slot of the stencil table, highest power first.
_COEFFS: dict[tuple[int, int], list[float]] = {
    # d(2,1)
    (1, 0): [
        -8.74634088067635e-4,
        -1.80530560296097e-3,
        -4.40512972481673e-4,
        4.74018847663366e-3,
        -1.93097802254349e-5,
        -2.92328221171893e-1,
        -6.58101498708345e-8,
        1.25420636437969,
    ],
    # d(3,1)
    (2, 0): [
        7.93317828964018e-4,
        1.61433256585486e-3,
        3.97244786277123e-4,
        5.46057645976549e-3,
        1.73781972873916e-5,
        5.88754971188371e-2,
        5.91706982879834e-8,
        -1.23406473759703e-1,
    ],
    # d(4,1)
    (3, 0): [
        -6.50217700538851e-4,
        -1.16449260340413e-3,
        -3.24403734066325e-4,
        -9.11483710059994e-3,
        -1.41739982312600e-5,
        2.33184077551615e-2,
        -4.82326094707544e-8,
        3.46342451534453e-2,
    ],
    # d(5,1)
    (4, 0): [
        4.67529510541428e-4,
        7.32736676632388e-4,
        2.32444388955328e-4,
        8.46419766685254e-3,
        1.01438593426278e-5,
        -3.17586249260511e-2,
        3.44988852042879e-8,
        -1.19674942518101e-2,
    ],
    # d(6,1)
    (5, 0): [
        -2.98416281187033e-4,
        -3.99380750669364e-4,
        -1.48203388388213e-4,
        -6.01788793192501e-3,
        -6.46543538517443e-6,
        2.41912754935119e-2,
        -2.19855171569984e-8,
        4.15554391204146e-3,
    ],
    # d(7,1)
    (6, 0): [
        1.67882669698981e-4,
        1.88195874702691e-4,
        8.30579218603960e-5,
        3.48461963201376e-3,
        3.61873162287129e-6,
        -1.49875789940005e-2,
        1.22979142197165e-8,
        -1.29213888778954e-3,
    ],
    # d(8,1)
    (7, 0): [
        -6.22209937489143e-5,
        -6.44890425871692e-5,
        -3.02936928954918e-5,
        -1.33386143898282e-3,
        -1.31215186728213e-6,
        6.70228205200379e-3,
        -4.44653967516776e-9,
        3.15659916047599e-4,
    ],
    # d(9,1)
    (8, 0): [
        6.84740881090240e-6,
        1.14082245705934e-5,
        3.03727593705750e-6,
        2.36122782444105e-4,
        1.26768491232397e-7,
        -1.53347270556276e-3,
        4.21617557752767e-10,
        -4.51948990428065e-5,
    ],
    # d(2,2)
    (1, 1): [
        2.13188763071246e-6,
        -7.41025068776257e-5,
        2.31652037371554e-6,
        -2.59495924602038e-3,
        1.20637183170338e-7,
        5.21123771632193e-2,
        4.42258843694177e-10,
        -4.20967682664542e-7,
    ],
}


def _round_half_away_from_zero(x: NDArray | float) -> NDArray:
    return np.floor(np.abs(x) + 0.5) * np.sign(x)


def _evaluate_slots(r: NDArray | float) -> dict[tuple[int, int], NDArray]:
    r = np.asarray(r, dtype=np.float64)
    return {key: np.polyval(np.asarray(coef, dtype=np.float64), r) for key, coef in _COEFFS.items()}


def stencil_table(*, cfl: float) -> NDArray[np.float64]:
    """Return the ``(9, 2)`` stencil coefficients at Courant number ``cfl``."""
    table = np.zeros((9, 2), dtype=np.float64)
    for (row, column), value in _evaluate_slots(float(cfl)).items():
        table[row, column] = float(value)
    return table


def n_zones(*, sound_speed: NDArray) -> int:
    """Return the number of one-m/s sound speed zones that span ``sound_speed``."""
    sound_speed = np.asarray(sound_speed, dtype=np.float64)
    spread = float(np.max(sound_speed)) - float(np.min(sound_speed))
    return int(_round_half_away_from_zero(spread)) + 1


def zone_table(*, sound_speed: NDArray, dt: float, dx: float) -> NDArray[np.float64]:
    """Return the ``(9, 2, n_zones)`` stencil coefficients of every sound speed zone."""
    sound_speed = np.asarray(sound_speed, dtype=np.float64)
    slowest = float(np.min(sound_speed))
    zones = n_zones(sound_speed=sound_speed)
    zone_speed = np.arange(zones, dtype=np.float64) + slowest
    table = np.zeros((9, 2, zones), dtype=np.float64)
    for (row, column), value in _evaluate_slots(zone_speed * (float(dt) / float(dx))).items():
        table[row, column, :] = value
    return table


def zone_index_map(*, sound_speed: NDArray) -> NDArray[np.int64]:
    """Return the 0-based sound speed zone of every cell."""
    sound_speed = np.asarray(sound_speed, dtype=np.float64)
    slowest = float(np.min(sound_speed))
    zones = n_zones(sound_speed=sound_speed)
    index = _round_half_away_from_zero(_round_half_away_from_zero(sound_speed) - slowest)
    return np.clip(index.astype(np.int64), 0, zones - 1)
