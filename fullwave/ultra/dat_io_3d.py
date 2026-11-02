"""Raw little-endian ``.dat`` I/O of 3D coordinates."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import ArrayLike, NDArray

ICMAT_ADD_BLOCK_STEPS = 256


def write_coords_3d(
    path: str | Path,
    i_idx: ArrayLike,
    j_idx: ArrayLike,
    k_idx: ArrayLike,
) -> None:
    """Write 0-based coordinates as ``int32``: all i, then all j, then all k."""
    columns = [np.asarray(idx, dtype=np.int32).ravel() for idx in (i_idx, j_idx, k_idx)]
    np.concatenate(columns).tofile(path)


def write_icmat_add_time_major(
    path: str | Path, signal: NDArray, *, block_steps: int = ICMAT_ADD_BLOCK_STEPS
) -> None:
    """Write an ``(ncoords_add, n_tic_add)`` signal as ``float32``, one time step after another.

    The transpose is written in blocks of ``block_steps`` steps, so no full copy is made.
    """
    signal = np.asarray(signal)
    with Path(path).open("wb") as f:
        for first in range(0, signal.shape[1], block_steps):
            block = signal[:, first : first + block_steps]
            np.ascontiguousarray(block.T, dtype=np.float32).tofile(f)
