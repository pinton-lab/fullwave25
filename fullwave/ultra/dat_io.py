"""Raw little-endian ``.dat`` scalar, source and output I/O."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def write_int(path: str | Path, value: int) -> None:
    """Write one ``int32`` to ``path``."""
    np.asarray([value], dtype=np.int32).tofile(path)


def write_float(path: str | Path, value: float) -> None:
    """Write one ``float32`` to ``path``."""
    np.asarray([value], dtype=np.float32).tofile(path)


def write_map(path: str | Path, array: NDArray) -> None:
    """Write a 2D or 3D map as C-order ``float32``."""
    np.ascontiguousarray(array, dtype=np.float32).tofile(path)


def write_dcmap(path: str | Path, zone_index: NDArray, n_zones: int) -> None:
    """Write the per-cell stencil zone index as ``int32``, clamped to the valid zones."""
    clamped = np.clip(np.asarray(zone_index, dtype=np.int64), 0, n_zones - 1)
    np.ascontiguousarray(clamped.astype(np.int32)).tofile(path)


def write_coords_2d(path: str | Path, i_idx: NDArray, j_idx: NDArray) -> None:
    """Write 0-based coordinates as ``int32``: all i, then all j."""
    columns = [np.asarray(idx, dtype=np.int32).ravel() for idx in (i_idx, j_idx)]
    np.concatenate(columns).tofile(path)


def write_table(path: str | Path, table: NDArray) -> None:
    """Write a stencil table as C-order ``float32``."""
    np.ascontiguousarray(table, dtype=np.float32).tofile(path)


def write_icmat(path: str | Path, blocks: list[NDArray]) -> None:
    """Write one ``(ncoords, nTic)`` source block per simulation as ``float32``."""
    with Path(path).open("wb") as f:
        for block in blocks:
            np.ascontiguousarray(block, dtype=np.float32).tofile(f)


def read_genout(path: str | Path, *, ncoordsout: int) -> NDArray[np.float32]:
    """Read ``genout.dat`` as ``(n_frames, ncoordsout)`` ``float32``."""
    return np.fromfile(path, dtype=np.float32).reshape(-1, ncoordsout)


def genout_name(member: int) -> str:
    """Return the output file name of batch member ``member``."""
    return "genout.dat" if member == 0 else f"genout_s{member}.dat"
