"""Find, inspect and run a user-supplied fullwave2-ultra binary."""

from __future__ import annotations

import mmap
import os
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from fullwave.solver import cuda_utils

from .errors import NO_BINARY_MESSAGE, UltraBinaryError

ENV_VAR = "FW2U_BATCH_BIN"
BINARY_NAME_2D = "bench_2d_batch"
BINARY_NAME_3D = "bench_3d_opt"

MARKER_3D = b"nZ.dat"
MARKER_AIR_CELLS_3D = b"icczero.dat"
MARKER_AIR_WINDOW_3D = b"nzero0.dat"
MARKER_ADDITIVE = b"ncoords_add.dat"
MARKER_ADDITIVE_SIGNALS = b"icmat_add.dat"
MARKER_AIR_HOLD_2D = b"nTzero.dat"
MARKER_AIR_WINDOW_2D = b"nzero0.dat"
MARKER_SOURCE_RELEASE_2D = b"nszero0.dat"
MARKER_HOLD_SOURCE_3D = b"nszero0.dat"


class PreflightError(RuntimeError):
    """Raised when no usable CUDA device is visible."""


@dataclass(frozen=True, kw_only=True)
class BinaryFeatures:
    """The inputs a binary reads, found by the file names in its bytes."""

    is_3d: bool
    has_air_cells_3d: bool
    has_additive_channel: bool
    has_additive_signals: bool
    has_air_hold_2d: bool
    has_source_release_2d: bool


def resolve_binary(path: str | Path | None, *, name: str) -> Path:
    """Return the binary from ``path`` or ``$FW2U_BATCH_BIN`` (a file, or a folder with ``name``).

    Raises
    ------
    UltraBinaryError
        When nothing is given or the file does not exist.
    """
    given = path if path is not None else os.environ.get(ENV_VAR)
    if not given:
        raise UltraBinaryError(NO_BINARY_MESSAGE)
    given = Path(given)
    found = given / name if given.is_dir() else given
    if not found.is_file():
        msg = f"fullwave2-ultra binary {str(found)!r} does not exist. {NO_BINARY_MESSAGE}"
        raise UltraBinaryError(msg)
    return found


def inspect_binary(binary: str | Path) -> BinaryFeatures:
    """Return the features of ``binary`` (cached until the file changes)."""
    path = Path(binary).resolve()
    stat = path.stat()
    found = _markers(_FileStamp(path=path, size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    return BinaryFeatures(
        is_3d=MARKER_3D in found,
        has_air_cells_3d=MARKER_AIR_CELLS_3D in found and MARKER_AIR_WINDOW_3D in found,
        has_additive_channel=MARKER_ADDITIVE in found,
        has_additive_signals=MARKER_ADDITIVE_SIGNALS in found,
        has_air_hold_2d=MARKER_AIR_HOLD_2D in found or MARKER_AIR_WINDOW_2D in found,
        has_source_release_2d=MARKER_SOURCE_RELEASE_2D in found,
    )


class _FileStamp(NamedTuple):
    """A file and the size and modification time that tell whether it changed."""

    path: Path
    size: int
    mtime_ns: int


@lru_cache(maxsize=64)
def _markers(stamp: _FileStamp) -> frozenset[bytes]:
    if stamp.size == 0:
        return frozenset()
    needles = (
        MARKER_3D,
        MARKER_AIR_CELLS_3D,
        MARKER_ADDITIVE,
        MARKER_ADDITIVE_SIGNALS,
        MARKER_AIR_HOLD_2D,
        MARKER_AIR_WINDOW_2D,
        MARKER_SOURCE_RELEASE_2D,
    )
    with stamp.path.open("rb") as fh, mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        return frozenset(n for n in needles if mm.find(n) != -1)


def preflight(device: int) -> None:
    """Raise :class:`PreflightError` unless CUDA device ``device`` is visible."""
    try:
        specs = cuda_utils.get_cuda_architecture()
    except Exception as error:
        msg = f"CUDA preflight failed: {error}"
        raise PreflightError(msg) from error
    if device >= len(specs):
        msg = f"CUDA device {device} not available ({len(specs)} GPU(s) visible)"
        raise PreflightError(msg)


def run_binary(
    binary: str | Path,
    rundir: str | Path,
    *,
    device: int | None = 0,
    verify_gpu: bool = True,
    timeout: float | None = None,
) -> str:
    """Run ``binary`` in ``rundir`` and return its standard output.

    Raises
    ------
    RuntimeError
        When the binary exits with a non-zero code.
    """
    if verify_gpu:
        preflight(device or 0)
    env = os.environ.copy()
    if device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(device)
    result = subprocess.run(  # noqa: S603
        [str(binary)],
        cwd=str(rundir),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if result.returncode != 0:
        tail = (result.stderr or result.stdout)[-2000:]
        msg = f"{Path(binary).name} failed (returncode={result.returncode}):\n{tail}"
        raise RuntimeError(msg)
    return result.stdout or ""
