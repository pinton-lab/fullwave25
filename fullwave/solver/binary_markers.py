"""Find out whether a solver binary reads an input file, by the file name in its bytes."""

import mmap
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

HOLD_SOURCE_MARKER = b"hold_source.dat"
DEFAULT_FIX = "rebuild the binary, or set FULLWAVE25_BINARY_TAG to the release that has it"


class HoldSourceUnsupportedError(RuntimeError):
    """Raised when the binary cannot hold the hard source after the drive ends."""


class _FileStamp(NamedTuple):
    """A file with the size and modification time that tell whether it changed."""

    path: Path
    size: int
    mtime_ns: int


@lru_cache(maxsize=64)
def _find(stamp: _FileStamp, marker: bytes) -> bool:
    if stamp.size == 0:
        return False
    with stamp.path.open("rb") as file, mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ) as mm:
        return mm.find(marker) != -1


def binary_has_marker(binary: str | Path, *, marker: bytes) -> bool:
    """Return whether the bytes of ``binary`` contain ``marker``.

    Parameters
    ----------
    binary : str | Path
        The solver binary.
    marker : bytes
        The name of the input file the binary reads.

    Returns
    -------
    bool
        True when the binary contains the name. The answer is cached until the file changes.

    """
    path = Path(binary).resolve()
    stat = path.stat()
    return _find(_FileStamp(path=path, size=stat.st_size, mtime_ns=stat.st_mtime_ns), marker)


def check_binary_holds_source(
    binary: str | Path,
    *,
    marker: bytes = HOLD_SOURCE_MARKER,
    error_type: type[Exception] = HoldSourceUnsupportedError,
    fix: str = DEFAULT_FIX,
) -> None:
    """Refuse a binary that would release the hard source cells when the drive ends.

    Parameters
    ----------
    binary : str | Path
        The solver binary.
    marker : bytes, optional
        The name of the input file that tells the binary to hold the source.
    error_type : type[Exception], optional
        The exception to raise.
    fix : str, optional
        What to do to get a binary that reads the file, worded for the message.

    Raises
    ------
    Exception
        ``error_type``, when the binary does not contain ``marker``.

    """
    if binary_has_marker(binary, marker=marker):
        return
    message = (
        f"{Path(binary).name} does not read {marker.decode()}, so it releases the hard source "
        "cells when the drive signal ends and source_type='hard' would not hold them. "
        f"Use a binary that reads it: {fix}. "
        "source_type='hard_released_after' needs no such binary."
    )
    raise error_type(message)
