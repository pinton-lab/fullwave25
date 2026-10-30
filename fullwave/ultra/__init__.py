"""Experimental fullwave2-ultra backend, reached through ``Solver(use_ultra=True)``."""

from .errors import (
    NO_BINARY_MESSAGE,
    FullwaveUltraExperimentalWarning,
    UltraBinaryError,
    UltraUnsupportedError,
)

__all__ = [
    "NO_BINARY_MESSAGE",
    "FullwaveUltraExperimentalWarning",
    "UltraBinaryError",
    "UltraUnsupportedError",
]
