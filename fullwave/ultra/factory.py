"""Pick the 2D or 3D fullwave2-ultra backend for a grid."""

from __future__ import annotations

from typing import TYPE_CHECKING

from .backend import UltraBackend3D
from .backend_2d import UltraBackend2D

if TYPE_CHECKING:
    import fullwave


def make_backend(
    *,
    grid: fullwave.Grid,
    **inputs: object,
) -> UltraBackend3D | UltraBackend2D:
    """Return the 3D backend for a 3D grid and the 2D backend otherwise."""
    backend_class = UltraBackend3D if grid.is_3d else UltraBackend2D
    return backend_class(grid=grid, **inputs)
