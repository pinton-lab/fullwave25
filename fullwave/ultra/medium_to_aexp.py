"""Medium padding and the exponential attenuation map (``Aexp``) with its sponge."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def extend_map(volume: NDArray, n: int) -> NDArray[np.float64]:
    """Pad ``volume`` by ``n`` cells on every side by replicating its edge."""
    volume = np.asarray(volume, dtype=np.float64)
    return volume.copy() if n == 0 else np.pad(volume, n, mode="edge")


AXIS_NAMES = ("x", "y", "z")


def face_names(ndim: int) -> tuple[str, ...]:
    """Return the face names ``x_low``, ``x_high``, ``y_low``, ... of an ``ndim``-D grid."""
    return tuple(f"{AXIS_NAMES[a]}_{side}" for a in range(ndim) for side in ("low", "high"))


def face_margins(
    *, ndim: int, mext: int, shell: int, nbdy: int, back_faces: frozenset[str]
) -> dict[str, int]:
    """Return the cells added on each face: ``shell + nbdy`` on a back face, ``mext`` elsewhere.

    On a back face the outermost ``shell`` cells are the frozen shell, the next ``nbdy`` cells
    are the sponge, and the first source row follows directly.

    Parameters
    ----------
    ndim : int
        Number of grid axes, 2 or 3.
    mext : int
        Cells added on a face that is not a back face (sponge plus stencil cells).
    shell : int
        Cells of the frozen shell, the outermost cells of a back face.
    nbdy : int
        Sponge thickness in cells; the sponge of a back face lies between the shell and the
        grid.
    back_faces : frozenset[str]
        Faces behind a held source on a grid edge, named as in ``face_names``.

    Raises
    ------
    ValueError
        When a name in ``back_faces`` is not a face of the grid.
    """
    unknown = set(back_faces) - set(face_names(ndim))
    if unknown:
        msg = f"unknown faces {sorted(unknown)}; the faces are {face_names(ndim)}"
        raise ValueError(msg)
    return {name: shell + nbdy if name in back_faces else mext for name in face_names(ndim)}


def extend_map_faces(volume: NDArray, margins: dict[str, int]) -> NDArray[np.float64]:
    """Pad ``volume`` by the cells of each face in ``margins``, replicating its edge."""
    volume = np.asarray(volume, dtype=np.float64)
    pads = [
        (margins[f"{AXIS_NAMES[a]}_low"], margins[f"{AXIS_NAMES[a]}_high"])
        for a in range(volume.ndim)
    ]
    return np.pad(volume, pads, mode="edge")


def back_faces_of_source(
    *, incoords: NDArray, shape: tuple[int, ...], held: bool
) -> frozenset[str]:
    """Return the faces behind a held source whose cells touch that grid edge.

    A face is a back face when the source is held, its cells touch that edge of the grid, and
    the edge is normal to the thin axis of the source (the axis of its smallest extent, which
    must be strictly smaller than the others).

    Parameters
    ----------
    incoords : NDArray
        0-based interior ``(N, ndim)`` source cells.
    shape : tuple[int, ...]
        Interior grid shape.
    held : bool
        True when the source is held at zero pressure after its signal.

    Returns
    -------
    frozenset[str]
        The back faces; empty when the source is not held, has no cells, or has no unique thin axis.
    """
    cells = np.asarray(incoords)
    if not held or cells.size == 0:
        return frozenset()
    cells = cells[:, : len(shape)]
    lows, highs = cells.min(axis=0), cells.max(axis=0)
    extents = highs - lows + 1
    thin = int(np.argmin(extents))
    if np.count_nonzero(extents == extents[thin]) != 1:
        return frozenset()
    faces = set()
    if lows[thin] == 0:
        faces.add(f"{AXIS_NAMES[thin]}_low")
    if highs[thin] == shape[thin] - 1:
        faces.add(f"{AXIS_NAMES[thin]}_high")
    return frozenset(faces)


def dbmhzcm2aexp(alpha: NDArray, *, c0: float, omega0: float, dt: float) -> NDArray[np.float64]:
    """Convert ``alpha`` [dB/(MHz cm)] to the per-step decay multiplier."""
    neper_in_db = -20.0 * np.log10(np.exp(-1.0))
    f0_mhz = omega0 / 2.0 / np.pi / 1e6
    nepers_per_s = np.asarray(alpha, dtype=np.float64) * f0_mhz * c0 / 1e-2 / neper_in_db
    return np.exp(-dt * nepers_per_s)


def sponge_nd(
    shape: tuple[int, ...],
    nbdy: int,
    *,
    back_faces: frozenset[str] = frozenset(),
    frozen_cells: int = 0,
) -> NDArray[np.float64]:
    """Return a sin^2 absorbing ring of ``nbdy`` cells on every face, ones inside.

    The ring of a face named in ``back_faces`` starts ``frozen_cells`` cells inside the grid
    edge: the outer ``frozen_cells`` cells (the frozen shell) are zero (full damping),
    the next ``nbdy`` cells follow the same profile as the other faces with full damping on the
    outer side, and no undamped cell lies between the ring and the interior.
    """
    out = np.ones(shape, dtype=np.float64)
    for axis, n in enumerate(shape):
        profile = np.ones(n, dtype=np.float64)
        low_skip = frozen_cells if f"{AXIS_NAMES[axis]}_low" in back_faces else 0
        high_skip = frozen_cells if f"{AXIS_NAMES[axis]}_high" in back_faces else 0
        profile[:low_skip] = 0.0
        if high_skip:
            profile[n - high_skip :] = 0.0
        for b in range(1, nbdy + 1):
            v = np.sin(np.pi / 2.0 * (b - 1) / nbdy) ** 2
            profile[low_skip + b - 1] = min(profile[low_skip + b - 1], v)
            profile[n - high_skip - b] = min(profile[n - high_skip - b], v)
        out *= profile.reshape([n if a == axis else 1 for a in range(len(shape))])
    return out


def aexp_from_amap(
    amap_ext: NDArray,
    *,
    c0: float,
    omega0: float,
    dt: float,
    nbdy: int,
    back_faces: frozenset[str] = frozenset(),
    frozen_cells: int = 0,
) -> NDArray[np.float64]:
    """Return the decay multiplier of the padded attenuation map times the sponge (2D or 3D).

    The sponge of a face named in ``back_faces`` starts ``frozen_cells`` cells inside the grid
    edge.
    """
    amap_ext = np.asarray(amap_ext, dtype=np.float64)
    return dbmhzcm2aexp(amap_ext, c0=c0, omega0=omega0, dt=dt) * sponge_nd(
        amap_ext.shape, nbdy, back_faces=back_faces, frozen_cells=frozen_cells
    )
