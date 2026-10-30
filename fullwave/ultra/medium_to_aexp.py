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


def dbmhzcm2aexp(alpha: NDArray, *, c0: float, omega0: float, dt: float) -> NDArray[np.float64]:
    """Convert ``alpha`` [dB/(MHz cm)] to the per-step decay multiplier."""
    neper_in_db = -20.0 * np.log10(np.exp(-1.0))
    f0_mhz = omega0 / 2.0 / np.pi / 1e6
    nepers_per_s = np.asarray(alpha, dtype=np.float64) * f0_mhz * c0 / 1e-2 / neper_in_db
    return np.exp(-dt * nepers_per_s)


def sponge_nd(shape: tuple[int, ...], nbdy: int) -> NDArray[np.float64]:
    """Return a sin^2 absorbing ring of ``nbdy`` cells on every face, ones inside."""
    out = np.ones(shape, dtype=np.float64)
    for axis, n in enumerate(shape):
        profile = np.ones(n, dtype=np.float64)
        for b in range(1, nbdy + 1):
            v = np.sin(np.pi / 2.0 * (b - 1) / nbdy) ** 2
            profile[b - 1] = min(profile[b - 1], v)
            profile[n - b] = min(profile[n - b], v)
        out *= profile.reshape([n if a == axis else 1 for a in range(len(shape))])
    return out


def aexp_from_amap(
    amap_ext: NDArray,
    *,
    c0: float,
    omega0: float,
    dt: float,
    nbdy: int,
) -> NDArray[np.float64]:
    """Return the decay multiplier of the padded attenuation map times the sponge (2D or 3D)."""
    amap_ext = np.asarray(amap_ext, dtype=np.float64)
    return dbmhzcm2aexp(amap_ext, c0=c0, omega0=omega0, dt=dt) * sponge_nd(amap_ext.shape, nbdy)
