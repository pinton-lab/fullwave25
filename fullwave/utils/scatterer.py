"""Generate scatterer maps for acoustic simulations.

This module provides functionality to create random scatterer distributions
for acoustic wave simulations using the fullwave package.
"""

import numpy as np
from numpy.typing import NDArray

from fullwave import Grid


def _verify_seed(rng: np.random.Generator | None, seed: int | None) -> np.random.Generator:
    if seed is not None and rng is not None:
        message = "Provide either seed or rng, not both."
        raise ValueError(message)
    elif seed is None and rng is None:  # noqa: RET506
        message = "Provide either seed or rng."
        raise ValueError(message)
    elif seed is not None and rng is None:
        rng = np.random.default_rng(seed=seed)
    return rng


def generate_wave_length_based_scatterer(
    grid: Grid,
    ratio_scatterer_num_to_wavelength: float = 0.3,
    scatter_value_std: float = 0.08,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], int]:
    """Generate a scatterer map with random values.

    Parameters
    ----------
    grid : Grid
        Grid object from fullwave.
    ratio_scatterer_num_to_wavelength : float, optional
        Ratio of scatterer number to wavelength, by default 0.3.
        It determines the scatterer number based on wavelength.
    scatter_value_std : float, optional
        Standard deviation of scatterer values, by default 0.08.
    seed : int | None, optional
        Random seed for reproducibility, by default None.
    rng : np.random.Generator | None, optional
        Random number generator, by default None.

    Returns
    -------
    tuple[NDArray[np.float64], int]
        Tuple containing the scatterer map,
        and number of scatterers per wavelength.

    """
    rng = _verify_seed(rng, seed)

    num_scatterer_per_wavelength = int(ratio_scatterer_num_to_wavelength * grid.ppw)
    num_scatterer_total = (
        int(
            grid.nx * grid.ny * grid.nz / grid.ppw**3 * num_scatterer_per_wavelength**3,
        )
        if grid.is_3d
        else int(
            grid.nx * grid.ny / grid.ppw**2 * num_scatterer_per_wavelength**2,
        )
    )

    scatterer = np.ones(grid.shape, dtype=float)

    scatterer_indices = rng.choice(
        grid.nx * grid.ny * grid.nz if grid.is_3d else grid.nx * grid.ny,
        size=int(num_scatterer_total),
        replace=False,
    )

    scatterer_values = rng.normal(
        loc=1.0,
        scale=scatter_value_std,
        size=int(num_scatterer_total),
    )
    scatterer.flat[scatterer_indices] = scatterer_values
    scatterer[scatterer < 0] = 0.0

    return scatterer, num_scatterer_per_wavelength


def _resolution_cell(
    wavelength: float,
    dy2: float,
    ay: float,
    n_cycles: int,
    dy: float,
    dz: float,
) -> float:
    res_y = wavelength * dy2 / ay
    res_z = wavelength * n_cycles / 2
    return res_y / dy * res_z / dz


def generate_resolution_based_scatterer(
    grid: Grid,
    num_scatterer: int,
    ncycles: int,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], int, float]:
    """Generate a scatterer map based on resolution cell.

    based on
    https://github.com/gfpinton/fullwave2/blob/f00c4bcbf031897c748bea2ffabe1ca636234fa1/rescell2d.m

    Parameters
    ----------
    grid : Grid
        Grid object from fullwave.
    num_scatterer : int
        Number of scatterers to generate.
    ncycles : int
        Number of pulse cycles.
    seed : int | None, optional
        Random seed for reproducibility, by default None.
    rng : np.random.Generator | None, optional
        Random number generator, by default None.

    Returns
    -------
    tuple[NDArray[np.float64], int, float]
        Tuple containing the scatterer map, total number of scatterers,
        and scatterer percentage.

    """
    rng = _verify_seed(rng, seed)

    resolution_cell = _resolution_cell(
        wavelength=grid.wavelength,
        dy2=grid.ny / 2 * grid.dy,
        ay=grid.domain_size[1],
        n_cycles=ncycles,
        dy=grid.dx,
        dz=grid.dy,
    )
    scat_density = num_scatterer / resolution_cell
    scatter_map = rng.random(grid.shape)

    scatter_map /= scat_density
    scatter_map[scatter_map > 1] = 0.5
    scatter_map -= 0.5

    scatterer_count = (scatter_map != 1).sum().item()
    grid_num_points = grid.nx * grid.ny * (grid.nz if grid.is_3d else 1)
    scatterer_percent = 100 * scatterer_count / grid_num_points
    return scatter_map, scatterer_count, scatterer_percent
