"""Generate scatterer maps for acoustic simulations.

This module provides functionality to create random scatterer distributions
for acoustic wave simulations using the fullwave package.
"""

import logging

import numpy as np
from numpy.typing import NDArray

from fullwave import Grid

logger = logging.getLogger("__main__." + __name__)


def _verify_seed(rng: np.random.Generator | None, seed: int | None) -> np.random.Generator:
    if seed is not None and rng is not None:
        message = "Provide either seed or rng, not both."
        logger.error(message)
        raise ValueError(message)
    elif seed is None and rng is None:  # noqa: RET506
        message = "Provide either seed or rng."
        logger.error(message)
        raise ValueError(message)
    elif seed is not None and rng is None:
        rng = np.random.default_rng(seed=seed)
    return rng


def _check_value_within_limit(value: float, limit: tuple) -> None:
    if value <= limit[0] or value >= limit[1]:
        message = f"value {value} must be between {limit[0]} and {limit[1]}."
        logger.error(message)
        raise ValueError(message)


def _generate_scatterer_from_num_scatterer(
    grid: Grid,
    rng: np.random.Generator,
    num_scatterer_total: float,
    scatter_value_std: float,
) -> NDArray[np.float64]:
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
    return scatterer


def generate_scatterer(
    grid: Grid,
    scatter_value_std: float = 0.08,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    *,
    ratio_scatterer_to_total_grid: float | None = None,
    ratio_scatterer_num_to_wavelength: float | None = None,
    num_scatterer_per_wavelength: int | None = None,
) -> tuple[NDArray[np.float64], dict]:
    """Generate a scatterer map with random values.

    This function switches between different generation methods based on input parameters.
    Exactly one of the scatterer density parameters must be provided.

    Parameters
    ----------
    grid : Grid
        Grid object from fullwave.
    scatter_value_std : float, optional
        Standard deviation of scatterer values, by default 0.08.
    seed : int | None, optional
        Random seed for reproducibility, by default None.
    rng : np.random.Generator | None, optional
        Random number generator, by default None.
    ratio_scatterer_to_total_grid : float | None, optional
        Ratio of scatterer to total grid (0 < ratio < 1), by default None.
    ratio_scatterer_num_to_wavelength : float | None, optional
        Ratio of scatterer number to wavelength (0 < ratio < 1), by default None.
    num_scatterer_per_wavelength : int | None, optional
        Number of scatterers per wavelength (0 < num < grid.ppw), by default None.

    Returns
    -------
    tuple[NDArray[np.float64], dict]
        Tuple containing:
        - scatterer_map (NDArray[np.float64]): The generated scatterer map.
        - scatterer_info (dict):
            Dictionary containing information
            about the scatterer distribution with the following keys:
            - "num_scatterer_total": Total number of scatterers placed in the grid.
            - "num_scatterer_per_wavelength": Number of scatterers per wavelength.
            - "ratio_scatterer_num_to_wavelength":
                Ratio of scatterers per wavelength to points per wavelength.
            - "ratio_scatterer_to_total_grid": Ratio of scatterers to total grid points.

    Raises
    ------
    ValueError
        If none or multiple scatterer density parameters are provided.

    """
    params_provided = sum(
        [
            ratio_scatterer_to_total_grid is not None,
            ratio_scatterer_num_to_wavelength is not None,
            num_scatterer_per_wavelength is not None,
        ],
    )

    if params_provided == 0:
        message = (
            "Must provide exactly one of: "
            "ratio_scatterer_to_total_grid, ratio_scatterer_num_to_wavelength, "
            "or num_scatterer_per_wavelength"
        )
        logger.error(message)
        raise ValueError(message)
    if params_provided > 1:
        message = (
            "Must provide only one of: "
            "ratio_scatterer_to_total_grid, ratio_scatterer_num_to_wavelength, "
            "or num_scatterer_per_wavelength"
        )
        logger.error(message)
        raise ValueError(message)

    if ratio_scatterer_to_total_grid is not None:
        return generate_scatterer_from_ratio_num_scatterer_to_total_grid(
            grid=grid,
            ratio_scatterer_to_total_grid=ratio_scatterer_to_total_grid,
            scatter_value_std=scatter_value_std,
            seed=seed,
            rng=rng,
        )
    if ratio_scatterer_num_to_wavelength is not None:
        return generate_scatterer_from_ratio_num_scatterer_to_wavelength(
            grid=grid,
            ratio_scatterer_num_to_wavelength=ratio_scatterer_num_to_wavelength,
            scatter_value_std=scatter_value_std,
            seed=seed,
            rng=rng,
        )

    # num_scatterer_per_wavelength is not None
    return generate_scatterer_from_num_scatterer_per_wavelength(
        grid=grid,
        num_scatterer_per_wavelength=num_scatterer_per_wavelength,
        scatter_value_std=scatter_value_std,
        seed=seed,
        rng=rng,
    )


def generate_scatterer_from_ratio_num_scatterer_to_total_grid(
    grid: Grid,
    ratio_scatterer_to_total_grid: float = 0.38,
    scatter_value_std: float = 0.08,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], dict]:
    """Generate a scatterer map with random values.

    Parameters
    ----------
    grid : Grid
        Grid object from fullwave.
    ratio_scatterer_to_total_grid : float, optional
        Ratio of scatterer to total grid, by default 0.38.
        It indicates how dense the scatterers are placed in the whole grid.
        0 <= ratio_scatterer_to_total_grid <= 1
    scatter_value_std : float, optional
        Standard deviation of scatterer values, by default 0.08.
    seed : int | None, optional
        Random seed for reproducibility, by default None.
    rng : np.random.Generator | None, optional
        Random number generator, by default None.

    Returns
    -------
    tuple[NDArray[np.float64], dict]
        Tuple containing:
        - scatterer_map (NDArray[np.float64]): The generated scatterer map.
        - scatterer_info (dict):
            Dictionary containing information
            about the scatterer distribution with the following keys:
            - "num_scatterer_total": Total number of scatterers placed in the grid.
            - "num_scatterer_per_wavelength": Number of scatterers per wavelength.
            - "ratio_scatterer_num_to_wavelength":
                Ratio of scatterers per wavelength to points per wavelength.
            - "ratio_scatterer_to_total_grid": Ratio of scatterers to total grid points.

    """
    rng = _verify_seed(rng, seed)

    _check_value_within_limit(ratio_scatterer_to_total_grid, (0.0, 1.0))
    _check_value_within_limit(scatter_value_std, (0.0, 1.0))
    # ratio_scatterer_to_total_grid
    num_scatterer_total = (
        int(ratio_scatterer_to_total_grid * grid.shape[0] * grid.shape[1] * grid.shape[2])
        if grid.is_3d
        else int(ratio_scatterer_to_total_grid * grid.shape[0] * grid.shape[1])
    )
    num_scatterer_per_wavelength = (
        int(
            (num_scatterer_total * grid.ppw**3 / (grid.nx * grid.ny * grid.nz)) ** (1 / 3),
        )
        if grid.is_3d
        else int(
            (num_scatterer_total * grid.ppw**2 / (grid.nx * grid.ny)) ** (1 / 2),
        )
    )
    ratio_scatterer_num_to_wavelength = num_scatterer_per_wavelength / grid.ppw

    scatterer = _generate_scatterer_from_num_scatterer(
        grid,
        rng,
        num_scatterer_total,
        scatter_value_std,
    )

    scatterer_info = {
        "num_scatterer_total": num_scatterer_total,
        "num_scatterer_per_wavelength": num_scatterer_per_wavelength,
        "ratio_scatterer_num_to_wavelength": ratio_scatterer_num_to_wavelength,
        "ratio_scatterer_to_total_grid": ratio_scatterer_to_total_grid,
    }

    return scatterer, scatterer_info


def generate_scatterer_from_ratio_num_scatterer_to_wavelength(
    grid: Grid,
    ratio_scatterer_num_to_wavelength: float = 0.6,
    scatter_value_std: float = 0.08,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], dict]:
    """Generate a scatterer map with random values from ratio of scatterer number to wavelength.

    Parameters
    ----------
    grid : Grid
        Grid object from fullwave.
    ratio_scatterer_num_to_wavelength : float, optional
        Ratio of scatterer number to wavelength, by default 0.6.
        It indicates how dense the scatterers are place in a wavelength.
        0 <= ratio <= 1
    scatter_value_std : float, optional
        Standard deviation of scatterer values, by default 0.08.
    seed : int | None, optional
        Random seed for reproducibility, by default None.
    rng : np.random.Generator | None, optional
        Random number generator, by default None.

    Returns
    -------
    tuple[NDArray[np.float64], dict]
        Tuple containing:
        - scatterer_map (NDArray[np.float64]): The generated scatterer map.
        - scatterer_info (dict):
            Dictionary containing information
            about the scatterer distribution with the following keys:
            - "num_scatterer_total": Total number of scatterers placed in the grid.
            - "num_scatterer_per_wavelength": Number of scatterers per wavelength.
            - "ratio_scatterer_num_to_wavelength":
                Ratio of scatterers per wavelength to points per wavelength.
            - "ratio_scatterer_to_total_grid": Ratio of scatterers to total grid points.

    """
    rng = _verify_seed(rng, seed)

    _check_value_within_limit(ratio_scatterer_num_to_wavelength, (0.0, 1.0))
    _check_value_within_limit(scatter_value_std, (0.0, 1.0))

    num_scatterer_per_wavelength = ratio_scatterer_num_to_wavelength * grid.ppw
    num_scatterer_total = (
        int(
            grid.nx * grid.ny * grid.nz / grid.ppw**3 * num_scatterer_per_wavelength**3,
        )
        if grid.is_3d
        else int(
            grid.nx * grid.ny / grid.ppw**2 * num_scatterer_per_wavelength**2,
        )
    )
    ratio_scatterer_to_total_grid = num_scatterer_total / (grid.shape[0] * grid.shape[1])

    scatterer = _generate_scatterer_from_num_scatterer(
        grid,
        rng,
        num_scatterer_total,
        scatter_value_std,
    )

    scatterer_info = {
        "num_scatterer_total": num_scatterer_total,
        "num_scatterer_per_wavelength": num_scatterer_per_wavelength,
        "ratio_scatterer_num_to_wavelength": ratio_scatterer_num_to_wavelength,
        "ratio_scatterer_to_total_grid": ratio_scatterer_to_total_grid,
    }

    return scatterer, scatterer_info


def generate_scatterer_from_num_scatterer_per_wavelength(
    grid: Grid,
    num_scatterer_per_wavelength: int = 6,
    scatter_value_std: float = 0.08,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[NDArray[np.float64], dict]:
    """Generate a scatterer map with random values.

    Parameters
    ----------
    grid : Grid
        Grid object from fullwave.
    num_scatterer_per_wavelength : int, optional
        Number of scatterers per wavelength, by default 6.
        It indicates how many pixels are placed in a wavelength.
        0 <= num_scatterer_per_wavelength <= grid.ppw
    scatter_value_std : float, optional
        Standard deviation of scatterer values, by default 0.08.
    seed : int | None, optional
        Random seed for reproducibility, by default None.
    rng : np.random.Generator | None, optional
        Random number generator, by default None.

    Returns
    -------
    tuple[NDArray[np.float64], dict]
        Tuple containing:
        - scatterer_map (NDArray[np.float64]): The generated scatterer map.
        - scatterer_info (dict):
            Dictionary containing information
            about the scatterer distribution with the following keys:
            - "num_scatterer_total": Total number of scatterers placed in the grid.
            - "num_scatterer_per_wavelength": Number of scatterers per wavelength.
            - "ratio_scatterer_num_to_wavelength":
                Ratio of scatterers per wavelength to points per wavelength.
            - "ratio_scatterer_to_total_grid": Ratio of scatterers to total grid points.

    """
    _check_value_within_limit(num_scatterer_per_wavelength, (0.0, grid.ppw))
    _check_value_within_limit(scatter_value_std, (0.0, 1.0))

    rng = _verify_seed(rng, seed)
    ratio_scatterer_num_to_wavelength = num_scatterer_per_wavelength / grid.ppw

    # num_scatterer_per_wavelength = int(ratio_scatterer_num_to_wavelength * grid.ppw)
    num_scatterer_total = (
        int(
            grid.nx * grid.ny * grid.nz / grid.ppw**3 * num_scatterer_per_wavelength**3,
        )
        if grid.is_3d
        else int(
            grid.nx * grid.ny / grid.ppw**2 * num_scatterer_per_wavelength**2,
        )
    )
    scatterer_ratio = num_scatterer_total / (grid.shape[0] * grid.shape[1])

    scatterer = _generate_scatterer_from_num_scatterer(
        grid,
        rng,
        num_scatterer_total,
        scatter_value_std,
    )

    scatterer_info = {
        "num_scatterer_total": num_scatterer_total,
        "num_scatterer_per_wavelength": num_scatterer_per_wavelength,
        "ratio_scatterer_num_to_wavelength": ratio_scatterer_num_to_wavelength,
        "ratio_scatterer_to_total_grid": scatterer_ratio,
    }

    return scatterer, scatterer_info


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
) -> tuple[NDArray[np.float64], dict]:
    """Generate a scatterer map based on resolution cell.

    based on
    https://github.com/gfpinton/fullwave2/blob/f00c4bcbf031897c748bea2ffabe1ca636234fa1/rescell2d.m

    Parameters
    ----------
    grid : Grid
        Grid object from fullwave.
    num_scatterer : int
        Number of scatterers to generate per resolution cell.
    ncycles : int
        Number of pulse cycles.
    seed : int | None, optional
        Random seed for reproducibility, by default None.
    rng : np.random.Generator | None, optional
        Random number generator, by default None.

    Returns
    -------
    tuple[NDArray[np.float64], dict]
        Tuple containing:
        - scatterer_map (NDArray[np.float64]): The generated scatterer map.
        - scatterer_info (dict):
            Dictionary containing information
            about the scatterer distribution with the following keys:
            - "num_scatterer_total": Total number of scatterers placed in the grid.
            - "num_scatterer_per_wavelength": Number of scatterers per wavelength.
            - "ratio_scatterer_num_to_wavelength":
                Ratio of scatterers per wavelength to points per wavelength.
            - "ratio_scatterer_to_total_grid": Ratio of scatterers to total grid points.


    """
    rng = _verify_seed(rng, seed)
    _check_value_within_limit(num_scatterer, (0.0, np.inf))
    _check_value_within_limit(ncycles, (0, np.inf))

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

    scatterer_info = {
        "num_scatterer_total": scatterer_count,
        "num_scatterer_per_wavelength": scat_density * grid.ppw,
        "ratio_scatterer_num_to_wavelength": scat_density,
        "ratio_scatterer_to_total_grid": scatterer_count / grid_num_points,
    }
    return scatter_map, scatterer_info
