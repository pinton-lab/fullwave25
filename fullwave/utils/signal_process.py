"""Signal utilities."""

import logging

import numpy as np
from numpy.typing import NDArray

from fullwave.grid import Grid

logger = logging.getLogger("__main__." + __name__)


def signed_power_compression(x: NDArray[np.float64], power: float = 1 / 3) -> NDArray[np.float64]:
    """Apply signed power compression to the input array.

    The function applies a power compression to the absolute value of the input array,
    while preserving the sign of the original values.

    Parameters
    ----------
    x: NDArray[np.float32]
        Input array to apply power compression to.
    power: float
        Power compression factor. Default is 1/3.

    Returns
    -------
    NDArray[np.float32]
        Array with power compression applied.

    """
    return np.sign(x) * np.abs(x) ** power


def reshape_whole_sensor_to_nx_ny_nt(
    sensor_output: NDArray[np.float64],
    grid: Grid,
) -> NDArray[np.float64]:
    """Reshape sensor output to nx, ny, nt.

    Parameters
    ----------
    sensor_output: NDArray[np.float64]
        Sensor output.
    grid: Grid
        Grid instance.

    Returns
    -------
    NDArray[np.float64]
        Sensor output reshaped to [nx, ny, nt]

    """
    return sensor_output.reshape((grid.ny, grid.nx, sensor_output.shape[-1])).transpose(
        1,
        0,
        2,
    )


def reshape_whole_sensor_to_nt_nx_ny(
    sensor_output: NDArray[np.float64],
    grid: Grid = None,
    nx: int | None = None,
    ny: int | None = None,
    *,
    use_grid: bool = True,
) -> NDArray[np.float64]:
    """Reshape sensor output to nt, nx, ny.

    Parameters
    ----------
    sensor_output: NDArray[np.float64]
        Sensor output.
    grid: Grid
        Grid instance.
    nx: int | None
        Number of grid points in x direction. If None, will use grid.nx.
    ny: int | None
        Number of grid points in y direction. If None, will use grid.ny.
    use_grid: bool
        If True, will use grid.nx and grid.ny. If False, will use nx and ny parameters.
        Default is True.

    Raises
    ------
    ValueError
        If neither grid nor both nx and ny are provided.

    Returns
    -------
    NDArray[np.float64]
        Sensor output reshaped to [nt, nx, ny]

    """
    if grid is not None and (nx is not None or ny is not None):
        error_msg = "Either grid or both nx and ny must be provided, not both."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if use_grid:
        nx = grid.nx
        ny = grid.ny
    elif nx is None or ny is None:
        error_msg = "Either grid or both nx and ny must be provided."
        logger.error(error_msg)
        raise ValueError(error_msg)
    else:
        nx = int(nx)
        ny = int(ny)
    return sensor_output.reshape((nx, ny, sensor_output.shape[-1])).transpose(
        2,
        0,
        1,
    )


# def reshape_whole_sensor_to_nt_nx_ny_nz(
#     sensor_output: NDArray[np.float64],
#     grid: Grid,
# ) -> NDArray[np.float64]:
#     """Reshape sensor output to nt, nx, ny.

#     Parameters
#     ----------
#     sensor_output: NDArray[np.float64]
#         Sensor output.
#     grid: Grid
#         Grid instance.

#     Returns
#     -------
#     NDArray[np.float64]
#         Sensor output reshaped to [nt, nx, ny]

#     """
#     return sensor_output.reshape((grid.nz, grid.ny, grid.nx, sensor_output.shape[-1])).transpose(
#         3,
#         2,
#         1,
#         0,
#     )


def reshape_whole_sensor_to_nt_nx_ny_nz(
    sensor_output: NDArray[np.float64],
    grid: Grid = None,
    nx: int | None = None,
    ny: int | None = None,
    nz: int | None = None,
    *,
    use_grid: bool = True,
) -> NDArray[np.float64]:
    """Reshape sensor output to nt, nx, ny.

    Parameters
    ----------
    sensor_output: NDArray[np.float64]
        Sensor output.
    grid: Grid
        Grid instance.
    nx: int | None
        Number of grid points in x direction. If None, will use grid.nx.
    ny: int | None
        Number of grid points in y direction. If None, will use grid.ny.
    nz: int | None
        Number of grid points in z direction. If None, will use grid.nz.
    use_grid: bool
        If True, will use grid.nx and grid.ny. If False, will use nx and ny parameters.
        Default is True.

    Raises
    ------
    ValueError
        If neither grid nor both nx and ny are provided.

    Returns
    -------
    NDArray[np.float64]
        Sensor output reshaped to [nt, nx, ny]

    """
    if grid is not None and (nx is not None or ny is not None):
        error_msg = "Either grid or both nx and ny must be provided, not both."
        logger.error(error_msg)
        raise ValueError(error_msg)

    if use_grid:
        nx = grid.nx
        ny = grid.ny
        nz = grid.nz
    elif nx is None or ny is None:
        error_msg = "Either grid or both nx and ny must be provided."
        logger.error(error_msg)
        raise ValueError(error_msg)
    else:
        nx = int(nx)
        ny = int(ny)
        nz = int(nz)
    return sensor_output.reshape((nx, ny, nz, sensor_output.shape[-1])).transpose(
        3,
        0,
        1,
        2,
    )
