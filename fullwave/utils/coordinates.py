"""It manipulates the coordinates of the mask."""

import numpy as np
from numpy.typing import NDArray


def make_circle_idx(
    dims: NDArray[np.float64],
    cen: NDArray[np.float64],
    rad: float,
) -> NDArray[np.bool_]:
    """Make a circle index mask.

    Args:
        dims: The dimensions of the mask.
        cen: The center of the circle.
        rad: The radius of the circle.

    Returns:
        mask: The mask of the circle.

    """
    x, y = np.meshgrid(np.arange(dims[0]), np.arange(dims[1]), indexing="ij")
    dist = np.sqrt(np.round((x - cen[0]) + 1e-9) ** 2 + np.round((y - cen[1]) + 1e-9) ** 2)
    return dist <= rad


def map_to_coords(
    map_data: NDArray[np.float64 | np.int64 | np.bool],
    *,
    export_as_xyz: bool = False,
) -> NDArray[np.int64]:
    """Map the mask map to coordinates.

    Returns:
        NDArray[np.int64]: An array of coordinates corresponding to non-zero elements in the mask.

    """
    is_3d = map_data.ndim == 3
    # indices = np.where(map_data.T != 0)
    indices = np.where(map_data != 0)
    if is_3d:
        # out = np.array([indices[2], indices[1], indices[0]]).T
        # out = np.array([indices[2], indices[1], indices[0]]).T
        out = np.array([*indices]).T

        if export_as_xyz:
            out = np.stack([out[:, 2], out[:, 1], out[:, 0]], axis=1)
    else:
        out = np.array([*indices]).T
        if export_as_xyz:
            out = np.stack([out[:, 1], out[:, 0]], axis=1)
    return out


def coords_to_map(
    coords: NDArray[np.int64],
    grid_shape: NDArray[np.int64] | tuple[int, ...],
    *,
    is_3d: bool = False,
) -> NDArray[np.int64]:
    """Map the coordinates to a mask map.

    Parameters
    ----------
    coords: NDArray[np.int64]
        The coordinates to map.
    grid_shape: tuple[int, ...]
        The shape of the grid.
    is_3d: bool
        Whether the grid is 3D or not.

    Returns
    -------
    NDArray[np.int64]: The mask map.

    """
    mask = np.zeros(grid_shape, dtype=int)
    if is_3d:
        mask = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2]), dtype=int)
        mask[coords[:, 0].astype(int), coords[:, 1].astype(int), coords[:, 2].astype(int)] = 1
    else:
        mask = np.zeros((grid_shape[0], grid_shape[1]), dtype=int)
        mask[coords[:, 0].astype(int), coords[:, 1].astype(int)] = 1
    return mask


def coords_to_index_map(
    coords: NDArray[np.int64],
    grid_shape: NDArray[np.int64] | tuple[int, ...],
    *,
    is_3d: bool = False,
) -> NDArray[np.int64]:
    """Map the coordinates to an index map."""
    mask = np.zeros(grid_shape, dtype=int)
    if is_3d:
        mask = np.zeros((grid_shape[0], grid_shape[1], grid_shape[2]), dtype=int)
        mask[coords[:, 0].astype(int), coords[:, 1].astype(int), coords[:, 2].astype(int)] = (
            np.arange(coords.shape[0]) + 1
        )
    else:
        mask = np.zeros((grid_shape[0], grid_shape[1]), dtype=int)
        mask[coords[:, 0].astype(int), coords[:, 1].astype(int)] = np.arange(coords.shape[0]) + 1
    return mask


def map_to_coords_with_sort(map_data: NDArray[np.int64]) -> NDArray[np.int64]:
    """Map the mask map to coordinates with sorting.

    Args:
        map_data: The mask map.

    Returns:
        NDArray[np.int64]: An array of coordinates corresponding to non-zero elements in the mask.

    """
    coords = map_to_coordinates(map_data)
    return coords[:, np.argsort(coords[0], kind="mergesort")].T


def map_to_coordinates(
    map_data: NDArray[np.int64],
    *,
    is_3d: bool = False,
    sort: bool = False,
) -> NDArray[np.int64]:
    """Map the mask map to coordinates.

    Args:
        map_data: The mask map.
        is_3d: Whether the grid is 3D or not.
        sort: Whether to sort the coordinates by the first dimension.

    Returns:
        NDArray[np.int64]: An array of coordinates corresponding to non-zero elements in the mask.

    """
    if is_3d:
        idx, idy, idz = np.where(map_data != 0)
        if idx.shape[0] == 0 or idy.shape[0] == 0 or idz.shape[0] == 0:
            return np.array([[], [], []]).T
        coords = np.array([idx, idy, idz])
    else:
        idx, idy = np.where(map_data != 0)
        if idx.shape[0] == 0 or idy.shape[0] == 0:
            return np.array([[], []]).T
        coords = np.array([idx, idy])

    unique_num_list = np.unique(coords[1])
    unique_num_list.sort()
    out_list = [np.sort(coords[:, coords[1] == value]) for value in unique_num_list]
    coords = np.concatenate(out_list, axis=1)
    if sort:
        return coords[:, np.argsort(coords[0], kind="mergesort")]
    return coords
