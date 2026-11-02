"""Label map of the combined phantom: point targets and contrast targets."""

import numpy as np
from numpy.typing import NDArray

BACKGROUND = 0
HYPOECHOIC_TARGET = 1
HYPERECHOIC_TARGET = 2
POINT_TARGET = 3

CANVAS_DEPTH_M = 312.8e-3
CANVAS_WIDTH_M = 181.4e-3

VERTICAL_POINT_DEPTHS_MM = (*range(10, 60, 5), *range(60, 260, 10))
HORIZONTAL_POINT_DEPTHS_MM = (10, 30, 50, 70)
HORIZONTAL_POINT_OFFSETS_MM = (-20, -15, -10, -5, 5, 10, 15, 20)
CONTRAST_TARGET_DEPTHS_MM = range(20, 220, 20)
CONTRAST_TARGET_RADIUS_M = 4e-3
DEFAULT_CONTRAST_TARGET_OFFSET_M = 10e-3


def generate_combined_phantom_map(
    *,
    dx_m: float,
    point_target_radius_px: int = 0,
    use_point_target: bool = True,
    use_contrast_target: bool = True,
    contrast_target_offset_m: float = DEFAULT_CONTRAST_TARGET_OFFSET_M,
) -> NDArray[np.int8]:
    """Draw the label map of the combined phantom.

    The map is laid out on a canvas of 312.8 mm in depth and 181.4 mm in width.
    Depth runs down the first axis from the top of the canvas.
    The lateral center of the phantom is the middle column.

    The point targets are one vertical column at the center, from 10 mm to 250 mm
    deep, and four horizontal rows at 10, 30, 50 and 70 mm with offsets
    of 5 to 20 mm to both sides.
    The contrast targets are two columns of ten cylinders of 8 mm diameter
    every 20 mm from 20 mm deep, ``contrast_target_offset_m`` to the left
    (hypoechoic) and to the right (hyperechoic) of the center.
    The cylinders are 10 mm deeper or shallower than the point target rows,
    so they do not touch the point targets.

    Parameters
    ----------
    dx_m : float
        Cell size [m].
    point_target_radius_px : int, optional
        Radius of a point target in cells. 0 draws a single cell.
    use_point_target : bool, optional
        Whether to draw the point targets.
    use_contrast_target : bool, optional
        Whether to draw the contrast targets.
    contrast_target_offset_m : float, optional
        Lateral distance [m] of each cylinder center from the center of the phantom.

    Returns
    -------
    NDArray[np.int8]
        Labels BACKGROUND, HYPOECHOIC_TARGET, HYPERECHOIC_TARGET and POINT_TARGET,
        shape [depth cells, lateral cells].

    """
    label_map = np.zeros((int(CANVAS_DEPTH_M / dx_m), int(CANVAS_WIDTH_M / dx_m)), dtype=np.int8)
    center_col = label_map.shape[1] // 2

    def draw(*, depth_m: float, offset_m: float, radius_px: int, label: int) -> None:
        _draw_disk(
            label_map,
            center_px=(int(depth_m / dx_m), center_col + int(offset_m / dx_m)),
            radius_px=radius_px,
            label=label,
        )

    if use_contrast_target:
        radius_px = int(CONTRAST_TARGET_RADIUS_M / dx_m)
        for label, depth_m, offset_m in contrast_target_centers_m(contrast_target_offset_m):
            draw(depth_m=depth_m, offset_m=offset_m, radius_px=radius_px, label=label)

    if use_point_target:
        for depth_m, offset_m in point_target_centers_m():
            draw(
                depth_m=depth_m,
                offset_m=offset_m,
                radius_px=point_target_radius_px,
                label=POINT_TARGET,
            )
    return label_map


def contrast_target_centers_m(
    contrast_target_offset_m: float = DEFAULT_CONTRAST_TARGET_OFFSET_M,
) -> list[tuple[int, float, float]]:
    """Return the centers of the contrast targets on the canvas.

    Parameters
    ----------
    contrast_target_offset_m : float, optional
        Lateral distance [m] of each cylinder center from the center of the phantom.

    Returns
    -------
    list[tuple[int, float, float]]
        (label, depth [m] from the top of the canvas, lateral offset [m] from the center),
        left (hypoechoic) before right (hyperechoic) for each depth.

    """
    centers = []
    for depth_mm in CONTRAST_TARGET_DEPTHS_MM:
        centers.append((HYPOECHOIC_TARGET, depth_mm * 1e-3, -contrast_target_offset_m))
        centers.append((HYPERECHOIC_TARGET, depth_mm * 1e-3, contrast_target_offset_m))
    return centers


def point_target_centers_m() -> list[tuple[float, float]]:
    """Return the centers of the point targets on the canvas.

    Returns
    -------
    list[tuple[float, float]]
        (depth [m] from the top of the canvas, lateral offset [m] from the center).

    """
    centers = [(depth_mm * 1e-3, 0.0) for depth_mm in VERTICAL_POINT_DEPTHS_MM]
    centers.extend(
        (depth_mm * 1e-3, offset_mm * 1e-3)
        for depth_mm in HORIZONTAL_POINT_DEPTHS_MM
        for offset_mm in HORIZONTAL_POINT_OFFSETS_MM
    )
    return centers


def _draw_disk(
    label_map: NDArray[np.int8],
    *,
    center_px: tuple[int, int],
    radius_px: int,
    label: int,
) -> None:
    row, col = center_px
    rows = slice(max(row - radius_px, 0), min(row + radius_px + 1, label_map.shape[0]))
    cols = slice(max(col - radius_px, 0), min(col + radius_px + 1, label_map.shape[1]))
    row_index, col_index = np.ogrid[rows, cols]
    inside_disk = (row_index - row) ** 2 + (col_index - col) ** 2 <= radius_px**2
    label_map[rows, cols][inside_disk] = label
