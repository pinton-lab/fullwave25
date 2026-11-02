"""combined phantom domain."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fullwave import Grid
from fullwave.utils import generate_scatterer

from fullwave.medium_builder.domain import Domain  # isort:skip
from fullwave.solver.shipped_database import ShippedDatabase

from .combined_phantom_map import (
    BACKGROUND,
    CONTRAST_TARGET_RADIUS_M,
    DEFAULT_CONTRAST_TARGET_OFFSET_M,
    HYPERECHOIC_TARGET,
    HYPOECHOIC_TARGET,
    POINT_TARGET,
    contrast_target_centers_m,
    generate_combined_phantom_map,
    point_target_centers_m,
)

SCATTERER_STD_FACTORS = {
    BACKGROUND: 1.0,
    HYPOECHOIC_TARGET: 0.2,
    HYPERECHOIC_TARGET: 3.0,
}
POINT_TARGET_CONTRAST = 1.2


class CombinedPhantomDomain(Domain):
    """An imaging phantom with point targets and contrast targets on a scattering background."""

    def __init__(
        self,
        grid: Grid,
        *,
        name: str = "combined_phantom",
        crop_start: list[int] | None = None,
        crop_end: list[int] | None = None,
        shift_px: list[int] | None = None,
        use_center: bool = False,
        use_px_based_point_target: bool = True,
        use_point_target: bool = True,
        use_contrast_target: bool = True,
        contrast_target_offset_m: float = DEFAULT_CONTRAST_TARGET_OFFSET_M,
        ratio_scatterer_to_total_grid: float = 0.38,
        base_scatterer_ratio: float = 0.035,
        base_sound_speed: float = 1540,
        base_density: float = 1000,
        base_alpha_coeff: float = 0.5,
        base_alpha_power: float = 1.0,
        base_beta: float = 0.0,
        rng: np.random.Generator | None = None,
        seed: int = 42,
        path_relaxation_parameters_database: Path = ShippedDatabase.table,
        n_relaxation_mechanisms: int = ShippedDatabase.mechanisms,
    ) -> None:
        """Initialize a CombinedPhantomDomain.

        The phantom is drawn on a canvas of 312.8 mm in depth and 181.4 mm in width
        (see ``generate_combined_phantom_map``). The grid sees the window of the canvas
        from ``crop_start`` to ``crop_end``, moved by ``shift_px``.
        The labels set the scatterer strength:
        hypoechoic targets 0.2 times and hyperechoic targets 3 times the background,
        and point targets are single dense inclusions without scatterers.

        Parameters
        ----------
        grid : Grid
            The 2D grid.
        name : str, optional
            The name of the domain.
        crop_start : list[int] | None, optional
            [depth, lateral] cell where the window starts on the canvas. Default [0, 0].
        crop_end : list[int] | None, optional
            [depth, lateral] cell where the window ends. Default is crop_start plus the grid shape.
        shift_px : list[int] | None, optional
            [depth, lateral] shift of the window content in cells. The cells shifted in are
            background. Default [0, 0].
        use_center : bool, optional
            True starts the window at the top of the canvas and centers it laterally,
            replacing crop_start and crop_end.
        use_px_based_point_target : bool, optional
            True draws a point target as one cell. False draws a disk of a quarter wavelength
            in radius.
        use_point_target : bool, optional
            Whether to include point targets.
        use_contrast_target : bool, optional
            Whether to include contrast targets.
        contrast_target_offset_m : float, optional
            Lateral distance [m] of each contrast target center from the center of the phantom.
        ratio_scatterer_to_total_grid : float, optional
            Ratio of cells that hold a scatterer.
        base_scatterer_ratio : float, optional
            Standard deviation of the background density relative to the base density.
        base_sound_speed : float, optional
            Background sound speed [m/s].
        base_density : float, optional
            Background density [kg/m^3].
        base_alpha_coeff : float, optional
            Background attenuation coefficient [dB/(MHz^y cm)].
        base_alpha_power : float, optional
            Background attenuation power y.
        base_beta : float, optional
            Background nonlinearity parameter.
        rng : np.random.Generator | None, optional
            Random number generator for the scatterers. Default is seeded with ``seed``.
        seed : int, optional
            Seed of the default random number generator.
        path_relaxation_parameters_database : Path, optional
            The path to the relaxation parameters database.
        n_relaxation_mechanisms : int, optional
            The number of relaxation mechanisms.

        Raises
        ------
        ValueError
            If the grid is 3D, if crop_start is negative or if crop_end is before crop_start.

        """
        if grid.is_3d:
            message = "CombinedPhantomDomain supports 2D grids only."
            raise ValueError(message)
        crop_start = crop_start or [0, 0]
        if min(crop_start) < 0:
            message = "crop_start must be non-negative. Use shift_px to shift the phantom."
            raise ValueError(message)
        crop_end = crop_end or [crop_start[0] + grid.nx, crop_start[1] + grid.ny]
        if crop_end[0] < crop_start[0] or crop_end[1] < crop_start[1]:
            message = "crop_end must be greater than crop_start."
            raise ValueError(message)

        self.grid = grid
        self.is_3d = False
        self.rng = rng if rng is not None else np.random.default_rng(seed=seed)
        self.ratio_scatterer_to_total_grid = ratio_scatterer_to_total_grid
        self.base_scatterer_ratio = base_scatterer_ratio
        self.base_sound_speed = base_sound_speed
        self.base_density = base_density
        self.base_alpha_coeff = base_alpha_coeff
        self.base_alpha_power = base_alpha_power
        self.base_beta = base_beta

        canvas = generate_combined_phantom_map(
            dx_m=grid.dx,
            point_target_radius_px=0 if use_px_based_point_target else grid.ppw // 4,
            use_point_target=use_point_target,
            use_contrast_target=use_contrast_target,
            contrast_target_offset_m=contrast_target_offset_m,
        )
        if use_center:
            crop_start = [0, canvas.shape[1] // 2 - grid.ny // 2]
            crop_end = [grid.nx, crop_start[1] + grid.ny]
        window = _crop_and_pad(canvas, crop_start=crop_start, crop_end=crop_end, shape=grid.shape)
        shift_px = shift_px or [0, 0]
        self.phantom_map = _shift_with_background_fill(window, shift_px=shift_px)

        self.contrast_target_offset_m = contrast_target_offset_m
        self.use_point_target = use_point_target
        self.use_contrast_target = use_contrast_target
        self._canvas_center_col = canvas.shape[1] // 2
        self._crop_start = crop_start
        self._window_shape = (crop_end[0] - crop_start[0], crop_end[1] - crop_start[1])
        self._shift_px = shift_px

        self.base_geometry = np.ones(grid.shape)
        self.sound_speed = np.full(grid.shape, base_sound_speed, dtype=np.float64)
        self.density = np.full(grid.shape, base_density, dtype=np.float64)
        self.alpha_coeff = np.full(grid.shape, base_alpha_coeff, dtype=np.float64)
        self.alpha_power = np.full(grid.shape, base_alpha_power, dtype=np.float64)
        self.beta = np.full(grid.shape, base_beta, dtype=np.float64)
        self.air = np.zeros(grid.shape, dtype=np.float64)
        self.scatterer = self._draw_scatterer()
        self.density *= self.scatterer
        self._add_point_targets()

        super().__init__(
            name=name,
            grid=grid,
            path_relaxation_parameters_database=path_relaxation_parameters_database,
            n_relaxation_mechanisms=n_relaxation_mechanisms,
        )

    def contrast_target_circles_in_grid_m(self) -> list[tuple[float, float, float]]:
        """Return the contrast target cylinders that lie in the grid.

        Returns
        -------
        list[tuple[float, float, float]]
            (lateral position, depth, radius) of each cylinder center in meters, in the
            frame of ``Grid.extent_mm``: lateral from the center of the grid and depth
            from its top. It follows the crop, shift and center options.

        """
        if not self.use_contrast_target:
            return []
        circles = []
        for _, depth_m, offset_m in contrast_target_centers_m(self.contrast_target_offset_m):
            center_m = self._grid_frame_position_m(depth_m=depth_m, offset_m=offset_m)
            if center_m is not None:
                circles.append((*center_m, CONTRAST_TARGET_RADIUS_M))
        return circles

    def point_target_centers_in_grid_m(self) -> list[tuple[float, float]]:
        """Return the point targets that lie in the grid.

        Returns
        -------
        list[tuple[float, float]]
            (lateral position, depth) of each point target in meters, in the frame of
            ``Grid.extent_mm``. It follows the crop, shift and center options.

        """
        if not self.use_point_target:
            return []
        centers = []
        for depth_m, offset_m in point_target_centers_m():
            center_m = self._grid_frame_position_m(depth_m=depth_m, offset_m=offset_m)
            if center_m is not None:
                centers.append(center_m)
        return centers

    def _grid_frame_position_m(
        self,
        *,
        depth_m: float,
        offset_m: float,
    ) -> tuple[float, float] | None:
        dx_m = self.grid.dx
        window_row = int(depth_m / dx_m) - self._crop_start[0]
        window_col = self._canvas_center_col + int(offset_m / dx_m) - self._crop_start[1]
        is_in_window = 0 <= window_row < self._window_shape[0] and (
            0 <= window_col < self._window_shape[1]
        )
        row = window_row + self._shift_px[0]
        col = window_col + self._shift_px[1]
        if not (is_in_window and 0 <= row < self.grid.nx and 0 <= col < self.grid.ny):
            return None
        return (col + 0.5 - self.grid.ny / 2) * dx_m, (row + 0.5) * dx_m

    def _draw_scatterer(self) -> NDArray[np.float64]:
        scatterer = np.ones(self.grid.shape)
        for label, std_factor in SCATTERER_STD_FACTORS.items():
            region = self.phantom_map == label
            label_scatterer, _ = generate_scatterer(
                grid=self.grid,
                ratio_scatterer_to_total_grid=self.ratio_scatterer_to_total_grid,
                scatter_value_std=std_factor * self.base_scatterer_ratio,
                rng=self.rng,
            )
            scatterer[region] = label_scatterer[region]
        return scatterer

    def _add_point_targets(self) -> None:
        point_target = self.phantom_map == POINT_TARGET
        self.scatterer[point_target] = 1.0
        self.density[point_target] = POINT_TARGET_CONTRAST * self.base_density
        self.sound_speed[point_target] = POINT_TARGET_CONTRAST * self.base_sound_speed

    def _setup_base_geometry(self) -> NDArray[np.float64]:
        return self.base_geometry

    def _setup_sound_speed(self) -> NDArray[np.float64]:
        return self.sound_speed

    def _setup_density(self) -> NDArray[np.float64]:
        return self.density

    def _setup_beta(self) -> NDArray[np.float64]:
        return self.beta

    def _setup_alpha_coeff(self) -> NDArray[np.float64]:
        return self.alpha_coeff

    def _setup_alpha_power(self) -> NDArray[np.float64]:
        return self.alpha_power

    def _setup_air(self) -> NDArray[np.int64]:
        return self.air


def _crop_and_pad(
    canvas: NDArray[np.int8],
    *,
    crop_start: list[int],
    crop_end: list[int],
    shape: tuple[int, ...],
) -> NDArray[np.int8]:
    window = canvas[crop_start[0] : crop_end[0], crop_start[1] : crop_end[1]]
    return np.pad(
        window,
        ((0, shape[0] - window.shape[0]), (0, shape[1] - window.shape[1])),
        constant_values=BACKGROUND,
    )


def _shift_with_background_fill(
    label_map: NDArray[np.int8],
    *,
    shift_px: list[int],
) -> NDArray[np.int8]:
    shifted = np.full_like(label_map, BACKGROUND)
    rows_from, rows_to = _shift_slices(label_map.shape[0], shift_px[0])
    cols_from, cols_to = _shift_slices(label_map.shape[1], shift_px[1])
    shifted[rows_to, cols_to] = label_map[rows_from, cols_from]
    return shifted


def _shift_slices(length: int, shift: int) -> tuple[slice, slice]:
    if shift >= 0:
        return slice(0, length - shift), slice(shift, length)
    return slice(-shift, length), slice(0, length + shift)
