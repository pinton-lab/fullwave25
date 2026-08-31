"""Transducer class for Fullwave.

adapted and modified from k-wave-python
https://github.com/waltsims/k-wave-python/blob/4590a9445ebf8cdd2b719e32ee792d3752f2f55a/kwave/ktransducer.py
"""

import logging
from functools import cached_property
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import fullwave
from fullwave import transmit
from fullwave.grid import Grid
from fullwave.solver import source_type as source_type_module
from fullwave.utils import check_functions
from fullwave.utils.coordinates import make_circle_idx, map_to_coordinates, map_to_coords_with_sort

logger = logging.getLogger("__main__." + __name__)


def _make_pos_int(val: float | tuple[float] | tuple[int]) -> NDArray[np.int64]:
    """Force value to be a positive integer.

    Returns
    -------
        NDArray[np.int64]: Array with positive integers.

    """
    return np.array(val).astype(int).clip(min=0)


class TransducerGeometry:
    """base transducer class."""

    def __init__(
        self,
        grid: Grid,
        number_elements: int = 128,
        element_width_m: float | None = None,
        element_height_m: float | None = None,
        element_spacing_m: float | None = None,
        position_m: tuple[float, float] | tuple[float, float, float] | None = None,
        element_layer_m: float | None = None,
        radius: float = float("inf"),
        element_width_px: int | None = None,
        element_height_px: int | None = None,
        element_spacing_px: int | None = None,
        element_layer_px: int | None = None,
        position_px: tuple[int, int] | tuple[int, int, int] | None = None,
        *,
        validate_input: bool = True,
        zero_offset: float = 0.0124,
        average_surface_signals: bool = True,
    ) -> None:
        """Initialize base transducer class.

        Parameters
        ----------
        grid: Grid
            Grid object
        number_elements:
            the total number of transducer elements
        element_width_m:
            the width of each element in m
        element_height_m:
            the height of each element in m. only used for 3D simulations.
        element_spacing_m:
            the spacing (kerf width) between the transducer elements in m
        position_m:
            the position of the corner of the transducer in m
        element_layer_m:
            the thickness of the transducer elements in m
        radius:
            the radius of curvature of the transducer [m]
        element_width_px:
            the width of each transducer element in pixels
        element_height_px:
            the height of each transducer element in pixels.
        element_spacing_px:
            the spacing (kerf width) in pixels between transducer elements
        element_layer_px:
            the thickness of each transducer element in pixels
        position_px:
            the position of the transducer in pixels
        validate_input: bool, optional
            Flag indicating whether to validate the input data.
            default is True.
        zero_offset: float
            The zero offset for the convex transducer position in meters.
            default is 0.0124 m. This value is only used for convex transducers (radius < inf).
        average_surface_signals: bool
            The class sets the output sensor mask to either entire element surface or center pixel.
            If True, the entire element surface is used for output sensor mask.
            The data can be post-processed to average
            the surface signals using transducer.post_process_sensor_output().
            If False, only the center pixel is used for output sensor mask.
            In case of convex transducer, this option is ignored
            and the entire element surface is used for the averaging.
            default is True.


        Raises
        ------
        ValueError
            If neither pixel nor meter dimensions are provided
            or if the transducer exceeds grid bounds.

        """
        if validate_input:
            check_functions.check_instance(grid, Grid)
        self.average_surface_signals = average_surface_signals
        self.grid = grid
        self.is_3d = grid.is_3d
        (
            self.element_width_px,
            self.element_width_m,
            self.element_height_px,
            self.element_height_m,
            self.element_spacing_px,
            self.element_spacing_m,
            self.element_layer_px,
            self.element_layer_m,
        ) = self._init_dimensions(
            grid,
            element_width_px,
            element_width_m,
            element_height_px,
            element_height_m,
            element_spacing_px,
            element_spacing_m,
            element_layer_px,
            element_layer_m,
        )
        self.element_width_px = _make_pos_int(self.element_width_px)
        self.element_spacing_px = _make_pos_int(self.element_spacing_px)

        self.stored_grid_size = (
            [
                grid.nx,
                grid.ny,
                grid.nz,
            ]
            if self.is_3d
            else [
                grid.nx,
                grid.ny,
            ]
        )
        # size of the grid in which the transducer is defined
        self.grid_spacing = (
            [
                grid.dx,
                grid.dy,
                grid.dz,
            ]
            if self.is_3d
            else [
                grid.dx,
                grid.dy,
            ]
        )
        # corresponding grid spacing

        self.number_elements = _make_pos_int(number_elements)

        self.position_px, self.position_m = self._init_positions(position_px, position_m)

        self.radius = radius
        self.zero_offset = zero_offset

        # check the transducer fits into the grid
        if (
            self.position_px[1]
            + self.number_elements * self.element_width_px
            + (self.number_elements - 1) * self.element_spacing_px
        ) > self.stored_grid_size[1] and self.radius == float("inf"):
            error_msg = (
                "The defined transducer is too large or"
                "positioned outside the grid in the y-direction:\n"
                f"position_px: {self.position_px[1]}, "
                f"number_elements: {self.number_elements}, "
                f"element_width_px: {self.element_width_px}, "
                f"element_spacing_px: {self.element_spacing_px}, "
                f"ny: {self.stored_grid_size[1]}, "
                f"transducer_width_px: {self.transducer_width_px}, "
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        if self.position_px[0] > self.stored_grid_size[0]:
            error_msg = "The defined transducer is positioned outside the grid in the x-direction"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Build coordinate arrays for source and sensor pixels.
        # _source_coords: shape [N_src, ndim], _source_ids: shape [N_src]  (1-based element id)
        # _sensor_coords: shape [N_snsr, ndim], _sensor_ids: shape [N_snsr]
        (
            self._source_coords,
            self._source_ids,
            self._sensor_coords,
            self._sensor_ids,
        ) = self._create_element_coords()
        logger.debug("TransducerGeometry instance created.")

    def _init_dimensions(  # noqa: PLR0912
        self,
        grid: Grid,
        element_width_px: int | None,
        element_width_m: float | None,
        element_height_px: int | None,
        element_height_m: float | None,
        element_spacing_px: int | None,
        element_spacing_m: float | None,
        element_layer_px: int | None,
        element_layer_m: float | None,
    ) -> tuple[int, float, int | None, float | None, int, float, int, float]:
        # Initialize element dimensions by converting between meters and pixels.
        if element_width_px is None and element_width_m is not None:
            element_width_px = round(element_width_m / grid.dy)
            element_width_px = max(1, element_width_px)
            self.use_px_in_width = False
        elif element_width_px is not None and element_width_m is None:
            element_width_m = element_width_px * grid.dy
            self.use_px_in_width = True
        else:
            error_msg = "Either element_width_px or element_width_m must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if self.is_3d is True and element_height_px is None and element_height_m is not None:
            element_height_px = round(element_height_m / grid.dz)
            element_height_px = max(1, element_height_px)
            self.use_px_in_width = False
        elif self.is_3d is True and element_height_px is not None and element_height_m is None:
            element_height_m = element_height_px * grid.dz
            self.use_px_in_width = True
        elif self.is_3d is True and (element_height_px is None and element_height_m is None):
            error_msg = "Either element_height_px or element_height_m must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
        elif self.is_3d is False and (
            element_height_px is not None or element_height_m is not None
        ):
            message = (
                "element_height_px and element_height_m are provided, "
                "but the transducer is not 3D. "
                "Ignoring element_height_px and element_height_m."
            )
            logger.info(message)
        else:
            element_height_px = 0
            element_height_m = 0.0

        if element_spacing_px is None and element_spacing_m is not None:
            element_spacing_px = round(element_spacing_m / grid.dy)
            element_spacing_px = max(0, element_spacing_px)
            self.use_px_in_space = False
        elif element_spacing_px is not None and element_spacing_m is None:
            element_spacing_m = element_spacing_px * grid.dy
            self.use_px_in_space = True
        else:
            error_msg = "Either element_spacing_px or element_spacing_m must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if element_layer_px is None and element_layer_m is not None:
            element_layer_px = round(element_layer_m / grid.dy)
            element_layer_px = max(1, element_layer_px)
        elif element_layer_px is not None and element_layer_m is None:
            element_layer_m = element_layer_px * grid.dy
        else:
            error_msg = "Either element_layer_px or element_layer_m must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)

        return (
            element_width_px,
            max(0, element_width_m),
            element_height_px if self.is_3d else None,
            max(0, element_height_m) if self.is_3d else None,
            element_spacing_px,
            max(0, element_spacing_m),
            element_layer_px,
            max(0, element_layer_m),
        )

    def _init_positions(self, position_px: int, position_m: float) -> tuple[int, float]:
        if position_px is None and position_m is None:
            position_px = (1, 1, 1) if self.is_3d else (1, 1)
            position_px = _make_pos_int(position_px)
            position_m = [
                pos * grid_spacing
                for pos, grid_spacing in zip(position_px, self.grid_spacing, strict=False)
            ]
        elif position_px is not None and position_m is None:
            position_m = [
                pos * grid_spacing
                for pos, grid_spacing in zip(position_px, self.grid_spacing, strict=False)
            ]
            position_px = _make_pos_int(position_px)
        elif position_px is None and position_m is not None:
            position_px = [
                round(pos / grid_spacing)
                for pos, grid_spacing in zip(position_m, self.grid_spacing, strict=False)
            ]
            position_px = _make_pos_int(position_px)
        else:
            error_msg = "Either position_px or position_m must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if self.is_3d:
            assert len(position_px) == 3, "position_px must have 3 elements for 3D transducer"
            assert len(position_m) == 3, "position_m must have 3 elements for 3D transducer"
        return position_px, position_m

    def _create_element_coords(  # noqa: PLR0912
        self,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
        """Build flat coordinate arrays for source and sensor pixels.

        Returns
        -------
        source_coords : NDArray[np.int64], shape [N_src, ndim]
            Grid coordinates of every source pixel.
        source_ids : NDArray[np.int64], shape [N_src]
            1-based element ID for each source pixel.
        sensor_coords : NDArray[np.int64], shape [N_snsr, ndim]
            Grid coordinates of every sensor pixel.
        sensor_ids : NDArray[np.int64], shape [N_snsr]
            1-based element ID for each sensor pixel.

        """
        source_coords_parts: list[NDArray] = []
        source_ids_parts: list[NDArray] = []
        sensor_coords_parts: list[NDArray] = []
        sensor_ids_parts: list[NDArray] = []

        if self.radius == float("inf"):
            if self.is_3d:
                for element_index in range(self.number_elements):
                    element_pos_x = self.position_px[0]
                    element_pos_y = round(
                        (
                            self.position_m[1]
                            + (self.element_width_m + self.element_spacing_m) * element_index
                        )
                        / self.grid_spacing[1],
                    )
                    element_pos_z = round(self.position_m[2] / self.grid_spacing[2])
                    if self.use_px_in_space or self.use_px_in_width:
                        element_pos_y = (
                            self.position_px[1]
                            + (self.element_width_px + self.element_spacing_px) * element_index
                        )

                    xs = np.arange(element_pos_x, element_pos_x + self.element_layer_px)
                    ys = np.arange(element_pos_y, element_pos_y + self.element_width_px)
                    zs = np.arange(element_pos_z, element_pos_z + self.element_height_px)
                    xx, yy, zz = np.meshgrid(xs, ys, zs, indexing="ij")
                    src = np.stack([xx.ravel(), yy.ravel(), zz.ravel()], axis=1)
                    source_coords_parts.append(src)
                    source_ids_parts.append(np.full(len(src), element_index + 1))

                    # Single center pixel for sensor output
                    sx = element_pos_x + self.element_layer_px - 1
                    sy = element_pos_y + self.element_width_px - 1
                    sz = element_pos_z + self.element_height_px // 2 - 1
                    sensor_coords_parts.append(np.array([[sx, sy, sz]]))
                    sensor_ids_parts.append(np.array([element_index + 1]))
            else:
                for element_index in range(self.number_elements):
                    element_pos_x = self.position_px[0]
                    element_pos_y = round(
                        (
                            self.position_m[1]
                            + (self.element_width_m + self.element_spacing_m) * element_index
                        )
                        / self.grid_spacing[1],
                    )
                    if self.use_px_in_space or self.use_px_in_width:
                        element_pos_y = (
                            self.position_px[1]
                            + (self.element_width_px + self.element_spacing_px) * element_index
                        )

                    xs = np.arange(element_pos_x, element_pos_x + self.element_layer_px)
                    ys = np.arange(element_pos_y, element_pos_y + self.element_width_px)
                    xx, yy = np.meshgrid(xs, ys, indexing="ij")
                    src = np.stack([xx.ravel(), yy.ravel()], axis=1)
                    source_coords_parts.append(src)
                    source_ids_parts.append(np.full(len(src), element_index + 1))

                    sx = element_pos_x + self.element_layer_px - 1
                    if self.average_surface_signals:
                        out_ys = np.arange(element_pos_y, element_pos_y + self.element_width_px)
                        out_xs = np.full(len(out_ys), sx)
                    else:
                        out_ys = np.array([element_pos_y + self.element_width_px // 2 - 1])
                        out_xs = np.array([sx])
                    snsr = np.stack([out_xs, out_ys], axis=1)
                    sensor_coords_parts.append(snsr)
                    sensor_ids_parts.append(np.full(len(snsr), element_index + 1))

        else:  # noqa: PLR5501 -- keep the if structure for future extension
            if self.is_3d:
                error_msg = "3D convex transducers are not implemented yet."
                logger.error(error_msg)
                raise NotImplementedError(error_msg)
            else:  # noqa: RET506 -- keep the if structure for future extension
                radius_px = round(self.radius / self.grid.dx)
                d_theta = np.arctan2(self.element_spacing_m / self.grid.dy, radius_px)
                theta_list = self._define_theta_at_center(
                    d_theta=d_theta,
                    num_elements=self.number_elements,
                )
                center = np.array(
                    [
                        self.zero_offset / self.grid.dx - radius_px + self.position_px[0],
                        self.grid.ny // 2 + self.position_px[1],
                    ],
                )
                if not self.average_surface_signals:
                    logger.warning(
                        "average_surface_signals is set to False, "
                        "but it is ignored for convex transducers.",
                    )

                in_map = self._calculate_inmap(center=center, radius=radius_px)
                out_map = self._calculate_outmap(center=center, radius=radius_px)

                in_raw = map_to_coords_with_sort(in_map)
                out_raw = map_to_coords_with_sort(out_map)
                in_raw, out_raw = self._assign_transducer_num_to_input(
                    in_coords=in_raw,
                    out_coords=out_raw,
                    center=center,
                    number_elements=self.number_elements,
                    d_theta=d_theta,
                    theta_list=theta_list,
                )
                # in_raw / out_raw: shape [N, 4], columns [x, y, 0, element_id]
                valid_in = in_raw[:, 3] > 0
                valid_out = out_raw[:, 3] > 0
                return (
                    in_raw[valid_in, :2].astype(np.int64),
                    in_raw[valid_in, 3].astype(np.int64),
                    out_raw[valid_out, :2].astype(np.int64),
                    out_raw[valid_out, 3].astype(np.int64),
                )

        source_coords = np.concatenate(source_coords_parts, axis=0).astype(np.int64)
        source_ids = np.concatenate(source_ids_parts, axis=0).astype(np.int64)
        sensor_coords = np.concatenate(sensor_coords_parts, axis=0).astype(np.int64)
        sensor_ids = np.concatenate(sensor_ids_parts, axis=0).astype(np.int64)
        return source_coords, source_ids, sensor_coords, sensor_ids

    @cached_property
    def indexed_element_mask_input(self) -> NDArray[np.int64]:
        """Indexed element mask for source pixels.

        (element id at each grid point, 0 = background)
        """
        mask = np.zeros(self.stored_grid_size, dtype=int)
        idx = tuple(self._source_coords[:, i] for i in range(self._source_coords.shape[1]))
        mask[idx] = self._source_ids
        return mask

    @cached_property
    def indexed_element_mask_output(self) -> NDArray[np.int64]:
        """Indexed element mask for sensor pixels.

        (element id at each grid point, 0 = background)
        """
        mask = np.zeros(self.stored_grid_size, dtype=int)
        idx = tuple(self._sensor_coords[:, i] for i in range(self._sensor_coords.shape[1]))
        mask[idx] = self._sensor_ids
        return mask

    @property
    def element_mask_input(self) -> NDArray[np.bool_]:
        """Boolean mask of all source pixels across all elements."""
        return self.indexed_element_mask_input > 0

    @property
    def element_mask_output(self) -> NDArray[np.bool_]:
        """Boolean mask of all sensor pixels across all elements."""
        return self.indexed_element_mask_output > 0

    def _assign_transducer_num_to_input(
        self,
        in_coords: NDArray[np.float64],
        out_coords: NDArray[np.float64],
        center: NDArray[np.float64],
        number_elements: int,
        d_theta: float,
        theta_list: NDArray[np.float64],
    ) -> tuple[NDArray[np.float64], ...]:
        # Assign which transducer number is assigned to each input.
        thetas_in = np.arctan2(in_coords[:, 1] - center[1], in_coords[:, 0] - center[0])
        thetas_out = np.arctan2(out_coords[:, 1] - center[1], out_coords[:, 0] - center[0])

        in_coords = np.append(in_coords, np.zeros((in_coords.shape[0], 2)), axis=1)
        in_coords[:, 2] = 0
        in_coords[:, 3] = 0

        out_coords = np.append(out_coords, np.zeros((out_coords.shape[0], 2)), axis=1)
        out_coords[:, 2] = 0
        out_coords[:, 3] = 0

        for tt in range(number_elements):
            # find which incoords are assigned to tt
            less_than_max = thetas_in < (theta_list[tt] + d_theta / 2)
            greater_than_min = thetas_in > (theta_list[tt] - d_theta / 2)
            id_theta = np.where(np.logical_and(less_than_max, greater_than_min))[0]
            in_coords[id_theta, 3] = tt + 1

            # find which outcoords are assigned to tt
            less_than_max = thetas_out < (theta_list[tt] + d_theta / 2)
            greater_than_min = thetas_out > (theta_list[tt] - d_theta / 2)
            id_theta = np.where(np.logical_and(less_than_max, greater_than_min))[0]
            out_coords[id_theta, 3] = tt + 1
        return in_coords, out_coords

    @staticmethod
    def _define_theta_at_center(d_theta: float, num_elements: int) -> NDArray[np.float64]:
        thetas = d_theta * (np.arange((-(num_elements - 1) / 2), ((num_elements - 1) / 2) + 1))
        for n in np.arange(num_elements):
            thetas[n] = (n + 1) * d_theta

        return thetas - np.mean(thetas)

    def _calculate_inmap(self, center: NDArray[np.float64], radius: float) -> np.ndarray:
        # Make a circle that defines the transducer surface
        in_map = np.zeros((self.grid.nx, self.grid.ny))
        in_map[make_circle_idx(in_map.shape, center, radius)] = 1
        output_map = np.zeros((self.grid.nx, self.grid.ny))

        # make outcoords from iccoords
        # Grab the coords on edge of the circle - larger circle for outcoords
        for i in range(self.grid.ny):
            # find inmap coords
            j = np.where(in_map[:, i] == 1)[0]
            if j.shape[0] == 0:
                continue
            j = j[-1]

            output_map[j - self.element_layer_px : j, i] = 1

        return output_map

    def _calculate_outmap(
        self,
        center: NDArray[np.float64],
        radius: float,
    ) -> np.ndarray:
        # Make a circle that defines the transducer surface
        out_map = np.zeros((self.grid.nx, self.grid.ny))
        out_map[make_circle_idx(out_map.shape, center, radius)] = 1
        output_map = np.zeros((self.grid.nx, self.grid.ny))

        # make outcoords from iccoords
        # Grab the coords on edge of the circle - larger circle for outcoords
        for i in range(self.grid.ny):
            # find inmap coords
            j = np.where(out_map[:, i] == 1)[0]
            if j.shape[0] == 0:
                continue
            j = j[-1]

            output_map[j, i] = 1

        return output_map

    @cached_property
    def indexed_element_mask_input_px(self) -> NDArray[np.int64]:
        """Return the pixel wise indexed element mask."""
        out_map = np.zeros_like(self.element_mask_input, dtype=int)
        coordinates = map_to_coordinates(self.element_mask_input).T
        index = 1
        for i in range(len(coordinates)):
            x = coordinates[i][0]
            y = coordinates[i][1]
            out_map[x.astype(int), y.astype(int)] = index
            index += 1
        return out_map

    @property
    def element_pitch_m(self) -> float:
        """Compute the pitch of the transducer elements in the y-direction."""
        return self.element_spacing_m + self.element_width_m

    @property
    def element_pitch_px(self) -> int:
        """Compute the pitch of the transducer elements in the y-direction."""
        return round(self.element_pitch_m / self.grid_spacing[1])

    @property
    def transducer_width_m(self) -> float:
        """Total width of the transducer in meter.

        Returns
        -------
        int
            Total width of the transducer in meter

        """
        return float(
            self.number_elements * self.element_width_m
            + (self.number_elements - 1) * self.element_spacing_m,
        )

    @property
    def transducer_width_px(self) -> int:
        """Total width of the transducer in grid points.

        Returns
        -------
        int
            Total width of the transducer in grid points.

        """
        return int(
            self.number_elements * self.element_width_px
            + (self.number_elements - 1) * self.element_spacing_px,
        )

    @property
    def transducer_surface(self) -> NDArray[np.int64]:
        """Return the coordinates of the transducer surface."""
        if self.radius != float("inf"):
            radius_px = round(self.radius / self.grid.dx)
            center = np.array(
                [
                    self.zero_offset / self.grid.dx - radius_px + self.position_px[0],
                    self.grid.ny // 2 + self.position_px[1],
                ],
            )

            out_map = self._calculate_outmap(
                center=center,
                radius=radius_px,
            )
            return map_to_coordinates(out_map)
        return map_to_coordinates(self.element_mask_output)

    @property
    def transducer_mask(self) -> NDArray[np.bool]:
        """Return the coordinates of the transducer mask."""
        if self.radius != float("inf"):
            radius_px = round(self.radius / self.grid.dx)
            center = np.array(
                [
                    self.zero_offset / self.grid.dx - radius_px + self.position_px[0],
                    self.grid.ny // 2 + self.position_px[1],
                ],
            )
            out_map = np.zeros((self.grid.nx, self.grid.ny), dtype=bool)
            out_map[make_circle_idx(out_map.shape, center, radius_px)] = True
            return out_map
        return self.element_mask_output

    @property
    def n_sources(self) -> int:
        """Return the total number of source pixels across all elements."""
        return len(self._source_coords)

    @property
    def n_sources_per_element(self) -> int:
        """Return the number of source pixels per element."""
        return len(self._source_coords) // self.number_elements

    def __str__(self) -> str:
        """Return string representation of the TransducerGeometry.

        Returns
        -------
        str
            String representation of the TransducerGeometry.

        """
        return (
            f"TransducerGeometry:\n"
            f"  Number of elements: {self.number_elements}\n"
            f"  Element width (m): {self.element_width_m}\n"
            f"  Element height (m): {self.element_height_m}\n"
            f"  Element spacing (m): {self.element_spacing_m}\n"
            f"  Element layer (m): {self.element_layer_m}\n"
            f"  Position (m): {self.position_m}\n"
            f"  Radius (m): {self.radius}\n"
            f"  Element width (px): {self.element_width_px}\n"
            f"  Element height (px): {self.element_height_px}\n"
            f"  Element spacing (px): {self.element_spacing_px}\n"
            f"  Element layer (px): {self.element_layer_px}\n"
            f"  Position (px): {self.position_px}\n"
        )


class TransducerStack:
    """The layers a probe puts between its elements and the tissue, with their unit.

    The values are the ones an ATS-539 acquisition was matched against on a
    Verasonics. The backing sits above the element row, so the element face must
    be placed at least `backing_thickness_m` below the top of the grid.
    """

    backing_thickness_m = 1.0e-3
    backing_sound_speed_m_s = 1450.0
    backing_density_kg_m3 = 1700.0
    backing_alpha_coeff_db_cm_mhz = 20.0
    backing_face_roughness_m = 0.30e-3

    matching_thickness_m = 0.10e-3
    matching_sound_speed_m_s = 1900.0
    matching_density_kg_m3 = 975.0
    matching_alpha_coeff_db_cm_mhz = 0.5

    lens_thickness_m = 0.34e-3
    lens_sound_speed_m_s = 1450.0
    lens_density_kg_m3 = 1400.0
    lens_alpha_coeff_db_cm_mhz = 1.0

    standoff_thickness_m = 0.80e-3
    standoff_sound_speed_m_s = 1480.0
    standoff_density_kg_m3 = 1000.0
    standoff_alpha_coeff_db_cm_mhz = 0.0022
    standoff_alpha_power = 1.9

    alpha_power = 1.0
    beta = 0.0


class Transducer:
    """General transducer class.

    it connects transducer geometry with fullwave Source and Sensor implementations.
    """

    def __init__(
        self,
        transducer_geometry: TransducerGeometry,
        grid: Grid,
        input_signal: NDArray[np.float64] | None = None,
        active_source_elements: tuple[bool] | None = None,
        active_sensor_elements: tuple[bool] | None = None,
        pulse: transmit.Pulse | None = None,
        *,
        validate_input: bool = True,
        sampling_modulus_time: int = 1,
        source_type: str = source_type_module.ADDITIVE,
    ) -> None:
        """Initialize the GeneralTransducer with the provided geometry, grid, and input signal.

        Parameters
        ----------
        transducer_geometry: TransducerGeometry
            TransducerGeometry object. it defines the geometry of the transducer.
        grid: Grid
            Grid object. it defines the spatial and temporal grid.
        input_signal: NDArray
            source signal emmited by the transducer elements. it has shape (number_elements, nt)
        active_source_elements: tuple[bool] | None
            boolean array that defines which elements are active sources.
            if None, all elements are active.
        active_sensor_elements: tuple[bool] | None
            boolean array that defines which elements are active sensors.
            if None, all elements are active.
        validate_input: bool, optional
            Flag indicating whether to validate the input data.
            default is True.
        pulse: transmit.Pulse | None
            The excitation one element emits. The transmit methods take it when
            they are called without one. None takes the default excitation.
        sampling_modulus_time: int
            Sampling modulus in time. Default is 1 (record at every time step).
            Changing this value to n will record the pressure every n time steps.
            It reduces the size of the output data.
        source_type: str
            "additive" adds the signal to the field and is the default. The signal
            is scaled so the aperture radiates the pressure it is given, at the
            grid's reference sound speed. "clamped" is a hard source, which
            assigns the signal to the pressure of each source pixel instead.

        """
        if validate_input:
            check_functions.check_instance(transducer_geometry, TransducerGeometry)
            check_functions.check_instance(grid, Grid)

        self.transducer_geometry = transducer_geometry
        self.grid = grid
        self.is_3d = grid.is_3d

        if active_source_elements is None:
            active_source_elements = np.ones(transducer_geometry.number_elements, dtype=bool)
        self.active_source_elements = np.array(active_source_elements)

        if active_sensor_elements is None:
            active_sensor_elements = np.ones(transducer_geometry.number_elements, dtype=bool)
        self.active_sensor_elements = active_sensor_elements

        self.sampling_modulus_time = sampling_modulus_time
        self.pulse = transmit.Pulse() if pulse is None else pulse
        if source_type not in source_type_module.SOURCE_TYPES:
            error_msg = (
                f"source_type {source_type!r} is not one of {source_type_module.SOURCE_TYPES}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        self.source_type = source_type

        if input_signal is not None:
            self._check_signal(input_signal)
            self._signal: NDArray[np.float64] | None = input_signal
        else:
            self._signal = None

    @classmethod
    def l7_4(
        cls,
        grid: Grid,
        position_m: tuple[float, ...] | None = None,
        *,
        element_layer_px: int = 2,
        sampling_modulus_time: int = 1,
        source_type: str = source_type_module.ADDITIVE,
    ) -> "Transducer":
        """Return an ATL Philips L7-4 linear array on one grid.

        128 elements of 0.298 mm pitch and 0.048 mm kerf at 5.208 MHz. The
        excitation is the one an ATS-539 acquisition was matched against. The
        elevation aperture of 7.5 mm and the elevation focus of 25 mm have no
        datasheet behind them.

        Parameters
        ----------
        grid : Grid
            The grid the array sits on.
        position_m : tuple[float, ...] | None
            Where the corner of the aperture sits [m]. None centers it across
            the grid with the face at the top.
        element_layer_px : int
            How many grid rows one element covers.
        sampling_modulus_time : int
            How many time steps separate two recorded samples.
        source_type : str
            "additive" adds the signal to the field, which is the default and what
            the calibrated setups use. "clamped" is a hard source and assigns it.

        Returns
        -------
        Transducer
            The array, with no excitation set.

        """
        return cls._preset(
            grid,
            position_m,
            number_elements=128,
            pitch_m=0.298e-3,
            kerf_m=0.048e-3,
            element_height_m=7.5e-3,
            element_layer_px=element_layer_px,
            pulse=transmit.Pulse(pressure=3.162e5, cycles=1.0, drop_off=2.0),
            sampling_modulus_time=sampling_modulus_time,
            source_type=source_type,
        )

    @classmethod
    def c5_2v(
        cls,
        grid: Grid,
        position_m: tuple[float, ...] | None = None,
        *,
        element_layer_px: int | None = None,
        sampling_modulus_time: int = 1,
        source_type: str = source_type_module.ADDITIVE,
    ) -> "Transducer":
        """Return a curved array of 49.57 mm radius on one grid.

        128 elements of 0.508 mm pitch at 3.7 MHz, as the convex examples of this
        package uses it. Three dimensional curved arrays are not supported.

        Parameters
        ----------
        grid : Grid
            The grid the array sits on.
        position_m : tuple[float, ...] | None
            Where the corner of the aperture sits [m]. None centers it.
        element_layer_px : int | None
            How many grid rows one element covers. None takes three wavelengths,
            which is what an arc needs to be represented on a grid.
        sampling_modulus_time : int
            How many time steps separate two recorded samples.
        source_type : str
            "additive" adds the signal to the field, which is the default and what
            the calibrated setups use. "clamped" is a hard source and assigns it.

        Returns
        -------
        Transducer
            The array, with no excitation set.

        """
        return cls._preset(
            grid,
            position_m,
            number_elements=128,
            pitch_m=0.508e-3,
            kerf_m=0.0,
            radius_m=49.57e-3,
            element_layer_px=round(grid.ppw * 3) if element_layer_px is None else element_layer_px,
            pulse=transmit.Pulse(),
            sampling_modulus_time=sampling_modulus_time,
            source_type=source_type,
        )

    @classmethod
    def p4_1c(
        cls,
        grid: Grid,
        position_m: tuple[float, ...] | None = None,
        *,
        element_layer_px: int = 4,
        sampling_modulus_time: int = 1,
        source_type: str = source_type_module.ADDITIVE,
    ) -> "Transducer":
        """Return a 64 element phased array of 27 mm aperture on one grid.

        Parameters
        ----------
        grid : Grid
            The grid the array sits on.
        position_m : tuple[float, ...] | None
            Where the corner of the aperture sits [m]. None centers it.
        element_layer_px : int
            How many grid rows one element covers.
        sampling_modulus_time : int
            How many time steps separate two recorded samples.
        source_type : str
            "additive" adds the signal to the field, which is the default and what
            the calibrated setups use. "clamped" is a hard source and assigns it.

        Returns
        -------
        Transducer
            The array, with no excitation set.

        """
        pitch = 27.0e-3 / 64
        return cls._preset(
            grid,
            position_m,
            number_elements=64,
            pitch_m=pitch,
            kerf_m=pitch * 0.2,
            element_layer_px=element_layer_px,
            pulse=transmit.Pulse(),
            sampling_modulus_time=sampling_modulus_time,
            source_type=source_type,
        )

    @classmethod
    def _preset(
        cls,
        grid: Grid,
        position_m: tuple[float, ...] | None,
        *,
        number_elements: int,
        pitch_m: float,
        kerf_m: float,
        element_layer_px: int,
        pulse: transmit.Pulse,
        sampling_modulus_time: int,
        source_type: str,
        radius_m: float = float("inf"),
        element_height_m: float | None = None,
    ) -> "Transducer":
        """Return one named array, placed and ready for an excitation."""
        aperture_m = number_elements * pitch_m
        curved = radius_m != float("inf")
        if position_m is None:
            # A curved array is placed from the centre of its own arc, so the
            # geometry centres it and the offset stays at zero.
            lateral = 0.0 if curved else (grid.domain_size[1] - aperture_m) / 2.0
            position_m = (
                (0.0, lateral, (grid.domain_size[2] - (element_height_m or 0.0)) / 2.0)
                if grid.is_3d
                else (0.0, lateral)
            )
        geometry = TransducerGeometry(
            grid,
            number_elements=number_elements,
            element_width_m=0.0 if curved else pitch_m - kerf_m,
            element_height_m=element_height_m if grid.is_3d else None,
            element_spacing_m=pitch_m if curved else kerf_m,
            element_layer_px=element_layer_px,
            position_m=position_m,
            radius=radius_m,
        )
        return cls(
            transducer_geometry=geometry,
            grid=grid,
            pulse=pulse,
            sampling_modulus_time=sampling_modulus_time,
            source_type=source_type,
        )

    def plane_wave(
        self,
        angle_deg: float = 0.0,
        *,
        elevation_angle_deg: float = 0.0,
        pulse: transmit.Pulse | None = None,
        apodization: float = 0.0,
    ) -> None:
        """Set the excitation to one steered plane wave.

        Parameters
        ----------
        angle_deg : float
            The steering angle in the lateral plane [degrees].
        elevation_angle_deg : float
            The steering angle in the elevation plane [degrees]. Three dimensions only.
        pulse : transmit.Pulse | None
            The excitation one element emits. None takes the transducer's own.
        apodization : float
            The Tukey taper across the aperture. 0 gives no taper.

        Returns
        -------
        None

        """
        delays = transmit.plane_wave_delays(
            self._element_centers_m(),
            angle_deg,
            float(self.grid.c0),
            elevation_angle_deg,
        )
        self._set_transmit(delays, pulse, apodization)

    def focus(
        self,
        focus_m: tuple[float, ...],
        *,
        pulse: transmit.Pulse | None = None,
        apodization: float = 0.0,
    ) -> None:
        """Set the excitation to a focus at one point.

        Parameters
        ----------
        focus_m : tuple[float, ...]
            Where the transmit focuses [m], as (axial, lateral) or
            (axial, lateral, elevation).
        pulse : transmit.Pulse | None
            The excitation one element emits. None takes the transducer's own.
        apodization : float
            The Tukey taper across the aperture. 0 gives no taper.

        Returns
        -------
        None

        """
        delays = transmit.focused_delays(
            self._element_centers_m(),
            np.asarray(focus_m, dtype=float),
            float(self.grid.c0),
        )
        self._set_transmit(delays, pulse, apodization)

    def diverging(
        self,
        virtual_source_m: tuple[float, ...],
        *,
        pulse: transmit.Pulse | None = None,
        apodization: float = 0.0,
    ) -> None:
        """Set the excitation to a wave spreading from a virtual source.

        A virtual source behind the aperture gives a wide beam in one transmit.

        Parameters
        ----------
        virtual_source_m : tuple[float, ...]
            Where the wave appears to start [m]. A negative axial value sits
            behind the aperture.
        pulse : transmit.Pulse | None
            The excitation one element emits. None takes the transducer's own.
        apodization : float
            The Tukey taper across the aperture. 0 gives no taper.

        Returns
        -------
        None

        """
        delays = transmit.diverging_delays(
            self._element_centers_m(),
            np.asarray(virtual_source_m, dtype=float),
            float(self.grid.c0),
        )
        self._set_transmit(delays, pulse, apodization)

    def synthetic_aperture(
        self,
        element: int,
        *,
        pulse: transmit.Pulse | None = None,
    ) -> None:
        """Set the excitation to one element alone.

        The other elements stay silent, which is one transmit of a full synthetic
        aperture acquisition. Every element still records.

        Parameters
        ----------
        element : int
            Which element fires, counted from one.
        pulse : transmit.Pulse | None
            The excitation one element emits. None takes the transducer's own.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the element is outside the aperture.

        """
        elements = self.transducer_geometry.number_elements
        if not 1 <= element <= elements:
            error_msg = f"element {element} is outside the aperture of {elements} elements"
            logger.error(error_msg)
            raise ValueError(error_msg)
        weights = np.zeros(elements)
        weights[element - 1] = 1.0
        self._set_transmit(np.zeros(elements), pulse, weights=weights)

    def apply_transducer_stack(
        self,
        sound_speed: NDArray[np.float64],
        density: NDArray[np.float64],
        alpha_coeff: NDArray[np.float64],
        alpha_power: NDArray[np.float64],
        beta: NDArray[np.float64],
        scatterer: NDArray[np.float64] | None = None,
        stack: type[TransducerStack] = TransducerStack,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Paint the layers of the probe onto the medium maps, in place.

        In order from the top of the grid: the backing above the element row,
        then the matching layer, the lens and the coupling standoff below it. The
        element rows are left alone, so the source still radiates from them. The
        wave then reverberates between the probe face and the tissue, which a
        medium of tissue alone cannot produce.

        Call this on the raw maps before the Medium is built. A curved array is
        refused, because the arc of its own elements is its backing.

        Parameters
        ----------
        sound_speed : NDArray[np.float64]
            The sound speed map [m/s], changed in place.
        density : NDArray[np.float64]
            The density map [kg/m^3], changed in place.
        alpha_coeff : NDArray[np.float64]
            The attenuation coefficient map [dB/(MHz^y cm)], changed in place.
        alpha_power : NDArray[np.float64]
            The attenuation power map [-], changed in place.
        beta : NDArray[np.float64]
            The nonlinearity map [-], changed in place.
        scatterer : NDArray[np.float64] | None
            The scatter map. When given, the backing keeps the tissue speckle.
        stack : type[TransducerStack]
            The layers to paint. The default is the calibrated probe.
        rng : np.random.Generator | None
            The generator the rough backing face uses. None takes a fresh one.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If the array is curved, or if the element face sits too shallow for
            the backing.

        """
        if self.transducer_geometry.radius != float("inf"):
            error_msg = (
                "a curved array is its own backing, because its elements fill an arc "
                "and the region behind them is the probe. This paints a flat stack "
                "above one row, which would cut through that arc."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        rng = np.random.default_rng() if rng is None else rng
        step_m = float(self.grid.dx)
        face_row = int(np.asarray(self.transducer_geometry._source_coords)[:, 0].min())
        backing_rows = round(stack.backing_thickness_m / step_m)
        if face_row < backing_rows:
            error_msg = (
                f"the element face sits at row {face_row} and the backing needs "
                f"{backing_rows} rows above it. Place the transducer at least "
                f"{stack.backing_thickness_m * 1e3:.2f} mm below the top of the grid."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        maps = (sound_speed, density, alpha_coeff, alpha_power, beta)
        rows = np.arange(sound_speed.shape[0])[:, None]
        lateral = sound_speed.shape[1]

        rough_rows = round(stack.backing_face_roughness_m / step_m)
        recess = rng.integers(0, rough_rows + 1, size=lateral)[None, :] if rough_rows else 0
        backing = np.broadcast_to(rows < (face_row - recess), sound_speed.shape)
        self._paint(
            maps,
            backing,
            stack.backing_sound_speed_m_s,
            stack.backing_density_kg_m3,
            stack.backing_alpha_coeff_db_cm_mhz,
            stack.alpha_power,
            stack.beta,
            scatterer,
        )

        below = face_row + self.transducer_geometry.element_layer_px
        for thickness_m, speed, rho, alpha, power in (
            (
                stack.matching_thickness_m,
                stack.matching_sound_speed_m_s,
                stack.matching_density_kg_m3,
                stack.matching_alpha_coeff_db_cm_mhz,
                stack.alpha_power,
            ),
            (
                stack.lens_thickness_m,
                stack.lens_sound_speed_m_s,
                stack.lens_density_kg_m3,
                stack.lens_alpha_coeff_db_cm_mhz,
                stack.alpha_power,
            ),
            (
                stack.standoff_thickness_m,
                stack.standoff_sound_speed_m_s,
                stack.standoff_density_kg_m3,
                stack.standoff_alpha_coeff_db_cm_mhz,
                stack.standoff_alpha_power,
            ),
        ):
            end = min(sound_speed.shape[0], below + round(thickness_m / step_m))
            if end > below:
                layer = np.broadcast_to((rows >= below) & (rows < end), sound_speed.shape)
                self._paint(maps, layer, speed, rho, alpha, power, stack.beta, None)
            below = end

    @staticmethod
    def _paint(
        maps: tuple[NDArray[np.float64], ...],
        where: NDArray[np.bool_],
        speed: float,
        rho: float,
        alpha: float,
        power: float,
        nonlinearity: float,
        scatterer: NDArray[np.float64] | None,
    ) -> None:
        """Write one layer into the medium maps."""
        sound_speed, density, alpha_coeff, alpha_power, beta = maps
        sound_speed[where] = speed
        alpha_coeff[where] = alpha
        alpha_power[where] = power
        beta[where] = nonlinearity
        density[where] = rho if scatterer is None else (rho * np.asarray(scatterer))[where]

    def _element_centers_m(self) -> NDArray[np.float64]:
        """Return the center of each element, in meters."""
        return transmit.element_centers_m(self.transducer_geometry, float(self.grid.dx))

    def _set_transmit(
        self,
        delays_s: NDArray[np.float64],
        pulse: transmit.Pulse | None,
        apodization: float = 0.0,
        weights: NDArray[np.float64] | None = None,
    ) -> None:
        """Build the excitation from one delay and one weight for each element."""
        elements = self.transducer_geometry.number_elements
        if weights is None:
            weights = transmit.tukey_weights(elements, apodization)
        active = np.where(self.active_source_elements)[0] + 1
        chosen = np.isin(self.transducer_geometry._source_ids, active)
        self.set_signal(
            transmit.signal_of(
                self.grid,
                self.transducer_geometry._source_coords[chosen],
                self.transducer_geometry._source_ids[chosen],
                self.pulse if pulse is None else pulse,
                delays_s,
                weights,
            )
        )

    def _check_signal(self, signal: NDArray[np.float64]) -> None:
        if signal.shape[1] != self.grid.nt:
            error_msg = "Input signal has the wrong number of time points"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if (signal == 0).all():
            error_msg = "Input signal is all zeros"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if signal.shape[0] != len(self.source_coords):
            error_msg = "Input signal has the wrong number of elements"
            logger.error(error_msg)
            raise ValueError(error_msg)

    @property
    def signal(self) -> NDArray[np.float64]:
        """Return the input signal.

        Raises
        ------
        ValueError
            If the signal is not set.

        """
        if self._signal is None:
            error_msg = "Input signal is not set. use set_signal() to set the signal."
            logger.error(error_msg)
            raise ValueError(error_msg)
        return self._signal

    @signal.setter
    def signal(self, value: NDArray[np.float64]) -> None:
        self._check_signal(value)
        self._signal = value

    def set_signal(self, value: NDArray[np.float64]) -> None:
        """Set the input signal.

        This method is used to set the input signal for the transducer.

        Parameters
        ----------
        value : NDArray[np.float64]
            The input signal to be set.

        """
        self.signal = value

    @property
    def source_coords(self) -> NDArray[np.int64]:
        """Coordinates of active source pixels; shape [N_src, ndim]."""
        active_ids = np.where(self.active_source_elements)[0] + 1
        mask = np.isin(self.transducer_geometry._source_ids, active_ids)
        return self.transducer_geometry._source_coords[mask]

    @property
    def sensor_coords(self) -> NDArray[np.int64]:
        """Coordinates of active sensor pixels; shape [N_snsr, ndim]."""
        active_ids = np.where(self.active_sensor_elements)[0] + 1
        mask = np.isin(self.transducer_geometry._sensor_ids, active_ids)
        return self.transducer_geometry._sensor_coords[mask]

    @property
    def element_id_to_element_surface(self) -> dict[int, NDArray[np.int64]]:
        """Return the dictionary mapping source elements to their center coordinates."""
        out_dict = {}
        indexed_element_surface = (
            np.roll(
                self.transducer_geometry.indexed_element_mask_input,
                shift=1,
                axis=0,
            )
            * self.sensor_mask
        )
        if indexed_element_surface.sum() == 0:
            indexed_element_surface = (
                self.transducer_geometry.indexed_element_mask_output * self.sensor_mask
            )

        for i in range(1, self.transducer_geometry.number_elements + 1):
            indexed_element_mask_list = np.stack(
                np.where(
                    indexed_element_surface == i,
                ),
            )
            out_dict[i] = indexed_element_mask_list.T
        return out_dict

    @property
    def element_surface_to_elemnt_id(self) -> dict[tuple[int, int], int]:
        """Return a mapping from element center coordinates to element IDs."""
        out_dict = {}
        for element_id, surface_coords in self.element_id_to_element_surface.items():
            for coord in surface_coords:
                out_dict[coord[0], coord[1]] = element_id
        return out_dict

    def post_process_sensor_output(
        self,
        sensor_output: NDArray[np.float32],
        *,
        average_surface_signals: bool = True,
    ) -> NDArray[np.float32]:
        """Sort the sensor output based on element ID order.

        Parameters
        ----------
        sensor_output
            The raw sensor output data.
        average_surface_signals: If True, average the sensor output
            over the entire element surface.
            If False, only the center pixel of each element is used.

        Returns
        -------
        NDArray[np.float32]
            The sorted sensor output data.

        """
        sensor_coordinates = self.sensor.outcoords

        # sort the sensor_output with element ID order
        n_elements = len(np.where(self.active_sensor_elements)[0])
        sorted_sensor_output = np.zeros(
            (n_elements, sensor_output.shape[1]),
            dtype=sensor_output.dtype,
        )
        element_id_to_element_surface = self.element_id_to_element_surface

        if not average_surface_signals:
            for element_index, element_id in enumerate(
                np.where(self.active_sensor_elements)[0] + 1,
            ):
                surface_coords = element_id_to_element_surface[element_id]
                center_coords = surface_coords[surface_coords.shape[0] // 2]
                sorted_sensor_output[element_index, :] = sensor_output[
                    np.where(
                        (sensor_coordinates[:, 0] == center_coords[0])
                        & (sensor_coordinates[:, 1] == center_coords[1]),
                    )[0][0],
                    :,
                ].copy()
            return sorted_sensor_output

        for element_index, element_id in enumerate(
            np.where(self.active_sensor_elements)[0] + 1,
        ):
            num_added = 0
            surface_coords = element_id_to_element_surface[element_id]
            for sensor_coord in surface_coords:
                # find the sensor index
                sensor_indices = np.where(
                    (sensor_coordinates[:, 0] == sensor_coord[0])
                    & (sensor_coordinates[:, 1] == sensor_coord[1]),
                )[0]
                for sensor_index in sensor_indices:
                    sorted_sensor_output[element_index, :] += sensor_output[sensor_index, :].copy()
                    num_added += 1
            sorted_sensor_output[element_index, :] /= num_added
        return sorted_sensor_output

    @property
    def transducer_surface(self) -> NDArray[np.int64]:
        """Return the coordinates of the transducer surface."""
        return self.transducer_geometry.transducer_surface

    @property
    def transducer_mask(self) -> NDArray[np.bool]:
        """Return the coordinates of the transducer mask."""
        return self.transducer_geometry.transducer_mask

    def make_suraface_reflective_with_air(self) -> NDArray[np.bool]:
        """Make the transducer surface reflective with air.

        Returns
        -------
        NDArray[np.bool]
            The air map for the transducer surface reflection
        .

        """
        air_map = np.zeros(self.grid.shape, dtype=bool)
        transducer_surface = self.transducer_surface
        element_thickness = self.transducer_geometry.element_layer_px
        for coord in transducer_surface.T:
            air_start = max(coord[0] - element_thickness, 0)
            air_map[
                air_start : coord[0],
                coord[1],
            ] = True

        indexed_element_mask_input = self.transducer_geometry.indexed_element_mask_input
        air_map[indexed_element_mask_input > 0] = False

        return air_map

    @property
    def sensor_mask(self) -> NDArray[np.bool_]:
        """Boolean mask of active sensor pixels."""
        mask = np.zeros(self.transducer_geometry.stored_grid_size, dtype=bool)
        coords = self.sensor_coords
        if len(coords):
            idx = tuple(coords[:, i] for i in range(coords.shape[1]))
            mask[idx] = True
        return mask

    @property
    def source_mask(self) -> NDArray[np.bool_]:
        """Boolean mask of active source pixels."""
        mask = np.zeros(self.transducer_geometry.stored_grid_size, dtype=bool)
        coords = self.source_coords
        if len(coords):
            idx = tuple(coords[:, i] for i in range(coords.shape[1]))
            mask[idx] = True
        return mask

    @property
    def dict_source_index_to_location(self) -> dict[int, NDArray[np.int64]]:
        """Return mapping from 1-based source pixel index to grid coordinates."""
        coords = self.source_coords
        return {i: coords[i - 1] for i in range(1, len(coords) + 1)}

    @property
    def element_id_to_element_center(self) -> dict[int, NDArray[np.int64]]:
        """Return the dictionary mapping element IDs to their center coordinates."""
        coords = self.transducer_geometry._source_coords
        ids = self.transducer_geometry._source_ids
        n = self.transducer_geometry.number_elements
        ndim = coords.shape[1] if len(coords) else (3 if self.is_3d else 2)
        out: dict[int, NDArray[np.int64]] = {}
        for eid in range(1, n + 1):
            sel = ids == eid
            if not sel.any():
                out[eid] = np.full(ndim, -1, dtype=np.int64)
            else:
                out[eid] = np.rint(coords[sel].mean(axis=0)).astype(np.int64)
        return out

    @property
    def sensor(self) -> fullwave.sensor.Sensor:
        """Return the Sensor object with active sensor coordinates.

        this property is used in the fullwave simulation run.
        """
        return fullwave.sensor.Sensor(
            coords=self.sensor_coords,
            grid_shape=tuple(self.transducer_geometry.stored_grid_size),
            sampling_modulus_time=self.sampling_modulus_time,
        )

    @property
    def source(self) -> fullwave.source.Source:
        """Return the Source object with active source coordinates and signal.

        this property is used in the fullwave simulation run.

        Raises
        ------
        ValueError
            If the signal is not set.

        """
        if self._signal is None:
            error_msg = "Input signal is not set. use set_signal() to set the signal."
            logger.error(error_msg)
            raise ValueError(error_msg)
        grid_shape = tuple(self.transducer_geometry.stored_grid_size)
        if source_type_module.is_additive(self.source_type):
            scale = source_type_module.additive_signal_scale(
                float(self.grid.c0), float(self.grid.dt), float(self.grid.dx)
            )
            additive = fullwave.source.Source(
                p0_additive=self.signal * scale,
                coords_additive=self.source_coords,
                grid_shape=grid_shape,
            )
            additive.additive_signal_is_scaled = True
            return additive
        return fullwave.source.Source(
            p0=self.signal,
            coords=self.source_coords,
            grid_shape=grid_shape,
        )

    @property
    def n_sources(self) -> int:
        """Return the total number of source pixels across all elements."""
        return self.transducer_geometry.n_sources

    def plot_source_mask(
        self,
        export_path: Path | str | None = Path("./temp/temp.png"),
        dpi: int = 300,
        *,
        show: bool = False,
    ) -> None:
        """Plot everything.

        it plots whole transducer geometry including the inactive/active source and sensor elements.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot_mask = np.zeros(self.transducer_geometry.stored_grid_size)
        plot_mask[self.transducer_geometry.element_mask_input] = 1
        plot_mask[self.source_mask] = 2
        pcm = ax.imshow(plot_mask, cmap="turbo")
        ax.set_title("Source Mask layout")
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        ax.set_aspect("equal")
        ax.set_xlim(0 - 10, self.grid.ny + 10)
        ax.set_ylim(0 - 10, self.grid.nx + 10)
        ax.invert_yaxis()
        cbar = fig.colorbar(
            pcm,
            ax=ax,
            label="Element Type",
        )
        cbar.set_ticks(
            ticks=[0, 1, 2],
            labels=["background", "inactive", "active"],
        )
        if export_path is not None:
            plt.savefig(export_path, dpi=dpi)
        if show:
            plt.show()
        plt.close()

    def plot_sensor_mask(
        self,
        export_path: Path | str | None = Path("./temp/temp.png"),
        dpi: int = 300,
        *,
        show: bool = False,
    ) -> None:
        """Plot everything.

        it plots whole transducer geometry including the inactive/active source and sensor elements.
        """
        import matplotlib.pyplot as plt  # noqa: PLC0415

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot_mask = np.zeros(self.transducer_geometry.stored_grid_size)
        plot_mask[self.transducer_geometry.element_mask_input] = 1
        plot_mask[self.sensor_mask] = 2
        pcm = ax.imshow(plot_mask, cmap="turbo")
        ax.set_title("Sensor Mask layout")
        ax.set_xlabel("y")
        ax.set_ylabel("x")
        ax.set_aspect("equal")
        ax.set_xlim(0 - 10, self.grid.ny + 10)
        ax.set_ylim(0 - 10, self.grid.nx + 10)
        ax.invert_yaxis()
        cbar = fig.colorbar(
            pcm,
            ax=ax,
            label="Element Type",
        )
        cbar.set_ticks(
            ticks=[0, 1, 2],
            labels=["background", "inactive", "active"],
        )
        if export_path is not None:
            plt.savefig(export_path, dpi=dpi)
        if show:
            plt.show()
        plt.close()

    def print_info(self) -> None:
        """Print information about the Transducer object."""
        print(str(self))

    def summary(self) -> None:
        """Alias for print_info."""
        self.print_info()

    def __str__(self) -> str:
        """Return a string representation of the Transducer object.

        Returns
        -------
        str
            A string representation of the Transducer object.

        """
        return (
            f"Transducer with {self.transducer_geometry.number_elements} elements\n"
            f"Element width (m): {self.transducer_geometry.element_width_m}\n"
            f"Element spacing (m): {self.transducer_geometry.element_spacing_m}\n"
            f"Transducer width (m): {self.transducer_geometry.transducer_width_m}\n"
            f"Position (m): {self.transducer_geometry.position_m}\n"
            f"Active source elements: {self.active_source_elements}\n"
            f"Active sensor elements: {self.active_sensor_elements}\n"
            f"Input signal shape: {self._signal.shape if self._signal is not None else None}\n"
        )

    def __repr__(self) -> str:
        """Return a string representation of the Transducer object.

        Returns
        -------
        str
            A string representation of the Transducer object.

        """
        return self.__str__()


class LinearTransducer(Transducer):
    """Linear transducer class.

    it implements a linear array transducer for fullwave simulations.
    """

    def __init__(
        self,
        grid: Grid,
        position_m: tuple[float, float] | tuple[float, float, float],
        active_source_elements: tuple[bool] | None = None,
        active_sensor_elements: tuple[bool] | None = None,
    ) -> None:
        """Initialize a LinearTransducer instance.

        Parameters
        ----------
        grid : Grid
            Grid object defining the spatial and temporal grid.
        position_m : tuple[float, float] | tuple[float, float, float])
            Position of the transducer in meters.
        input_signal : (NDArray[np.float64])
            Input signal emitted by the transducer.
        active_source_elements : (tuple[bool] | None)
            Flags indicating active source elements.
        active_sensor_elements : (tuple[bool] | None)
            Flags indicating active sensor elements.

        """
        transducer_geometry = TransducerGeometry(
            grid=grid,
            number_elements=128,
            element_width_m=1.459375e-4,  # 1.459375e-4 [m] = 0.1459375 [mm]
            element_spacing_m=1.459375e-4,  # 1.459375e-4 [m] = 0.1459375 [mm]
            position_m=position_m,
        )
        input_signal = np.ones((transducer_geometry.number_elements, grid.nt))
        super().__init__(
            transducer_geometry=transducer_geometry,
            grid=grid,
            input_signal=input_signal,
            active_source_elements=active_source_elements,
            active_sensor_elements=active_sensor_elements,
        )


def make_p4_1c_trasnducer(
    grid: Grid,
    position_m: tuple[float, float] | None = (0.0, 0.0),
    position_px: tuple[int, int] | None = None,
) -> Transducer:
    """Create a P4.1C transducer.

    Parameters
    ----------
    Args:
    grid : Grid
        Grid object defining the spatial and temporal grid.
    position_m : tuple[float, float] | tuple[float, float, float])
        Position of the transducer in meters.
    position_px : tuple[int, int] | None
        Position of the transducer in pixels. If None, it will be calculated from position_m.

    Returns
    -------
    Transducer
        A Transducer object representing the P4.1C transducer.

    """
    transducer_width_m = 27e-3
    element_layer_px = 4
    transducer_geometry = fullwave.TransducerGeometry(
        grid,
        number_elements=64,
        # -
        element_width_m=transducer_width_m / 64 * 0.8,
        # -
        element_spacing_m=transducer_width_m / 64 * 0.2,
        # -
        element_layer_px=element_layer_px,
        # -
        # [axial, lateral]
        position_m=position_m,
        position_px=position_px,
        # -
        radius=float("inf"),
    )
    return fullwave.Transducer(
        transducer_geometry=transducer_geometry,
        grid=grid,
    )
