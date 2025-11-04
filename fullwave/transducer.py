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
from fullwave.grid import Grid
from fullwave.utils import check_functions
from fullwave.utils.coordinates import make_circle_idx, map_to_coordinates, map_to_coords_with_sort

logger = logging.getLogger("__main__." + __name__)


def _make_pos_int(val: float | tuple[float] | tuple[int]) -> NDArray[np.int64]:
    """Force value to be a positive integer.

    Returns:
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

        Raises
        ------
        ValueError
            If neither pixel nor meter dimensions are provided
            or if the transducer exceeds grid bounds.

        """
        if validate_input:
            check_functions.check_instance(grid, Grid)
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
        # if (self.position_px[2] + self.element_length_px) > self.stored_grid_size[2]:
        #     logger.debug(self.position_px[2])
        #     logger.debug(self.element_length_px)
        #     logger.debug(self.stored_grid_size[2])
        #     error_msg = (
        #         "The defined transducer is too large or"
        #         " positioned outside the grid in the z-direction"
        #     )
        #     raise ValueError(
        #         error_msg,
        #     )
        if self.position_px[0] > self.stored_grid_size[0]:
            error_msg = "The defined transducer is positioned outside the grid in the x-direction"
            logger.error(error_msg)
            raise ValueError(error_msg)

        # create the transducer mask
        self.indexed_element_mask_input, self.indexed_element_mask_output = (
            self._create_element_mask()
        )
        self.element_mask_input = self.indexed_element_mask_input > 0
        self.element_mask_output = self.indexed_element_mask_output > 0
        logger.debug("TransducerGeometry instance created.")

    def _init_dimensions(  # noqa: C901, PLR0912
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

    def _create_element_mask(self) -> tuple[NDArray[np.int64], ...]:
        indexed_element_mask_input = np.zeros(self.stored_grid_size, dtype=int)
        indexed_element_mask_output = np.zeros(self.stored_grid_size, dtype=int)
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
                    indexed_element_mask_input[
                        element_pos_x : element_pos_x + self.element_layer_px,
                        element_pos_y : element_pos_y + self.element_width_px,
                        element_pos_z : element_pos_z + self.element_height_px,
                    ] = element_index + 1
                    indexed_element_mask_output[
                        element_pos_x + self.element_layer_px - 1,
                        element_pos_y + self.element_width_px - 1,
                        element_pos_z + self.element_height_px // 2 - 1,
                    ] = element_index + 1
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
                    indexed_element_mask_input[
                        element_pos_x : element_pos_x + self.element_layer_px,
                        element_pos_y : element_pos_y + self.element_width_px,
                    ] = element_index + 1
                    indexed_element_mask_output[
                        element_pos_x + self.element_layer_px - 1,
                        element_pos_y + self.element_width_px // 2 - 1,
                    ] = element_index + 1
        elif self.is_3d:
            error_msg = "3D convex transducers are not implemented yet."
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
        else:
            radius_px = round(self.radius / self.grid.dx)
            d_theta = np.arctan2(self.element_spacing_m / self.grid.dy, radius_px)
            theta_list = self._define_theta_at_center(
                d_theta=d_theta,
                num_elements=self.number_elements,
            )
            center = np.array(
                [
                    self.zero_offset / self.grid.dx - radius_px,
                    self.grid.ny // 2,
                ],
            )
            in_map = self._calculate_inmap(
                center=center,
                radius=radius_px,
            )
            out_map = self._calculate_outmap(
                center=center,
                radius=radius_px,
            )

            in_coords = map_to_coords_with_sort(in_map)
            out_coords = map_to_coords_with_sort(out_map)
            in_coords, out_coords = self._assign_transducer_num_to_input(
                in_coords=in_coords,
                out_coords=out_coords,
                center=center,
                number_elements=self.number_elements,
                d_theta=d_theta,
                theta_list=theta_list,
            )
            indexed_element_mask_input = self._coords_to_index_map(
                in_coords,
                grid_shape=self.stored_grid_size,
            )
            indexed_element_mask_output = self._coords_to_index_map(
                out_coords,
                grid_shape=self.stored_grid_size,
            )

        return indexed_element_mask_input, indexed_element_mask_output

    @staticmethod
    def _coords_to_index_map(
        coords: NDArray[np.float64],
        grid_shape: NDArray[np.float64],
    ) -> NDArray[np.int64]:
        indexed_element_mask = np.zeros(grid_shape, dtype=int)
        for coord in coords:
            x = coord[0]
            y = coord[1]
            index = coord[3]
            indexed_element_mask[x.astype(int), y.astype(int)] = index
        return indexed_element_mask

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

        # out_coords2 = np.zeros((number_elements, 2))
        # in_coords2 = np.zeros((number_elements, 2))

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
            # in_coords2[tt, 0] = np.mean(in_coords[id_theta, 0])
            # in_coords2[tt, 1] = np.mean(in_coords[id_theta, 1])

            # find which outcoords are assigned to tt
            less_than_max = thetas_out < (theta_list[tt] + d_theta / 2)
            greater_than_min = thetas_out > (theta_list[tt] - d_theta / 2)
            id_theta = np.where(np.logical_and(less_than_max, greater_than_min))[0]
            out_coords[id_theta, 3] = tt + 1
            # out_coords2[tt, 0] = np.mean(out_coords[id_theta, 0])
            # out_coords2[tt, 1] = np.mean(out_coords[id_theta, 1])
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

    def _calculate_outmap(self, center: NDArray[np.float64], radius: float) -> np.ndarray:
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

            output_map[j - 1, i] = 1

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
    def n_sources(self) -> NDArray[np.int64]:
        """Return the number of source elements."""
        return self.element_mask_input.sum()

    @property
    def n_sources_per_element(self) -> NDArray[np.int64]:
        """Return the number of source elements."""
        return self.element_mask_input.sum() // self.number_elements

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
        *,
        validate_input: bool = True,
        sampling_modulus_time: int = 1,
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
        sampling_modulus_time: int
            Sampling modulus in time. Default is 1 (record at every time step).
            Changing this value to n will record the pressure every n time steps.
            It reduces the size of the output data.

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

        if input_signal is not None:
            self._check_signal(input_signal)
            self._signal: NDArray[np.float64] | None = input_signal
        else:
            self._signal = None

    def _check_signal(self, signal: NDArray[np.float64]) -> None:
        if signal.shape[1] != self.grid.nt:
            error_msg = "Input signal has the wrong number of time points"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if (signal == 0).all():
            error_msg = "Input signal is all zeros"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if signal.shape[0] != self.source_mask.sum():
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
    def sensor_mask(self) -> NDArray[np.bool]:
        """Return the sensor mask indicating active sensor elements from the transducer geometry."""
        active_ids = np.where(self.active_sensor_elements)[0] + 1
        return np.isin(self.transducer_geometry.indexed_element_mask_output, active_ids)

    @property
    def source_mask(self) -> NDArray[np.bool]:
        """Return the source mask indicating active source elements from the transducer geometry."""
        active_ids = np.where(self.active_source_elements)[0] + 1
        return np.isin(self.transducer_geometry.indexed_element_mask_input, active_ids)

    @property
    def dict_source_index_to_location(self) -> dict[int, NDArray[np.int64]]:
        """Return the dictionary mapping source elements to their locations."""
        # get the coordinates of the active source elements
        coords = map_to_coordinates(self.source_mask, is_3d=self.is_3d, sort=True).T
        # create a dictionary mapping source elements to their coordinates
        return {i: coords[i - 1] for i in range(1, self.transducer_geometry.n_sources + 1)}

    @property
    def element_id_to_element_center(self) -> dict[int, NDArray[np.int64]]:
        """Return the dictionary mapping source elements to their center coordinates."""
        out_dict = {}
        for i in range(1, self.transducer_geometry.number_elements + 1):
            indexed_element_mask = np.stack(
                np.where(
                    self.transducer_geometry.indexed_element_mask_input == i,
                ),
            )
            center = np.round(indexed_element_mask.mean(axis=1))
            out_dict[i] = center
        return out_dict

    @property
    def sensor(self) -> fullwave.sensor.Sensor:
        """Return the Sensor object with the sensor mask.

        this property is used in the fullwave simulation run.
        """
        return fullwave.sensor.Sensor(
            self.sensor_mask,
            sampling_modulus_time=self.sampling_modulus_time,
        )

    @property
    def source(self) -> fullwave.source.Source:
        """Return the Source object with the sensor mask and signal.

        this property is used in the fullwave simulation run.

        Raises
        ------
        ValueError
            If the signal is not set.

        """
        # check if the signal is set
        if self._signal is None:
            error_msg = "Input signal is not set. use set_signal() to set the signal."
            logger.error(error_msg)
            raise ValueError(error_msg)
        return fullwave.source.Source(self.signal, self.source_mask)

    @property
    def n_sources(self) -> NDArray[np.int64]:
        """Return the number of source elements."""
        return self.transducer_geometry.n_sources

    @property
    def tranducer_surface(self) -> NDArray[np.int64]:
        """Return the coordinates of the transducer surface."""
        return map_to_coordinates(self.sensor_mask == 1)[0]

    @property
    def tranducer_mask(self) -> NDArray[np.bool]:
        """Return the coordinates of the transducer mask."""
        mask = np.zeros(self.transducer_geometry.stored_grid_size, dtype=bool)
        tranducer_surface = self.tranducer_surface
        for i in range(len(tranducer_surface)):
            mask[: tranducer_surface[i].astype(int), i] = 1
        return mask

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
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot_mask = np.zeros(self.transducer_geometry.stored_grid_size)
        plot_mask[self.transducer_geometry.element_mask_input] = 1
        # plot_mask[np.roll(self.source_mask, 10, axis=0)] = 2
        # plot_mask[np.roll(self.sensor_mask, 5, axis=0)] = 3
        plot_mask[self.source_mask] = 2
        # plot_mask[self.sensor_mask + 1] = 3
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
            # orientation="horizontal",
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
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
        plot_mask = np.zeros(self.transducer_geometry.stored_grid_size)
        plot_mask[self.transducer_geometry.element_mask_input] = 1
        # plot_mask[np.roll(self.source_mask, 10, axis=0)] = 2
        # plot_mask[np.roll(self.sensor_mask, 5, axis=0)] = 3
        plot_mask[self.sensor_mask] = 2
        # plot_mask[self.sensor_mask + 1] = 3
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
            # orientation="horizontal",
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
