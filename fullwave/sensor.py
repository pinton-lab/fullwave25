"""Source class for Fullwave."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from fullwave.utils import plot_utils
from fullwave.utils.coordinates import coords_to_index_map, coords_to_map, map_to_coords

logger = logging.getLogger("__main__." + __name__)


@dataclass
class Sensor:
    """Sensor class for Fullwave."""

    outcoords: NDArray[np.int64]
    sampling_modulus_time: int = 1

    def __init__(
        self,
        mask: NDArray[np.bool] | None = None,
        sampling_modulus_time: int = 1,
        *,
        coords: NDArray[np.int64] | None = None,
        grid_shape: tuple[int, ...] | None = None,
        mod_x: int | None = None,
        mod_y: int | None = None,
        mod_z: int = 0,
    ) -> None:
        """Sensor class for Fullwave.

        Parameters
        ----------
        mask : NDArray[np.bool] | None
            Binary matrix where the pressure is recorded at each time-step
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D.
            Mutually exclusive with coords/grid_shape and mod_x/mod_y.
        sampling_modulus_time: int
            Sampling modulus in time. Default is 1 (record at every time step).
            Changing this value to n will record the pressure every n time steps.
            It reduces the size of the output data.
        coords : NDArray[np.int64] | None
            Coordinate array of sensor positions, shape [n_sensors, ndim].
            Must be provided together with grid_shape.
        grid_shape : tuple[int, ...] | None
            Shape of the computational grid. Required when using coords input.
        mod_x : int | None
            Spatial decimation stride in x (depth) for sparse-grid sensor output.
            When provided together with mod_y, activates sparse-grid mode.
            In this mode the solver binary generates sensor positions automatically
            as [::mod_x, ::mod_y] (and [::mod_z] for 3D) across the domain,
            ignoring any explicit coords/mask.  Mutually exclusive with mask/coords.
        mod_y : int | None
            Spatial decimation stride in y (lateral).  Must be provided with mod_x.
        mod_z : int
            Spatial decimation stride in z (elevational).  Only relevant in 3D simulations.
            Providing mod_z > 0 together with mod_x and mod_y creates a 3D sparse sensor.
            Default 0 (2D sensor, or no z-subsampling when used in a 3D run).

        Raises
        ------
        ValueError
            If grid_shape is not provided when using coords input.
            If both mask and coords are provided (mutually exclusive).
            If only one of mod_x / mod_y is provided.
            If mod_x/mod_y are mixed with mask or coords.
            If none of the three input modes is supplied.

        """
        if mod_x is not None or mod_y is not None:
            # --- sparse-grid mode ---
            if mod_x is None or mod_y is None:
                msg = "Both mod_x and mod_y must be provided for sparse-grid sensor mode"
                raise ValueError(msg)
            if mask is not None or coords is not None:
                msg = "mod_x/mod_y are mutually exclusive with mask and coords"
                raise ValueError(msg)
            if mod_x <= 0 or mod_y <= 0:
                msg = "mod_x and mod_y must be positive integers"
                raise ValueError(msg)
            if mod_z < 0:
                msg = "mod_z must be a non-negative integer"
                raise ValueError(msg)
            self.is_sparse_grid = True
            self.mod_x = mod_x
            self.mod_y = mod_y
            self.mod_z = mod_z
            self.is_3d = mod_z > 0
            ndim = 3 if self.is_3d else 2
            # Empty placeholder — the binary computes positions from mod values.
            self.outcoords = np.empty((0, ndim), dtype=np.int64)
            self.grid_shape = None
        elif coords is not None:
            if grid_shape is None:
                msg = "grid_shape is required when using coords input"
                raise ValueError(msg)
            if mask is not None:
                msg = "mask and coords are mutually exclusive"
                raise ValueError(msg)
            self.is_sparse_grid = False
            self.mod_x = 0
            self.mod_y = 0
            self.mod_z = 0
            self.outcoords = np.atleast_2d(coords).astype(np.int64, copy=False)
            self.grid_shape = tuple(grid_shape)
            self.is_3d = len(self.grid_shape) == 3
        elif mask is not None:
            self.is_sparse_grid = False
            self.mod_x = 0
            self.mod_y = 0
            self.mod_z = 0
            mask = np.atleast_2d(mask)
            self.grid_shape = mask.shape
            self.outcoords = map_to_coords(mask)
            self.is_3d = len(self.grid_shape) == 3
        else:
            msg = "Either mask, coords (with grid_shape), or mod_x with mod_y must be provided"
            raise ValueError(msg)

        self.sampling_modulus_time = sampling_modulus_time
        super().__init__()
        logger.debug("Sensor instance created.")

    def validate(self, grid_shape: NDArray[np.int64] | tuple) -> None:
        """Check if the sensor coordinates are consistent with the grid shape."""
        if self.is_sparse_grid:
            grid_shape = tuple(grid_shape) if isinstance(grid_shape, np.ndarray) else grid_shape
            if len(grid_shape) == 3 and self.mod_z == 0:
                msg = (
                    "Sparse-grid sensor used in a 3D simulation but mod_z was not provided. "
                    "Pass mod_z > 0 to Sensor (e.g. Sensor(mod_x=4, mod_y=4, mod_z=4))."
                )
                raise ValueError(msg)
            logger.debug("Sparse-grid sensor validated.")
            return
        grid_shape = tuple(grid_shape) if isinstance(grid_shape, np.ndarray) else grid_shape
        assert self.grid_shape == grid_shape, f"{self.grid_shape} != {grid_shape}"
        assert self.n_sensors > 0, "No active sensor found."
        logger.debug("Sensor validated against grid shape.")

    @property
    def mask(self) -> NDArray[np.int64]:
        """Returns the source mask.

        it calculates the source mask from the source coordinates to reduce the memory usage.
        """
        return coords_to_map(
            self.outcoords,
            grid_shape=self.grid_shape,
            is_3d=self.is_3d,
        )

    @property
    def indexed_mask(self) -> NDArray[np.int64]:
        """Returns the source mask.

        it calculates the source mask from the source coordinates to reduce the memory usage.
        """
        return coords_to_index_map(
            self.outcoords,
            grid_shape=self.grid_shape,
            is_3d=self.is_3d,
        )

    @property
    def ncoordsout(self) -> int:
        """Return the number of sensors.

        ailiased to n_sensors for the compatibility with matlab version
        """
        return self.n_sensors

    @property
    def n_sensors(self) -> int:
        """Return the number of sensors."""
        return self.outcoords.shape[0]

    def plot(
        self,
        export_path: Path | str | None = Path("./temp/temp.png"),
        *,
        show: bool = False,
        dpi: int = 300,
    ) -> None:
        """Plot the transducer mask, optionally exporting and displaying the figure.

        Raises
        ------
            ValueError: If the sensor is 3D because plotting is not supported.

        """
        if self.is_3d:
            error_msg = "3D plotting is not supported yet."
            raise ValueError(error_msg)
        plt.close("all")
        fig, ax = plt.subplots()
        fig = plot_utils.plot_array(
            self.mask,
            xlim=[-10, self.mask.shape[1] + 10],
            ylim=[-10, self.mask.shape[0] + 10],
            reverse_y_axis=True,
            save=False,
            clear_all=False,
            fig=fig,
            colorbar=True,
        )
        if export_path is not None:
            plt.savefig(export_path, dpi=dpi)
        if show:
            plt.show()

        plt.close("all")

    def print_info(self) -> None:
        """Print sensor information to the logger."""
        print(str(self))

    def summary(self) -> None:
        """Alias for print_info."""
        self.print_info()

    def __str__(self) -> str:
        """Show sensor information.

        Returns
        -------
        str
            Formatted string containing source information.

        """
        if self.is_sparse_grid:
            mod_str = f"mod_x={self.mod_x}, mod_y={self.mod_y}"
            if self.is_3d:
                mod_str += f", mod_z={self.mod_z}"
            return f"Sensor (sparse-grid): \n  Strides: {mod_str}\n  Is 3D: {self.is_3d}\n"
        return (
            f"Sensor: \n"
            f"  Number of sensors: {self.n_sensors}\n"
            f"  Grid shape: {self.grid_shape}\n"
            f"  Is 3D: {self.is_3d}\n"
        )

    def __repr__(self) -> str:
        """Show sensor information.

        Returns
        -------
        str
            Formatted string containing source information.

        """
        return self.__str__()
