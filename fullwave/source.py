"""Source class for Fullwave."""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from fullwave.utils import plot_utils
from fullwave.utils.coordinates import coords_to_map, map_to_coords

logger = logging.getLogger("__main__." + __name__)


@dataclass
class Source:
    """Source class for Fullwave."""

    p0: NDArray[np.float64]
    incoords: NDArray[np.int64]
    grid_shape: tuple[int, ...]

    def __init__(
        self,
        p0: NDArray[np.float64],
        mask: NDArray[np.bool] | None = None,
        *,
        coords: NDArray[np.int64] | None = None,
        grid_shape: tuple[int, ...] | None = None,
    ) -> None:
        """Source class for Fullwave.

        Parameters
        ----------
        p0 : NDArray[np.float64]
            time varying pressure at each of the source positions given by source.p_mask
            shape: [n_sources, nt]
        mask : NDArray[np.bool] | None
            binary matrix specifying the positions of the time varying pressure source distribution
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D.
            Mutually exclusive with coords/grid_shape.
        coords : NDArray[np.int64] | None
            Coordinate array of source positions, shape [n_sources, ndim].
            Must be provided together with grid_shape.
        grid_shape : tuple[int, ...] | None
            Shape of the computational grid. Required when using coords input.

        Raises
        ------
        ValueError
            If grid_shape is not provided when using coords input.
            If both mask and coords are provided (mutually exclusive).
            If neither mask nor coords (with grid_shape) are provided.

        """
        self.p0 = np.atleast_2d(p0)

        if coords is not None:
            if grid_shape is None:
                msg = "grid_shape is required when using coords input"
                raise ValueError(msg)
            if mask is not None:
                msg = "mask and coords are mutually exclusive"
                raise ValueError(msg)
            self.incoords = np.atleast_2d(coords).astype(np.int64, copy=False)
            self.grid_shape = tuple(grid_shape)
        elif mask is not None:
            mask = np.atleast_2d(mask)
            self.grid_shape = mask.shape
            self.incoords = map_to_coords(mask)
        else:
            msg = "Either mask or coords (with grid_shape) must be provided"
            raise ValueError(msg)

        self.is_3d = len(self.grid_shape) == 3
        super().__init__()
        self.__post_init__()
        logger.debug("Source instance created.")

    def __post_init__(self) -> None:
        """Post-initialization processing for Source.

        Raises
        ------
        ValueError
            If the number of sources in the input signal
            does not match the number of source coordinates.

        """
        self.p0 = np.atleast_2d(self.p0)
        if self.p0.shape[0] != self.incoords.shape[0]:
            error_msg = "Input signal has the wrong number of elements"
            raise ValueError(error_msg)

    def validate(self, grid_shape: NDArray[np.int64] | tuple) -> None:
        """Check if the source coordinates are consistent with the grid shape."""
        grid_shape = tuple(grid_shape) if isinstance(grid_shape, np.ndarray) else grid_shape
        assert self.grid_shape == grid_shape, f"{self.grid_shape} != {grid_shape}"
        assert self.n_sources > 0, "No active source found."
        logger.debug("Source validated against grid shape.")

    @property
    def icmat(self) -> NDArray[np.float64]:
        """Returns icmat for the compatibility with the fullwave code."""
        return self.p0

    @property
    def mask(self) -> NDArray[np.int64]:
        """Returns the source mask.

        it calculates the source mask from the source coordinates to reduce the memory usage.
        """
        return coords_to_map(
            self.incoords,
            grid_shape=self.grid_shape,
            is_3d=self.is_3d,
        )

    @property
    def ncoords(self) -> int:
        """Return the number of sources.

        ailiased to n_sensors for the compatibility with matlab version
        """
        return self.n_sources

    @property
    def n_sources(self) -> int:
        """Return the number of sources."""
        return self.incoords.shape[0]

    def plot(
        self,
        export_path: Path | str | None = Path("./temp/temp.png"),
        *,
        show: bool = False,
        dpi: int = 3000,
    ) -> None:
        """Plot the transducer mask, optionally exporting and displaying the figure.

        Raises
        ------
        ValueError
            If 3D plotting is requested but not supported.

        """
        if self.is_3d:
            error_msg = "3D plotting is not supported yet."
            raise ValueError(error_msg)
        plt.close("all")
        fig, _ = plt.subplots()
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
        """Print source information to the logger."""
        print(str(self))

    def summary(self) -> None:
        """Alias for print_info."""
        self.print_info()

    def __str__(self) -> str:
        """Show source information.

        Returns
        -------
        str
            Formatted string containing source information.

        """
        return (
            f"Source: \n"
            f"  Number of sources: {self.n_sources}\n"
            f"  Grid shape: {self.grid_shape}\n"
            f"  Is 3D: {self.is_3d}\n"
            f"  p0 shape: {self.p0.shape}\n"
        )

    def __repr__(self) -> str:
        """Show source information.

        Returns
        -------
        str
            Formatted string containing source information.

        """
        return self.__str__()
