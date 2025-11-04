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

    def __init__(self, mask: NDArray[np.bool], sampling_modulus_time: int = 1) -> None:
        """Sensor class for Fullwave.

        Parameters
        ----------
        mask : NDArray[np.bool]
            Binary matrix where the pressure is recorded at each time-step
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D
        sampling_modulus_time: int
            Sampling modulus in time. Default is 1 (record at every time step).
            Changing this value to n will record the pressure every n time steps.
            It reduces the size of the output data.

        """
        mask = np.atleast_2d(mask)

        self.grid_shape = mask.shape
        self.sampling_modulus_time = sampling_modulus_time
        self.is_3d = len(self.grid_shape) == 3
        outcoords = map_to_coords(mask)
        if self.is_3d:
            self.outcoords = outcoords
        else:
            # self.outcoords = np.stack([outcoords[:, 1], outcoords[:, 0]]).T
            self.outcoords = outcoords
        super().__init__()
        logger.debug("Sensor instance created.")

    def validate(self, grid_shape: NDArray[np.int64] | tuple) -> None:
        """Check if the source mask has the correct shape."""
        grid_shape = tuple(grid_shape) if isinstance(grid_shape, np.ndarray) else grid_shape
        assert self.mask.shape == grid_shape, f"{self.mask.shape} != {grid_shape}"
        assert np.any(self.mask), "No active sensor found."
        logger.debug("Sensor mask validated against grid shape.")

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

        Raises:
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
