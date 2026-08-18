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
    p0_additive: NDArray[np.float64] | None = None
    incoords_add: NDArray[np.int64] | None = None
    u0: NDArray[np.float64] | None = None
    incoords_u: NDArray[np.int64] | None = None
    v0: NDArray[np.float64] | None = None
    incoords_v: NDArray[np.int64] | None = None
    w0: NDArray[np.float64] | None = None
    incoords_w: NDArray[np.int64] | None = None

    def __init__(  # noqa: PLR0912
        self,
        p0: NDArray[np.float64] | None = None,
        mask: NDArray[np.bool] | None = None,
        *,
        # ---
        p0_additive: NDArray[np.float64] | None = None,
        mask_additive: NDArray[np.bool] | None = None,
        # ---
        coords: NDArray[np.int64] | None = None,
        coords_additive: NDArray[np.int64] | None = None,
        # ---
        grid_shape: tuple[int, ...] | None = None,
        # --- velocity source components ---
        u0: NDArray[np.float64] | None = None,
        coords_u: NDArray[np.int64] | None = None,
        mask_u: NDArray[np.bool] | None = None,
        v0: NDArray[np.float64] | None = None,
        coords_v: NDArray[np.int64] | None = None,
        mask_v: NDArray[np.bool] | None = None,
        w0: NDArray[np.float64] | None = None,
        coords_w: NDArray[np.int64] | None = None,
        mask_w: NDArray[np.bool] | None = None,
    ) -> None:
        """Source class for Fullwave.

        Parameters
        ----------
        p0 : NDArray[np.float64] | None
            Time-varying pressure at each source position; shape [n_sources, nt].
            If None, either p0_additive or a velocity component (u0/v0/w0) must be provided.
            When omitted with a velocity-only source, the pressure node list is empty
            and icmat is written as a zero-row matrix.
        mask : NDArray[np.bool] | None
            binary matrix specifying the positions of the time varying pressure source distribution
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D.
            Mutually exclusive with coords/grid_shape.
        coords : NDArray[np.int64] | None
            Coordinate array of source positions (hard source); shape [n_sources, ndim].
            Must be provided together with grid_shape.
        grid_shape : tuple[int, ...] | None
            Shape of the computational grid. Required when using coords or velocity-only input.
        p0_additive : NDArray[np.float64] | None
            Optional additive (soft) source term; shape [n_sources_add, nt].
            When provided, positions come from coords_additive or mask_additive,
            or default to primary coords.
        mask_additive : NDArray[np.bool] | None
            Mask for additive source positions; shape matches grid_shape.
            Mutually exclusive with coords_additive.
        coords_additive : NDArray[np.int64] | None
            Coordinates for the additive source; shape [n_sources_add, ndim].
            If None and p0_additive is provided, defaults to primary incoords.
            Mutually exclusive with mask_additive.
        u0 : NDArray[np.float64] | None
            Time-varying velocity in the depth (x) direction; shape [n_sources_u, nt].
        coords_u : NDArray[np.int64] | None
            Coordinates for the u-velocity source; shape [n_sources_u, ndim].
            Mutually exclusive with mask_u.
        mask_u : NDArray[np.bool] | None
            Boolean mask for u-velocity source positions. Mutually exclusive with coords_u.
        v0 : NDArray[np.float64] | None
            Time-varying velocity in the lateral (y) direction; shape [n_sources_v, nt].
        coords_v : NDArray[np.int64] | None
            Coordinates for the v-velocity source; shape [n_sources_v, ndim].
            Mutually exclusive with mask_v.
        mask_v : NDArray[np.bool] | None
            Boolean mask for v-velocity source positions. Mutually exclusive with coords_v.
        w0 : NDArray[np.float64] | None
            Time-varying velocity in the elevational (z) direction; shape [n_sources_w, nt].
        coords_w : NDArray[np.int64] | None
            Coordinates for the w-velocity source; shape [n_sources_w, ndim].
            Mutually exclusive with mask_w.
        mask_w : NDArray[np.bool] | None
            Boolean mask for w-velocity source positions. Mutually exclusive with coords_w.

        Raises
        ------
        ValueError
            If grid_shape is missing when using coords, additive-only, or velocity-only.
            If both mask and coords, or both coords_additive and mask_additive, are provided.
            If primary source positions cannot be resolved (need coords+grid_shape, mask,
            additive-only args, or at least one velocity component with grid_shape).
            If all of p0, p0_additive, u0, v0, and w0 are None.
            If p0 / p0_additive row counts do not match their coordinate arrays.

        """
        # --- Resolve primary source positions and grid_shape ---
        if coords is not None:
            if grid_shape is None:
                error_msg = "grid_shape is required when using coords"
                raise ValueError(error_msg)
            if mask is not None:
                error_msg = "mask and coords are mutually exclusive"
                raise ValueError(error_msg)
            self.incoords = np.atleast_2d(coords).astype(np.int64, copy=False)
            self.grid_shape = tuple(grid_shape)
        elif mask is not None:
            mask = np.atleast_2d(mask)
            self.grid_shape = mask.shape
            self.incoords = map_to_coords(mask)
        elif p0_additive is not None and (coords_additive is not None or mask_additive is not None):
            # Additive-only: there is no hard source, so the hard node list stays
            # empty. Copying the additive positions here would make the solver
            # assign a zero pressure at them on every step, which is a
            # pressure-release wall standing on the source.
            if coords_additive is not None:
                if grid_shape is None:
                    error_msg = "grid_shape is required for additive-only with coords_additive"
                    raise ValueError(
                        error_msg,
                    )
                self.grid_shape = tuple(grid_shape)
            else:
                assert mask_additive is not None
                self.grid_shape = np.atleast_2d(mask_additive).shape
            self.incoords = np.empty((0, len(self.grid_shape)), dtype=np.int64)
        elif u0 is not None or v0 is not None or w0 is not None:
            # Velocity-only: no hard pressure nodes; pressure arrays will be empty
            if grid_shape is None:
                error_msg = "grid_shape is required for velocity-only source"
                raise ValueError(error_msg)
            self.grid_shape = tuple(grid_shape)
            self.incoords = np.empty((0, len(self.grid_shape)), dtype=np.int64)
        else:
            error_msg = (
                "Provide (coords + grid_shape), or mask, or"
                " (p0_additive + (coords_additive or mask_additive)), or"
                " a velocity component (u0/v0/w0) with grid_shape"
            )
            raise ValueError(
                error_msg,
            )

        if p0 is None and p0_additive is None and u0 is None and v0 is None and w0 is None:
            error_msg = (
                "At least one of p0, p0_additive, or a velocity component"
                " (u0/v0/w0) must be provided"
            )
            raise ValueError(error_msg)

        # --- Resolve additive coords (for later use) ---
        if coords_additive is not None and mask_additive is not None:
            error_msg = "coords_additive and mask_additive are mutually exclusive"
            raise ValueError(error_msg)
        if coords_additive is not None:
            _coords_add = np.atleast_2d(coords_additive).astype(np.int64, copy=False)
        elif mask_additive is not None:
            mask_add = np.atleast_2d(mask_additive)
            if mask_add.shape != self.grid_shape:
                error_msg = "mask_additive shape must match grid_shape"
                raise ValueError(error_msg)
            _coords_add = map_to_coords(mask_add)
        else:
            _coords_add = None

        # --- Set p0 and p0_additive ---
        if p0 is not None:
            self.p0 = np.atleast_2d(p0)
            if self.p0.shape[0] != self.incoords.shape[0]:
                error_msg = (
                    f"p0 has {self.p0.shape[0]} rows but "
                    f"incoords has {self.incoords.shape[0]} (must match)"
                )
                raise ValueError(error_msg)
            self.p0_additive = np.atleast_2d(p0_additive) if p0_additive is not None else None
        elif p0_additive is not None:
            p0_add = np.atleast_2d(p0_additive)
            n_add = _coords_add.shape[0] if _coords_add is not None else self.incoords.shape[0]
            if p0_add.shape[0] != n_add:
                error_msg = (
                    f"p0_additive shape {p0_add.shape} must have {n_add} rows "
                    f"(coords_additive/mask_additive or primary coords)"
                )
                raise ValueError(error_msg)
            self.p0 = np.zeros((self.incoords.shape[0], p0_add.shape[1]), dtype=np.float64)
            self.p0_additive = p0_add
        else:
            # Velocity-only: no pressure nodes; derive nt from the first velocity signal
            _first_vel = next(s for s in (u0, v0, w0) if s is not None)
            nt_vel = np.atleast_2d(_first_vel).shape[1]
            self.p0 = np.zeros((0, nt_vel), dtype=np.float64)
            self.p0_additive = None

        # --- Set incoords_add for writer/binary ---
        if self.p0_additive is not None:
            self.incoords_add = _coords_add if _coords_add is not None else self.incoords
        else:
            self.incoords_add = None

        # --- Resolve velocity source components ---
        self.incoords_u, self.u0 = self._resolve_velocity_component("u", u0, coords_u, mask_u)
        self.incoords_v, self.v0 = self._resolve_velocity_component("v", v0, coords_v, mask_v)
        self.incoords_w, self.w0 = self._resolve_velocity_component("w", w0, coords_w, mask_w)

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
            If p0_additive length does not match incoords_add.

        """
        self.p0 = np.atleast_2d(self.p0)
        if self.p0.shape[0] != self.incoords.shape[0]:
            error_msg = "Input signal has the wrong number of elements"
            raise ValueError(error_msg)
        if self.p0_additive is not None and self.incoords_add is not None:
            self.p0_additive = np.atleast_2d(self.p0_additive)
            if self.p0_additive.shape[0] != self.incoords_add.shape[0]:
                error_msg = (
                    f"p0_additive shape {self.p0_additive.shape} must have "
                    f"{self.incoords_add.shape[0]} rows (incoords_add)"
                )
                raise ValueError(error_msg)

    @staticmethod
    def _resolve_velocity_component(
        name: str,
        signal: NDArray | None,
        coords: NDArray | None,
        mask: NDArray | None,
    ) -> tuple[NDArray | None, NDArray | None]:
        """Resolve a velocity source component to (incoords, signal_arr).

        Returns
        -------
        tuple[NDArray | None, NDArray | None]
            (incoords, signal_arr) if the component is provided, else (None, None).

        Raises
        ------
        ValueError
            If coords and mask are both provided.
            If signal is provided but neither coords nor mask are given.
            If signal is None but coords/mask are provided.
            If signal row count does not match the number of coordinate points.

        """
        if signal is None:
            if coords is not None or mask is not None:
                error_msg = f"coords_{name}/mask_{name} provided without {name}0 signal"
                raise ValueError(error_msg)
            return None, None
        if coords is not None and mask is not None:
            error_msg = f"coords_{name} and mask_{name} are mutually exclusive"
            raise ValueError(error_msg)
        if coords is None and mask is None:
            error_msg = f"{name}0 signal provided but neither coords_{name} nor mask_{name} given"
            raise ValueError(error_msg)
        if coords is not None:
            incoords = np.atleast_2d(coords).astype(np.int64, copy=False)
        else:
            incoords = map_to_coords(np.atleast_2d(mask))
        signal_arr = np.atleast_2d(signal)
        if signal_arr.shape[0] != incoords.shape[0]:
            error_msg = (
                f"{name}0 has {signal_arr.shape[0]} rows but "
                f"coords_{name}/mask_{name} has {incoords.shape[0]} (must match)"
            )
            raise ValueError(error_msg)
        return incoords, signal_arr

    def validate(self, grid_shape: NDArray[np.int64] | tuple) -> None:
        """Check if the source coordinates are consistent with the grid shape."""
        grid_shape = tuple(grid_shape) if isinstance(grid_shape, np.ndarray) else grid_shape
        assert self.grid_shape == grid_shape, f"{self.grid_shape} != {grid_shape}"
        assert (
            self.n_sources > 0
            or self.n_sources_add > 0
            or self.n_sources_u > 0
            or self.n_sources_v > 0
            or self.n_sources_w > 0
        ), "No active source found."
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
        """Return the number of sources (hard / primary)."""
        return self.incoords.shape[0]

    @property
    def n_sources_add(self) -> int:
        """Return the number of additive (soft) source positions."""
        if self.incoords_add is None:
            return 0
        return self.incoords_add.shape[0]

    @property
    def n_sources_u(self) -> int:
        """Return the number of velocity-u (depth) source positions."""
        return self.incoords_u.shape[0] if self.incoords_u is not None else 0

    @property
    def n_sources_v(self) -> int:
        """Return the number of velocity-v (lateral) source positions."""
        return self.incoords_v.shape[0] if self.incoords_v is not None else 0

    @property
    def n_sources_w(self) -> int:
        """Return the number of velocity-w (elevational) source positions."""
        return self.incoords_w.shape[0] if self.incoords_w is not None else 0

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
        lines = [
            "Source:",
            f"  Number of sources: {self.n_sources}",
            f"  Grid shape: {self.grid_shape}",
            f"  Is 3D: {self.is_3d}",
            f"  p0 shape: {self.p0.shape}",
        ]
        if self.p0_additive is not None:
            lines.append(f"  p0_additive shape: {self.p0_additive.shape}")
        if self.incoords_add is not None:
            lines.append(f"  incoords_add: {self.incoords_add.shape[0]} points")
        for comp, sig, n in (
            ("u", self.u0, self.n_sources_u),
            ("v", self.v0, self.n_sources_v),
            ("w", self.w0, self.n_sources_w),
        ):
            if sig is not None:
                lines.append(f"  {comp}0 shape: {sig.shape} ({n} points)")
        return "\n".join(lines) + "\n"

    def __repr__(self) -> str:
        """Show source information.

        Returns
        -------
        str
            Formatted string containing source information.

        """
        return self.__str__()
