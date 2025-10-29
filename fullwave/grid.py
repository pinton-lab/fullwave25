"""Class to calculate the grid size, spacing, time step."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Grid:
    """Grid class to calculate grid size, spacing, and time step.

    Parameters
    ----------
    domain_size : tuple
        Size of the domain in x, y, and z directions [m].
        For 2D simulations, (x, y) is required.
        For 3D simulations, (x, y, z) is required.
        e.g. (0.1, 0.1) for 2D and (0.1, 0.1, 0.1) for 3D.
    f0 : float
        Center frequency of the wave [Hz].
        e.g. 1e6 for 1 MHz.
    duration : float
        Duration of the simulation [s].
        e.g. 1e-6 for 1 us.
    c0 : float, optional
        Speed of sound in the medium [m/s], by default 1540.
        This value is only used to calculate the grid spacing.
        The sound speed map is defined in the medium class.
    ppw : int, optional
        Points per wavelength, by default 12.
        The number of grid points per wavelength.
        It changes the accuracy of the simulation.
    cfl : float, optional
        Courant-Friedrichs-Lewy number, by default 0.2.
        It changes the stability of the simulation.
        lower values are more stable but slower.
        0.2 - 0.45 is a good range.

    """

    domain_size: tuple[float, ...]
    f0: float
    duration: float
    c0: float = 1540
    ppw: int = 12
    cfl: float = 0.2

    def __post_init__(self) -> None:
        """Post-initialization processing for Grid.

        Check the input parameters.
        """
        self._check_input()
        self.is_3d = len(self.domain_size) == 3

    def _check_input(self) -> None:
        """Check the input parameters."""
        assert len(self.domain_size) in [2, 3], "Domain size must be 2D or 3D"
        assert self.f0 > 0, "Center frequency must be greater than 0"
        assert self.f0 < 30e6, "Center frequency must be less than 30 MHz"
        assert self.duration > 0, "Duration must be greater than 0"
        assert self.c0 > 0, "Speed of sound must be greater than 0"
        assert self.ppw > 5, "Points per wavelength must be greater than 5"
        assert self.cfl > 0.01, "CFL number must be greater than 0.01"
        assert self.cfl < 0.5, "CFL number must be less than 0.5"

    @property
    def omega(self) -> float:
        """Angular frequency of the wave [rad/s]."""
        return 2 * np.pi * self.f0

    @property
    def wavelength(self) -> float:
        """Wavelength of the wave [m]."""
        return self.c0 / self.f0

    @property
    def nx(self) -> int:
        """Number of grid points in x-direction."""
        return int(np.round(self.domain_size[0] / self.dx).item())

    @property
    def ny(self) -> int:
        """Number of grid points in y-direction."""
        return int(np.round(self.domain_size[1] / self.dy).item())

    @property
    def nz(self) -> int:
        """Number of grid points in z-direction."""
        return (
            int(np.round(self.domain_size[2] / self.dz).item()) if len(self.domain_size) == 3 else 0
        )

    @property
    def n_axial(self) -> int:
        """Number of grid points in axial-direction."""
        return self.nx

    @property
    def n_lateral(self) -> int:
        """Number of grid points in lateral-direction."""
        return self.ny

    @property
    def n_elevational(self) -> int:
        """Number of grid points in elevational-direction."""
        return self.nz

    @property
    def dx(self) -> float:
        """Grid point spacing in x-direction [m]."""
        return self.wavelength / self.ppw

    @property
    def dy(self) -> float:
        """Grid point spacing in y-direction [m]."""
        return self.wavelength / self.ppw

    @property
    def dz(self) -> float:
        """Grid point spacing in z-direction [m]."""
        return self.wavelength / self.ppw if len(self.domain_size) == 3 else 0

    @property
    def nt(self) -> int:
        """Number of time steps."""
        return round(self.duration / self.dt)

    @property
    def dt(self) -> float:
        """Time step [s]."""
        return self.cfl * self.dx / self.c0

    @property
    def t(self) -> np.ndarray:
        """Time step [s]."""
        return np.arange(0, self.duration, self.dt)

    @property
    def shape(self) -> tuple[int, int, int] | tuple[int, int]:
        """Shape of the grid."""
        return (self.nx, self.ny, self.nz) if len(self.domain_size) == 3 else (self.nx, self.ny)

    def print_info(self) -> None:
        """Print grid information."""
        print(str(self))

    def __str__(self) -> str:
        """Print string representation of the Grid object.

        Returns
        -------
        str
            String representation of the Grid object.

        """
        return (
            "Grid Information:\n"
            f"  Domain size: ({self.domain_size[0]:.2e} m, {self.domain_size[1]:.2e} m"
            + (f", {self.domain_size[2]:.2f} m)" if self.is_3d else ")")
            + "\n"
            f"  Center frequency: {self.f0 / 1e6} MHz\n"
            f"  Duration: {self.duration:.2e} s\n"
            f"  Speed of sound: {self.c0} m/s\n"
            f"  Points per wavelength (PPW): {self.ppw}\n"
            f"  Courant-Friedrichs-Lewy (CFL) number: {self.cfl}\n"
            f"  Wavelength: {self.wavelength * 1e3:.2e} m\n"
            "  Grid spacing (dx, dy, dz): "
            f"({self.dx * 1e3:.2e}, {self.dy * 1e3:.2e}, {self.dz * 1e3:.2e}) m\n"
            f"  Number of grid points (nx, ny, nz): ({self.nx}, {self.ny}, {self.nz})\n"
            f"  Time step (dt): {self.dt:.2e} sec\n"
            f"  Number of time steps (nt): {self.nt}\n"
            f"  is 3D simulation: {self.is_3d}"
        )

    def __repr__(self) -> str:
        """Print string representation of the Grid object.

        Returns
        -------
        str
            String representation of the Grid object.

        """
        return self.__str__()
