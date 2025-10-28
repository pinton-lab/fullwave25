"""domain base module."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fullwave import Grid, Medium
from fullwave.utils import check_functions


@dataclass
class Domain(ABC):
    """A base domain class defining abstract methods for domain parameter setup.

    override abstract methods (_setup_*) to use in the derived classes.
    """

    name: str
    grid: Grid
    nx: int = field(init=False)
    ny: int = field(init=False)
    nz: int = field(init=False)
    base_geometry: NDArray[np.float64] = field(init=False)
    sound_speed: NDArray[np.float64] = field(init=False)
    density: NDArray[np.float64] = field(init=False)
    alpha_coeff: NDArray[np.float64] = field(init=False)
    alpha_power: NDArray[np.float64] = field(init=False)
    beta: NDArray[np.float64] = field(init=False)
    air: NDArray[np.int64] = field(init=False)

    def __init__(
        self,
        name: str,
        grid: Grid,
        *,
        path_relaxation_parameters_database: Path = Path(__file__).parent.parent
        / "solver"
        / "bins"
        / "database"
        / "relaxation_params_database_num_relax=2_20251027_1437.mat",
        n_relaxation_mechanisms: int = 2,
    ) -> None:
        """Initialize the Domain class.

        Parameters
        ----------
        name: (str)
            The name of the domain.
        grid: (Grid)
            The grid associated with the domain.
        path_relaxation_parameters_database: (Path, optional)
            The path to the relaxation parameters database.
        n_relaxation_mechanisms: (int, optional)
            The number of relaxation mechanisms.

        """
        self.name = name
        self.grid = grid
        self.is_3d = self.grid.is_3d
        self.path_relaxation_parameters_database = path_relaxation_parameters_database
        self.n_relaxation_mechanisms = n_relaxation_mechanisms
        self.__post_init__()

    def __post_init__(self) -> None:
        """Post-initialization processing for Domain.

        will be called after __init__.
        """
        check_functions.check_instance(self.grid, Grid)

        self.nx = self.grid.nx
        self.ny = self.grid.ny
        if self.is_3d:
            self.nz = self.grid.nz

        self.base_geometry = self._setup_base_geometry()

        self.sound_speed = self._setup_sound_speed()
        self.density = self._setup_density()
        self.alpha_coeff = self._setup_alpha_coeff()
        self.alpha_power = self._setup_alpha_power()
        self.beta = self._setup_beta()
        self.air = self._setup_air()

    @abstractmethod
    def _setup_base_geometry(self) -> NDArray[np.float64]:
        if self.is_3d:
            return np.zeros((self.nx, self.ny, self.nz))
        return np.zeros((self.nx, self.ny))

    @abstractmethod
    def _setup_sound_speed(self) -> NDArray[np.float64]:
        if self.is_3d:
            return np.zeros((self.nx, self.ny, self.nz))
        return np.zeros((self.nx, self.ny))

    @abstractmethod
    def _setup_density(self) -> NDArray[np.float64]:
        if self.is_3d:
            return np.zeros((self.nx, self.ny, self.nz))
        return np.zeros((self.nx, self.ny))

    @abstractmethod
    def _setup_alpha_coeff(self) -> NDArray[np.float64]:
        if self.is_3d:
            return np.zeros((self.nx, self.ny, self.nz))
        return np.zeros((self.nx, self.ny))

    @abstractmethod
    def _setup_alpha_power(self) -> NDArray[np.float64]:
        if self.is_3d:
            return np.zeros((self.nx, self.ny, self.nz))
        return np.zeros((self.nx, self.ny))

    @abstractmethod
    def _setup_beta(self) -> NDArray[np.float64]:
        if self.is_3d:
            return np.zeros((self.nx, self.ny, self.nz))
        return np.zeros((self.nx, self.ny))

    def _setup_air(self) -> NDArray[np.int64]:
        if self.is_3d:
            return np.zeros((self.nx, self.ny, self.nz), dtype=bool)
        return np.zeros((self.nx, self.ny), dtype=bool)

    @property
    def medium(self) -> Medium:
        """Returns the medium instance with its configured properties."""
        return Medium(
            grid=self.grid,
            sound_speed=self.sound_speed,
            density=self.density,
            alpha_coeff=self.alpha_coeff,
            alpha_power=self.alpha_power,
            beta=self.beta,
            air_map=self.air,
            path_relaxation_parameters_database=self.path_relaxation_parameters_database,
            n_relaxation_mechanisms=self.n_relaxation_mechanisms,
        )

    def plot(
        self,
        export_path: Path | str | None = Path("./temp/temp.png"),
        *,
        show: bool = False,
    ) -> None:
        """Plot the fields using matplotlib."""
        self.medium.plot(export_path=export_path, show=show)
