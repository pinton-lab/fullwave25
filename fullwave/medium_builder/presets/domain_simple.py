"""simple domain module."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fullwave import Grid

from fullwave.medium_builder.domain import Domain  # isort:skip


class SimpleDomain(Domain):
    """A simple domain implementation for custom medium builder domains."""

    def __init__(
        self,
        grid: Grid,
        name: str,
        geometry: NDArray[np.float64],
        maps: dict[str, NDArray[np.float64] | NDArray[np.int64]],
        *,
        path_relaxation_parameters_database: Path = Path(__file__).parent.parent
        / "solver"
        / "bins"
        / "database"
        / "database"
        / "relaxation_params_database_num_relax=2_20251027_1437.mat",
        n_relaxation_mechanisms: int = 2,
    ) -> None:
        """Initialize a SimpleDomain.

        This domain is designed to make a arbitrary domain with given maps,
        so that users can easily create a domain with their own maps
        and integrate it into the medium_builder.

        Parameters
        ----------
        grid: Grid
            The grid instance.
        name : str
            The name of the domain.
        geometry : NDArray
            The geometry array.
        maps : NDArray
            A dictionary containing maps
            for sound speed, density, beta, alpha_coeff, alpha_power, and optionally air.
        path_relaxation_parameters_database : Path, optional
            The path to the relaxation parameters database.
        n_relaxation_mechanisms : int, optional
            The number of relaxation mechanisms.

        """
        self.grid = grid
        self.is_3d = grid.is_3d
        self.name = name
        self.base_geometry = geometry
        self.sound_speed = maps["sound_speed"]
        self.density = maps["density"]
        self.beta = maps["beta"]
        self.alpha_coeff = maps["alpha_coeff"]
        self.alpha_power = maps["alpha_power"]
        if "air" in maps:
            self.air = maps["air"]
        else:
            self.air = np.zeros_like(geometry, dtype=int)
        super().__init__(
            grid=grid,
            name=name,
            path_relaxation_parameters_database=path_relaxation_parameters_database,
            n_relaxation_mechanisms=n_relaxation_mechanisms,
        )

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
