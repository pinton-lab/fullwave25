"""scatterer domain."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fullwave import Grid
from fullwave.constants import MaterialProperties
from fullwave.utils import check_functions

from fullwave.medium_builder.domain import Domain  # isort:skip


class ScattererDomain(Domain):
    """represents the base medium properties for the simulation."""

    def __init__(
        self,
        grid: Grid,
        num_scatterer: int,
        ncycles: int,
        material_properties: MaterialProperties | None = None,
        *,
        path_relaxation_parameters_database: Path = Path(__file__).parent.parent
        / "solver"
        / "bins"
        / "database"
        / "relaxation_params_database_num_relax=2_20251027_1437.mat",
        n_relaxation_mechanisms: int = 2,
        seed: int | None = None,
    ) -> None:
        """Initialize a Background instance.

        Parameters
        ----------
        grid : Grid
            Grid instance.
        num_scatterer : int
            Number of scatterer per resolution cell.
        ncycles : int
            Number of cycles for the transmit pulse.
        material_properties: MaterialProperties, optional
            Material properties.
            Defaults to MaterialProperties().
        path_relaxation_parameters_database: (Path, optional)
            The path to the relaxation parameters database.
        n_relaxation_mechanisms: (int, optional)
            The number of relaxation mechanisms.
        seed: (int, optional)
            Random seed for scatterer generation.

        """
        check_functions.check_instance(grid, Grid)
        self.grid = grid
        self.is_3d = grid.is_3d
        self.nx = grid.nx
        self.ny = grid.ny
        if self.is_3d:
            self.nz = grid.nz
        self.path_relaxation_parameters_database = path_relaxation_parameters_database
        self.n_relaxation_mechanisms = n_relaxation_mechanisms

        if material_properties is None:
            self.material_properties: MaterialProperties = MaterialProperties()
        else:
            self.material_properties = material_properties

        self.num_scatterer = num_scatterer
        self.ncycles = ncycles

        self.seed = seed
        self.rng = np.random.default_rng(self.seed)

        self.base_geometry = self._setup_base_geometry()

        self.scatterer_map, self.scatterer_count, self.scatterer_percent = self._setup_scatter_map()

        self.sound_speed = self._setup_sound_speed()
        self.density = self._setup_density()
        self.alpha_coeff = self._setup_alpha_coeff()
        self.alpha_power = self._setup_alpha_power()
        self.beta = self._setup_beta()

        self.air = self._setup_air()

    def _setup_base_geometry(self) -> NDArray[np.float64]:
        if self.is_3d:
            return np.ones((self.nx, self.ny, self.nz))
        return np.ones((self.nx, self.ny))

    def _setup_sound_speed(self) -> NDArray[np.float64]:
        if self.is_3d:
            return np.zeros((self.nx, self.ny, self.nz))
        return np.zeros((self.nx, self.ny))

    def _setup_density(self) -> NDArray[np.float64]:
        return self.scatterer_map * self.material_properties.density

    def _setup_alpha_coeff(self) -> NDArray[np.float64]:
        if self.is_3d:
            return np.zeros((self.nx, self.ny, self.nz))
        return np.zeros((self.nx, self.ny))

    def _setup_alpha_power(self) -> NDArray[np.float64]:
        if self.is_3d:
            return np.zeros((self.nx, self.ny, self.nz))
        return np.zeros((self.nx, self.ny))

    def _setup_beta(self) -> NDArray[np.float64]:
        if self.is_3d:
            return np.zeros((self.nx, self.ny, self.nz))
        return np.zeros((self.nx, self.ny))

    def _setup_scatter_map(
        self,
    ) -> tuple[NDArray[np.float64], int, float]:
        # if self.is_3d:
        #     res_cell = self._rescell3ds(
        #         self.grid.wavelength,
        #         self.grid.ny / 2 * self.grid.dy,
        #         self.grid.domain_size[1],
        #         self.ncycles,
        #         self.grid.dx,
        #         self.grid.dy,
        #         self.grid.dz,
        #     )
        # else:
        res_cell = self._rescell2ds(
            self.grid.wavelength,
            self.grid.ny / 2 * self.grid.dy,
            self.grid.domain_size[1],
            self.ncycles,
            self.grid.dx,
            self.grid.dy,
        )
        scat_density = self.num_scatterer / res_cell

        if self.is_3d:
            scatter_map = self.rng.random((self.grid.nx, self.grid.ny, self.grid.nz))
        else:
            scatter_map = self.rng.random((self.grid.nx, self.grid.ny))

        scatter_map /= scat_density
        scatter_map[scatter_map > 1] = 0.5
        scatter_map -= -0.5

        scatterer_count = len(scatter_map != 0)
        scatterer_percent = 100 * scatterer_count / (scatter_map.shape[0] * scatter_map.shape[0])

        scatter_map *= self.base_geometry
        return scatter_map, scatterer_count, scatterer_percent

    def _rescell2ds(
        self,
        wavelength: float,
        dy2: float,  # ?
        ay: float,
        n_cycles: int,
        dy: float,
        dz: float,
    ) -> float:
        res_y = wavelength * dy2 / ay
        res_z = wavelength * n_cycles / 2
        return res_y / dy * res_z / dz

    def _rescell3ds(
        self,
        wavelength: float,
        dy2: float,  # ?
        ay: float,
        n_cycles: int,
        dx: float,
        dy: float,
        dz: float,
    ) -> float:
        res_x = wavelength * n_cycles / 2
        res_y = wavelength * dy2 / ay
        res_z = wavelength * n_cycles / 2
        return res_x / dx * res_y / dy * res_z / dz
