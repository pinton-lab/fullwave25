"""background domain."""

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from fullwave import Grid
from fullwave.constants import MaterialProperties

from fullwave.medium_builder.domain import Domain  # isort:skip


class BackgroundDomain(Domain):
    """represents the base medium properties for the simulation."""

    def __init__(
        self,
        grid: Grid,
        background_property_name: str | None = None,
        material_properties: MaterialProperties | None = None,
        *,
        path_relaxation_parameters_database: Path = Path(__file__).parent.parent
        / "solver"
        / "bins"
        / "database"
        / "relaxation_params_database_num_relax=2_20251027_1437.mat",
        n_relaxation_mechanisms: int = 2,
    ) -> None:
        """Initialize a Background instance.

        Parameters
        ----------
        grid: Grid
            Grid instance.
        background_property_name: str, optional
            Background property name.
            Defaults to None.
        material_properties: MaterialProperties, optional
            Material properties.
            Defaults to MaterialProperties().
        path_relaxation_parameters_database: (Path, optional)
            The path to the relaxation parameters database.
        n_relaxation_mechanisms: (int, optional)
            The number of relaxation mechanisms.

        """
        self.background_property_name = background_property_name
        if material_properties is None:
            self.material_properties: MaterialProperties = MaterialProperties()
        else:
            self.material_properties = material_properties

        super().__init__(
            name="background",
            grid=grid,
            path_relaxation_parameters_database=path_relaxation_parameters_database,
            n_relaxation_mechanisms=n_relaxation_mechanisms,
        )

    def _setup_base_geometry(self) -> NDArray[np.float64]:
        if self.is_3d:
            return np.ones((self.nx, self.ny, self.nz))
        return np.ones((self.nx, self.ny))

    def _setup_sound_speed(self) -> NDArray[np.float64]:
        base_map: NDArray[np.float64]
        if self.is_3d:
            base_map = np.ones((self.nx, self.ny, self.nz))
        else:
            base_map = np.ones((self.nx, self.ny))
        if self.background_property_name is not None:
            sound_speed = (
                base_map
                * getattr(self.material_properties, self.background_property_name)["sound_speed"]
            )
        else:
            sound_speed = base_map * self.material_properties.sound_speed
        return sound_speed

    def _setup_density(self) -> NDArray[np.float64]:
        base_map: NDArray[np.float64]
        if self.is_3d:
            base_map = np.ones((self.nx, self.ny, self.nz))
        else:
            base_map = np.ones((self.nx, self.ny))
        if self.background_property_name is not None:
            density = (
                base_map
                * getattr(self.material_properties, self.background_property_name)["density"]
            )
        else:
            density = base_map * self.material_properties.density
        return density

    def _setup_alpha_coeff(self) -> NDArray[np.float64]:
        base_map: NDArray[np.float64]
        if self.is_3d:
            base_map = np.ones((self.nx, self.ny, self.nz))
        else:
            base_map = np.ones((self.nx, self.ny))

        if self.background_property_name is not None:
            alpha_coeff = (
                base_map
                * getattr(self.material_properties, self.background_property_name)["alpha_coeff"]
            )
        else:
            alpha_coeff = base_map * self.material_properties.alpha_coeff
        return alpha_coeff

    def _setup_alpha_power(self) -> NDArray[np.float64]:
        base_map: NDArray[np.float64]
        if self.is_3d:
            base_map = np.ones((self.nx, self.ny, self.nz))
        else:
            base_map = np.ones((self.nx, self.ny))

        if self.background_property_name is not None:
            alpha_power = (
                base_map
                * getattr(self.material_properties, self.background_property_name)["alpha_power"]
            )
        else:
            alpha_power = base_map * self.material_properties.alpha_power
        return alpha_power

    def _setup_beta(self) -> NDArray[np.float64]:
        base_map: NDArray[np.float64]
        if self.is_3d:
            base_map = np.ones((self.nx, self.ny, self.nz))
        else:
            base_map = np.ones((self.nx, self.ny))

        if self.background_property_name is not None:
            beta = (
                base_map * getattr(self.material_properties, self.background_property_name)["beta"]
            )
        else:
            beta = base_map * self.material_properties.beta
        return beta
