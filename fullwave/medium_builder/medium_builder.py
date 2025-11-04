"""medium builder module."""

import logging
from collections import OrderedDict
from pathlib import Path

import numpy as np

from fullwave import Medium
from fullwave.constants import MaterialProperties
from fullwave.grid import Grid
from fullwave.medium_builder.domain import Domain
from fullwave.utils import check_functions

logger = logging.getLogger("__main__." + __name__)


class MediumBuilder:
    """organize the domain properties and export Medium instances.

    registered domains will be combined to construct the final domain properties.
    """

    def __init__(
        self,
        grid: Grid,
        background_domain_properties: str = "water",
        *,
        ignore_non_linearity: bool = False,
        material_properties: MaterialProperties | None = None,
        path_relaxation_parameters_database: Path = Path(__file__).parent.parent
        / "solver"
        / "bins"
        / "database"
        / "relaxation_params_database_num_relax=2_20251027_1437.mat",
        n_relaxation_mechanisms: int = 2,
        attenuation_builder: str = "lookup",
    ) -> None:
        """Initialize a DomainOrganizer.

        with background domain properties, non-linearity flag, and material properties.

        Parameters
        ----------
        grid: Grid
            grid instance.
        background_domain_properties: str
            The background domain properties.
            Defaults to "water".
        ignore_non_linearity: bool
            Ignore non-linearity.
            Defaults to False.
        material_properties: MaterialProperties, optional
            Material properties.
            Defaults to None.
        path_relaxation_parameters_database : Path, optional
            Path to the relaxation parameters database.
        n_relaxation_mechanisms : int, optional
            Number of relaxation mechanisms, by default 2
        attenuation_builder : str, optional
            Attenuation builder type, by default "lookup".
            Currently supports "lookup" only.
            If "lookup", it uses the lookup table for attenuation.

        """
        check_functions.check_instance(grid, Grid)
        self.grid = grid
        self.registered_domain_dict: OrderedDict = OrderedDict()
        self.constructed_domain_dict: OrderedDict = OrderedDict()

        if material_properties is None:
            self.material_properties: MaterialProperties = MaterialProperties()
        else:
            self.material_properties = material_properties

        self.background_domain_properties: str = background_domain_properties
        self.ignore_non_linearity: bool = ignore_non_linearity
        self.path_relaxation_parameters_database = path_relaxation_parameters_database
        self.n_relaxation_mechanisms = n_relaxation_mechanisms
        self.attenuation_builder = attenuation_builder

    def register_domain(self, domain: Domain) -> None:
        """Register a single Domain instances.

        Parameters
        ----------
        domain: Domain
            Domain instances to register.

        """
        check_functions.check_instance(domain, Domain)

        domain_name = domain.name
        c_map = domain.sound_speed
        rho_map = domain.density
        beta_map = domain.beta
        alpha_coeff_map = domain.alpha_coeff
        alpha_power_map = domain.alpha_power

        base_geometry = domain.base_geometry

        air_map = domain.air if hasattr(domain, "air") else np.zeros_like(base_geometry)

        self.registered_domain_dict[domain_name] = {
            "rho_map": rho_map,
            "beta_map": beta_map,
            "c_map": c_map,
            "a_map": alpha_coeff_map,
            "a_power_map": alpha_power_map,
            "air_map": air_map,
            "geometry": base_geometry,
        }

    def register_domain_list(self, domain_list: list[Domain]) -> None:
        """Register a list of Domain instances.

        Parameters
        ----------
        domain_list: List
            List of Domain instances to register.

        """
        for domain in domain_list:
            self.register_domain(domain)

    def _check_registered_domain(self) -> None:
        if "background" not in self.registered_domain_dict:
            error_msg = "Background domain is not registered."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if len(self.registered_domain_dict) == 0:
            error_msg = "No domain is registered."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if next(iter(self.registered_domain_dict.keys())) != "background":
            error_msg = "The first domain must be background domain."
            logger.error(error_msg)
            raise ValueError(error_msg)

    def run(self) -> Medium:
        """Construct maps from registered maps.

        Combine the maps like an image stecker.
        The first map must be background map.

        Returns
        -------
        Medium
            A Medium instance with the constructed domain properties.

        """
        self._check_registered_domain()

        for map_name in ["c_map", "rho_map", "a_map", "a_power_map", "beta_map", "air_map"]:
            base_map_data = self.registered_domain_dict["background"][map_name].copy()
            for domain_name in self.registered_domain_dict:
                if domain_name == "background":
                    continue

                non_zero_index = np.where(
                    self.registered_domain_dict[domain_name]["geometry"] != 0,
                )
                base_map_data[non_zero_index] = self.registered_domain_dict[domain_name][map_name][
                    non_zero_index
                ]
            self.constructed_domain_dict[map_name] = base_map_data.copy()

        if self.ignore_non_linearity:
            self.constructed_domain_dict["beta_map"] = np.zeros_like(
                self.constructed_domain_dict["beta_map"],
            )

        base_geometry_data = np.zeros_like(self.registered_domain_dict["background"]["geometry"])
        for i, domain_name in enumerate(self.registered_domain_dict.keys()):
            geometry_non_zero_index = np.where(
                self.registered_domain_dict[domain_name]["geometry"] != 0,
            )
            base_geometry_data[geometry_non_zero_index] = i
        self.constructed_domain_dict["geometry"] = base_geometry_data.copy()
        return Medium(
            grid=self.grid,
            sound_speed=self.constructed_domain_dict["c_map"],
            density=self.constructed_domain_dict["rho_map"],
            alpha_coeff=self.constructed_domain_dict["a_map"],
            alpha_power=self.constructed_domain_dict["a_power_map"],
            beta=self.constructed_domain_dict["beta_map"],
            air_map=self.constructed_domain_dict["air_map"],
            path_relaxation_parameters_database=self.path_relaxation_parameters_database,
            n_relaxation_mechanisms=self.n_relaxation_mechanisms,
            attenuation_builder=self.attenuation_builder,
        )

    def plot_current_map(
        self,
        export_path: Path | str | None = Path("./temp/temp.png"),
        *,
        show: bool = False,
    ) -> None:
        """Plot the medium fields using matplotlib."""
        medium = self.run()
        medium.plot(
            export_path=export_path,
            show=show,
        )
