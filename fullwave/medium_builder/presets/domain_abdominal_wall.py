"""abdominal wall domain."""

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from fullwave import Grid
from fullwave.constants import MaterialProperties
from fullwave.medium_builder.domain import Domain
from fullwave.utils import check_functions
from fullwave.utils.numerical import matlab_interp2easy

logger = logging.getLogger("__main__." + __name__)


def _make_abdominal_property(
    grid: Grid,
    abdominal_wall_mat_path: Path = Path(
        "fullwave/medium_builder/presets/data/abdominal_wall/i2365f_etfw1.mat",
    ),
    crop_depth: float = 0.8e-2,
    start_depth: float = 0,
    background_domain_properties: str = "liver",
    *,
    apply_tissue_compression: bool = True,
    compression_ratio: float = 0.655,
    use_center_region: bool = True,
    skip_background_definition: bool = True,
    transducer_surface: NDArray[np.float64] | None = None,
) -> dict[str, np.ndarray]:
    mat_data = loadmat(abdominal_wall_mat_path)
    abdominal_wall_properties = mat_data["cut"].astype(float)

    # this is the pixel size in interpd Visual Human slice.
    dm = 0.33e-3 / 4

    compression_ratio = compression_ratio if apply_tissue_compression else 1

    interpolation_x = compression_ratio * dm / grid.dx
    interpolation_y = dm / grid.dy

    abdominal_wall_properties = matlab_interp2easy(
        abdominal_wall_properties,
        interpolation_x=interpolation_x,
        interpolation_y=interpolation_y,
    )

    crop_depth_index = round(crop_depth / grid.dx) - 1
    start_depth_index = round(start_depth / grid.dx)

    # axis definition: [axial, lateral], x-> depth, y-> width
    abdominal_wall_properties = abdominal_wall_properties[
        crop_depth_index : crop_depth_index + grid.nx,
        :,
    ]

    if use_center_region:
        offset = grid.ny % 2
        center_y = abdominal_wall_properties.shape[1] // 2
        abdominal_wall_properties = abdominal_wall_properties[
            : grid.nx,
            center_y - grid.ny // 2 : center_y + grid.ny // 2 + offset,
        ]
    assert abdominal_wall_properties.shape[1] == grid.ny

    base_map = np.zeros((grid.nx, grid.ny))

    if abdominal_wall_properties.shape[0] + start_depth_index > grid.nx:
        abdominal_wall_properties = abdominal_wall_properties[: grid.nx - start_depth_index, :]
    if transducer_surface is not None:
        for i in range(abdominal_wall_properties.shape[1]):
            does_transducer_surface_exist = i in transducer_surface[1]

            if does_transducer_surface_exist:
                j_indices = np.where(transducer_surface[1] == i)[0]
                transducer_surface_part = transducer_surface[:, j_indices]
                transducer_surface_value = transducer_surface_part[0, 0]
                if transducer_surface_value - 3 <= 0:
                    continue
                abdominal_wall_properties[transducer_surface_value - 3 :, i] = (
                    abdominal_wall_properties[
                        0 : -transducer_surface_value + 3,
                        i,
                    ]
                )
                abdominal_wall_properties[0 : transducer_surface_value - 3, i] = 0
            else:
                continue

    base_map[
        start_depth_index : start_depth_index + abdominal_wall_properties.shape[0],
        : abdominal_wall_properties.shape[1],
    ] = abdominal_wall_properties

    material_properties = MaterialProperties()

    density = np.ones_like(base_map) * -1
    sound_speed = np.ones_like(base_map) * -1
    alpha_coeff = np.ones_like(base_map) * -1
    alpha_power = np.ones_like(base_map) * -1
    beta = np.ones_like(base_map) * -1

    for i, tissue_name in enumerate(
        [
            background_domain_properties,
            "connective",
            "muscle",
            "fat",
            "connective",
            "connective",
        ],
    ):
        if i == 0 and skip_background_definition:
            continue
        target_index = np.where(base_map == i)
        density[target_index] = getattr(material_properties, tissue_name)["density"]
        sound_speed[target_index] = getattr(material_properties, tissue_name)["sound_speed"]
        alpha_coeff[target_index] = getattr(material_properties, tissue_name)["alpha_coeff"]
        alpha_power[target_index] = getattr(material_properties, tissue_name)["alpha_power"]
        beta[target_index] = getattr(material_properties, tissue_name)["beta"]

    # if use_smoothing:
    #     # use gaussian smoothing to smooth the abdominal wall
    #     sigma = (5 / 10) ** 2 * grid.ppw / 2
    #     rho_map_blurred = matlab_gaussian_filter(density, sigma=sigma)
    #     c_map_blurred = matlab_gaussian_filter(sound_speed, sigma=sigma)
    #     a_coeff_map_blurred = matlab_gaussian_filter(alpha_coeff, sigma=sigma)
    #     a_power_map_blurred = matlab_gaussian_filter(alpha_power, sigma=sigma)

    #     density = rho_map_blurred
    #     sound_speed = c_map_blurred
    #     alpha_coeff = a_coeff_map_blurred
    #     alpha_power = a_power_map_blurred

    air_map = np.zeros_like(base_map)
    return {
        "base_geometry": base_map > 0,
        "sound_speed": sound_speed,
        "density": density,
        "alpha_coeff": alpha_coeff,
        "alpha_power": alpha_power,
        "beta": beta,
        "air_map": air_map,
    }


class AbdominalWallDomain(Domain):
    """Represents the abdominal wall domain for simulation."""

    def __init__(
        self,
        grid: Grid,
        *,
        abdominal_wall_mat_path: Path = Path(__file__).parent.parent.parent
        / "medium_builder"
        / "presets"
        / "data"
        / "abdominal_wall"
        / "i2365f_etfw1.mat",
        crop_depth: float = 0.8e-2,
        start_depth: float = 0,
        name: str = "abdominal_wall",
        background_domain_properties: str = "water",
        material_properties: MaterialProperties | None = None,
        apply_tissue_compression: bool = True,
        use_center_region: bool = True,
        skip_background_definition: bool = True,
        path_relaxation_parameters_database: Path = Path(__file__).parent.parent.parent
        / "solver"
        / "bins"
        / "database"
        / "relaxation_params_database_num_relax=2_20251027_1437.mat",
        n_relaxation_mechanisms: int = 2,
        transducer_surface: NDArray[np.float64] | None = None,
    ) -> None:
        """Define AbdominalWallDomain instance.

        test

        Parameters
        ----------
        grid: Grid
            The grid instance.
        abdominal_wall_mat_path: Path, optional
            The path to the abdominal wall mat file.
            Defaults to
            Path("fullwave/medium_builder/presets/data/abdominal_wall/i2365f_etfw1.mat").
        crop_depth: float, optional
            how much you want to crop the abdominal wall from the top.
            Defaults to 0.8e-2.
        start_depth: float, optional
            it defines the place where the abdominal wall starts in the domain.
            The start depth.
            Defaults to 0.
        name: str, optional
            The name of the domain.
            Defaults to "abdominal_wall".
        background_domain_properties: str, optional
            The background domain properties.
            this value will be used to define the background domain properties.
            this is ignored if skip_background_definition is True.
            Defaults to "water".
        material_properties: MaterialProperties, optional
            Material properties to be used.
            Defaults to None.
        apply_tissue_compression: bool, optional
            if apply tissue compression emulation to the abdominal wall.
            Defaults to True.
        use_center_region: bool, optional
            if use the center region of the abdominal wall.
            Defaults to True.
        skip_background_definition: bool, optional
            if skip background definition.
            Defaults to True.
        path_relaxation_parameters_database: Path, optional
            The path to the relaxation parameters database.
            Defaults to
            Path(__file__).parent.parent / "solver" / "bins"
            / "database"
            / "relaxation_params_database_num_relax=2_20251027_1437.mat",
        n_relaxation_mechanisms: int, optional
            The number of relaxation mechanisms.
            Defaults to 4.
        transducer_surface: NDArray[np.float64] | None, optional
            The transducer surface.
            Defaults to None.
            the shape of this array should be (ny,).
            and each value contains the depth of the transducer surface at each lateral position.

        """
        self.material_properties = material_properties

        self.background_domain_properties = background_domain_properties
        self.crop_depth = crop_depth
        self.start_depth = start_depth
        self.abdominal_wall_mat_path = abdominal_wall_mat_path

        self.apply_tissue_compression = apply_tissue_compression
        self.skip_background_definition = skip_background_definition

        self.use_center_region = use_center_region

        check_functions.check_instance(grid, Grid)

        self.name = name
        self.grid = grid
        self.is_3d = grid.is_3d
        if self.is_3d:
            error_msg = "3D abdominal wall domain is not supported yet."
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
        self.path_relaxation_parameters_database = path_relaxation_parameters_database
        self.n_relaxation_mechanisms = n_relaxation_mechanisms
        self.transducer_surface = transducer_surface
        (
            self.base_geometry,
            self.sound_speed,
            self.density,
            self.alpha_coeff,
            self.alpha_power,
            self.beta,
            self.air,
        ) = self._setup_maps()

    def _setup_maps(
        self,
    ) -> tuple[np.ndarray, ...]:
        maps = _make_abdominal_property(
            grid=self.grid,
            abdominal_wall_mat_path=self.abdominal_wall_mat_path,
            crop_depth=self.crop_depth,
            start_depth=self.start_depth,
            background_domain_properties=self.background_domain_properties,
            apply_tissue_compression=self.apply_tissue_compression,
            use_center_region=self.use_center_region,
            skip_background_definition=self.skip_background_definition,
            transducer_surface=self.transducer_surface,
        )

        return (
            maps["base_geometry"],
            maps["sound_speed"],
            maps["density"],
            maps["alpha_coeff"],
            maps["alpha_power"],
            maps["beta"],
            maps["air_map"],
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
