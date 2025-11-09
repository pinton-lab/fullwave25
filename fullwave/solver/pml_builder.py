"""Perfectly Matched Layer (PML) setup for Fullwave."""

import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

import fullwave
from fullwave.solver.utils import initialize_relaxation_param_dict
from fullwave.utils import check_functions, plot_utils

logger = logging.getLogger("__main__." + __name__)


def _smooth_transition_function_part(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.where(x > 0, np.exp(-1 / (x + 1e-20)), 0)


def _smooth_transition_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return _smooth_transition_function_part(x) / (
        _smooth_transition_function_part(x) + _smooth_transition_function_part(1 - x)
    )


def _linear_transition_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return x


def _n_th_deg_polynomial_function(x: NDArray[np.float64], n: int = 2) -> NDArray[np.float64]:
    return x**n


def _cosine_transition_function(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (1 - np.cos(np.pi * x))


def _obtain_relax_var_rename_dict(
    n_relaxation_mechanisms: int,
    *,
    is_3d: bool = False,
    use_isotropic_relaxation: bool = False,
) -> dict:
    if use_isotropic_relaxation:
        rename_dict = {
            "kappa_x": "kappa_x2",
            "kappa_u": "kappa_x1",
        }

        for nu in range(1, n_relaxation_mechanisms + 1):
            rename_dict[f"d_u_nu{nu}"] = f"d_x1_nu{nu}"
            rename_dict[f"d_x_nu{nu}"] = f"d_x2_nu{nu}"

            rename_dict[f"alpha_u_nu{nu}"] = f"alpha_x1_nu{nu}"
            rename_dict[f"alpha_x_nu{nu}"] = f"alpha_x2_nu{nu}"
    else:
        rename_dict = {
            "kappa_x": "kappa_x2",
            "kappa_y": "kappa_x2",
            "kappa_u": "kappa_x1",
            "kappa_w": "kappa_x1",
        }
        if is_3d:
            rename_dict.update(
                {
                    "kappa_z": "kappa_x2",
                    "kappa_v": "kappa_x1",
                },
            )
        for nu in range(1, n_relaxation_mechanisms + 1):
            rename_dict[f"d_u_nu{nu}"] = f"d_x1_nu{nu}"
            rename_dict[f"d_w_nu{nu}"] = f"d_x1_nu{nu}"
            rename_dict[f"d_x_nu{nu}"] = f"d_x2_nu{nu}"
            rename_dict[f"d_y_nu{nu}"] = f"d_x2_nu{nu}"

            rename_dict[f"alpha_u_nu{nu}"] = f"alpha_x1_nu{nu}"
            rename_dict[f"alpha_w_nu{nu}"] = f"alpha_x1_nu{nu}"
            rename_dict[f"alpha_x_nu{nu}"] = f"alpha_x2_nu{nu}"
            rename_dict[f"alpha_y_nu{nu}"] = f"alpha_x2_nu{nu}"
            if is_3d:
                rename_dict[f"d_v_nu{nu}"] = f"d_x1_nu{nu}"
                rename_dict[f"d_z_nu{nu}"] = f"d_x2_nu{nu}"
                rename_dict[f"alpha_v_nu{nu}"] = f"alpha_x1_nu{nu}"
                rename_dict[f"alpha_z_nu{nu}"] = f"alpha_x2_nu{nu}"

    return rename_dict


@dataclass
class PMLBuilder:
    """Setup for Perfectly Matched Layers (PML) in fullwave simulations."""

    medium_org: fullwave.Medium
    source_org: fullwave.Source
    sensor_org: fullwave.Sensor

    m_spatial_order: int
    n_pml_layer: int
    n_relaxation: int
    n_transition_layer: int

    extended_grid: fullwave.Grid = field(init=False)
    extended_medium: fullwave.Medium | fullwave.MediumRelaxationMaps = field(init=False)
    extended_source: fullwave.Source = field(init=False)
    extended_sensor: fullwave.Sensor = field(init=False)

    pml_mask_x: NDArray[np.float64] = field(init=False)
    pml_mask_y: NDArray[np.float64] = field(init=False)

    def __init__(
        self,
        grid: fullwave.Grid,
        medium: fullwave.Medium,
        source: fullwave.Source,
        sensor: fullwave.Sensor,
        *,
        m_spatial_order: int = 8,
        n_pml_layer: int = 40,
        n_transition_layer: int = 40,
        use_isotropic_relaxation: bool = False,
        # pml_alpha_target: float = 1.1,
        # pml_alpha_power_target: float = 1.6,
        # pml_strength_factor: float = 2.0,
        # use_2_relax_mechanisms: bool = False,
    ) -> None:
        """Initialize the PMLSetup with the given medium, source, sensor, and PML parameters.

        Parameters
        ----------
        grid: fullwave.Grid
            The grid configuration.
        medium : fullwave.Medium)
            The medium relaxation maps.
        source : fullwave.Source
            The source configuration.
        sensor : fullwave.Sensor
            The sensor configuration.
        m_spatial_order : int, optional
            fullwave simulation's spatial order (default is 8).
            It depends on the fullwave simulation binary version.
            Fullwave simulation has 2M th order spatial accuracy and fourth order accuracy in time.
            see Pinton, G. (2021) http://arxiv.org/abs/2106.11476 for more detail.
        n_pml_layer : int, optional
            PML layer thickness (default is 40).
        n_transition_layer : int, optional
            Number of transition layers (default is 40).
        use_isotropic_relaxation : bool, optional
            Whether to use isotropic relaxation mechanisms for attenuation modeling
            to reduce memory usage while retaining accuracy.
            For 2D it will reduce the memory usage by approximately 15%.
            For 3D it will reduce the memory usage by approximately 25%.
            This option omits the anisotropic relaxation mechanisms to model the attenuation.
            We usually recommend using isotropic relaxation mechanisms
            unless the anisotropic attenuation is required for the simulation.

        """
        check_functions.check_instance(
            grid,
            fullwave.Grid,
        )
        check_functions.check_instance(
            medium,
            [fullwave.Medium, fullwave.MediumRelaxationMaps],
        )
        check_functions.check_instance(
            source,
            fullwave.Source,
        )
        check_functions.check_instance(
            sensor,
            fullwave.Sensor,
        )

        self.grid_org = grid
        self.medium_org = medium
        self.source_org = source
        self.sensor_org = sensor
        self.is_3d = grid.is_3d
        self.use_isotropic_relaxation = use_isotropic_relaxation

        self.m_spatial_order = m_spatial_order
        self.n_pml_layer = n_pml_layer
        self.n_transition_layer = n_transition_layer

        domain_size: tuple[float, ...]
        if self.is_3d:
            domain_size = (
                (self.medium_org.sound_speed.shape[0] + 2 * self.num_boundary_points)
                * self.grid_org.dx,
                (self.medium_org.sound_speed.shape[1] + 2 * self.num_boundary_points)
                * self.grid_org.dy,
                (self.medium_org.sound_speed.shape[2] + 2 * self.num_boundary_points)
                * self.grid_org.dz,
            )
        else:
            domain_size = (
                (self.medium_org.sound_speed.shape[0] + 2 * self.num_boundary_points)
                * self.grid_org.dx,
                (self.medium_org.sound_speed.shape[1] + 2 * self.num_boundary_points)
                * self.grid_org.dy,
            )

        logger.debug("building extended grid for pml...")
        self.extended_grid = fullwave.Grid(
            domain_size=domain_size,
            f0=self.grid_org.f0,
            duration=self.grid_org.duration,
            c0=self.grid_org.c0,
            ppw=self.grid_org.ppw,
            cfl=self.grid_org.cfl,
        )

        logger.debug("building extended medium for pml...")
        if isinstance(self.medium_org, fullwave.MediumRelaxationMaps):
            self.extended_medium = fullwave.MediumRelaxationMaps(
                grid=self.extended_grid,
                sound_speed=self._extend_map_for_pml(self.medium_org.sound_speed),
                density=self._extend_map_for_pml(self.medium_org.density),
                beta=self._extend_map_for_pml(self.medium_org.beta),
                relaxation_param_dict={
                    key: self._extend_map_for_pml(value)
                    for key, value in self.medium_org.relaxation_param_dict.items()
                },
                air_map=self._extend_map_for_pml(self.medium_org.air_map, fill_edge=False),
                n_relaxation_mechanisms=self.medium_org.n_relaxation_mechanisms,
            )
        else:
            self.extended_medium = fullwave.Medium(
                grid=self.extended_grid,
                sound_speed=self._extend_map_for_pml(self.medium_org.sound_speed),
                density=self._extend_map_for_pml(self.medium_org.density),
                beta=self._extend_map_for_pml(self.medium_org.beta),
                alpha_coeff=self._extend_map_for_pml(self.medium_org.alpha_coeff),
                alpha_power=self._extend_map_for_pml(self.medium_org.alpha_power),
                air_map=self._extend_map_for_pml(self.medium_org.air_map, fill_edge=False),
                n_relaxation_mechanisms=self.medium_org.n_relaxation_mechanisms,
                path_relaxation_parameters_database=self.medium_org.path_relaxation_parameters_database,
                attenuation_builder=self.medium_org.attenuation_builder,
            )

        logger.debug("building extended source for pml...")
        self.extended_source = fullwave.Source(
            p0=self.source_org.p0,
            mask=self._extend_map_for_pml(self.source_org.mask, fill_edge=False),
        )
        logger.debug("building extended sensor for pml...")
        self.extended_sensor = fullwave.Sensor(
            mask=self._extend_map_for_pml(self.sensor_org.mask, fill_edge=False),
            sampling_modulus_time=self.sensor_org.sampling_modulus_time,
        )
        if self.is_3d:
            self.pml_mask_x, self.pml_mask_y, self.pml_mask_z = self._localize_pml_region()
        else:
            self.pml_mask_x, self.pml_mask_y = self._localize_pml_region()

        self.pml_layer_m = self.extended_grid.dx * self.n_pml_layer
        self.transition_layer_m = self.extended_grid.dx * self.n_transition_layer

        self.n_polynomial = 2
        self.theoritical_reflection_coefficient = 10 ** (-30)

        if self.n_pml_layer == 0:
            self.n_transition_layer = 0

    # ---
    @cached_property
    def num_boundary_points(self) -> int:
        """Returns the number of the boundary points.

        Number of PML layer and ghost cells.
        """
        return self.n_transition_layer + self.n_pml_layer + self.m_spatial_order

    @cached_property
    def nx(self) -> int:
        """Returns the number of grid points in x-direction."""
        return self.extended_grid.nx

    @cached_property
    def ny(self) -> int:
        """Returns the number of grid points in y-direction."""
        return self.extended_grid.ny

    @cached_property
    def nz(self) -> int:
        """Returns the number of grid points in y-direction."""
        return self.extended_grid.nz

    @cached_property
    def nt(self) -> int:
        """Returns the number of time steps."""
        return self.extended_grid.nt

    @cached_property
    def n_sources(self) -> int:
        """Return the number of sources."""
        return self.extended_source.n_sources

    @cached_property
    def n_sensors(self) -> int:
        """Return the number of sources."""
        return self.extended_sensor.n_sensors

    @cached_property
    def n_air(self) -> int:
        """Return the number of air coordinates."""
        return self.extended_medium.n_air

    @cached_property
    def n_coords_zero(self) -> int:
        """Return the number of air coordinates.

        (alias for self.n_air)
        """
        return self.n_air

    def _extend_map_for_pml(  # noqa: PLR0915
        self,
        input_map: NDArray[np.float64 | np.int64 | np.bool],
        *,
        fill_edge: bool = True,
    ) -> NDArray[np.float64 | np.int64 | np.bool]:
        output_map: NDArray[np.float64 | np.int64 | np.bool]
        if self.is_3d:
            output_map = np.zeros(
                (
                    input_map.shape[0] + 2 * self.num_boundary_points,
                    input_map.shape[1] + 2 * self.num_boundary_points,
                    input_map.shape[2] + 2 * self.num_boundary_points,
                ),
            )
            # center
            output_map[
                self.num_boundary_points : -self.num_boundary_points,
                self.num_boundary_points : -self.num_boundary_points,
                self.num_boundary_points : -self.num_boundary_points,
            ] = input_map
            # edges
            if fill_edge:
                output_map[
                    self.num_boundary_points : -self.num_boundary_points,
                    : self.num_boundary_points,
                    self.num_boundary_points : -self.num_boundary_points,
                ] = input_map[:, [0], :]
                output_map[
                    self.num_boundary_points : -self.num_boundary_points,
                    -self.num_boundary_points :,
                    self.num_boundary_points : -self.num_boundary_points,
                ] = input_map[:, [-1], :]

                output_map[
                    : self.num_boundary_points,
                    self.num_boundary_points : -self.num_boundary_points,
                    self.num_boundary_points : -self.num_boundary_points,
                ] = input_map[[0], :, :]
                output_map[
                    -self.num_boundary_points :,
                    self.num_boundary_points : -self.num_boundary_points,
                    self.num_boundary_points : -self.num_boundary_points,
                ] = input_map[[-1], :, :]

                output_map[
                    self.num_boundary_points : -self.num_boundary_points,
                    self.num_boundary_points : -self.num_boundary_points,
                    : self.num_boundary_points,
                ] = input_map[:, :, [0]]
                output_map[
                    self.num_boundary_points : -self.num_boundary_points,
                    self.num_boundary_points : -self.num_boundary_points,
                    -self.num_boundary_points :,
                ] = input_map[:, :, [-1]]

                # corners
                output_map[
                    : self.num_boundary_points,
                    : self.num_boundary_points,
                    : self.num_boundary_points,
                ] = input_map[
                    0,
                    0,
                    0,
                ]

                output_map[
                    -self.num_boundary_points :,
                    : self.num_boundary_points,
                    : self.num_boundary_points,
                ] = input_map[
                    -1,
                    0,
                    0,
                ]
                output_map[
                    : self.num_boundary_points,
                    -self.num_boundary_points :,
                    : self.num_boundary_points,
                ] = input_map[
                    0,
                    -1,
                    0,
                ]
                output_map[
                    : self.num_boundary_points,
                    : self.num_boundary_points,
                    -self.num_boundary_points :,
                ] = input_map[
                    0,
                    0,
                    -1,
                ]
                output_map[
                    -self.num_boundary_points :,
                    -self.num_boundary_points :,
                    : self.num_boundary_points,
                ] = input_map[
                    -1,
                    -1,
                    0,
                ]
                output_map[
                    : self.num_boundary_points,
                    -self.num_boundary_points :,
                    -self.num_boundary_points :,
                ] = input_map[
                    0,
                    -1,
                    -1,
                ]
                output_map[
                    -self.num_boundary_points :,
                    : self.num_boundary_points,
                    -self.num_boundary_points :,
                ] = input_map[
                    -1,
                    0,
                    -1,
                ]
                output_map[
                    -self.num_boundary_points :,
                    -self.num_boundary_points :,
                    -self.num_boundary_points :,
                ] = input_map[
                    -1,
                    -1,
                    -1,
                ]
                # ---
                output_map[
                    : self.num_boundary_points,
                    : self.num_boundary_points,
                    : self.num_boundary_points,
                ] = input_map[
                    0,
                    0,
                    0,
                ]
                for i in range(input_map.shape[0]):
                    output_map[
                        self.num_boundary_points + i,
                        : self.num_boundary_points,
                        : self.num_boundary_points,
                    ] = input_map[
                        i,
                        [0],
                        [0],
                    ]
                    output_map[
                        self.num_boundary_points + i,
                        : self.num_boundary_points,
                        -self.num_boundary_points :,
                    ] = input_map[
                        i,
                        [0],
                        [-1],
                    ]
                    output_map[
                        -self.num_boundary_points - (i + 1),
                        : self.num_boundary_points,
                        : self.num_boundary_points,
                    ] = input_map[
                        -(i + 1),
                        [0],
                        [0],
                    ]
                    output_map[
                        -self.num_boundary_points - (i + 1),
                        : self.num_boundary_points,
                        -self.num_boundary_points :,
                    ] = input_map[
                        -(i + 1),
                        [0],
                        [-1],
                    ]
                    output_map[
                        self.num_boundary_points + i,
                        -self.num_boundary_points :,
                        : self.num_boundary_points,
                    ] = input_map[
                        i,
                        [-1],
                        [0],
                    ]

                    output_map[
                        -self.num_boundary_points - (i + 1),
                        -self.num_boundary_points :,
                        : self.num_boundary_points,
                    ] = input_map[
                        -(i + 1),
                        [-1],
                        [0],
                    ]
                    output_map[
                        -self.num_boundary_points - (i + 1),
                        -self.num_boundary_points :,
                        -self.num_boundary_points :,
                    ] = input_map[
                        -(i + 1),
                        [-1],
                        [-1],
                    ]
                for i in range(input_map.shape[1]):
                    output_map[
                        : self.num_boundary_points,
                        self.num_boundary_points + i,
                        : self.num_boundary_points,
                    ] = input_map[
                        [0],
                        i,
                        [0],
                    ]
                    output_map[
                        : self.num_boundary_points,
                        self.num_boundary_points + i,
                        -self.num_boundary_points :,
                    ] = input_map[
                        [0],
                        i,
                        [-1],
                    ]
                    output_map[
                        : self.num_boundary_points,
                        -self.num_boundary_points - (i + 1),
                        : self.num_boundary_points,
                    ] = input_map[
                        [0],
                        -(i + 1),
                        [0],
                    ]
                    output_map[
                        : self.num_boundary_points,
                        -self.num_boundary_points - (i + 1),
                        -self.num_boundary_points :,
                    ] = input_map[
                        [0],
                        -(i + 1),
                        [-1],
                    ]
                    output_map[
                        -self.num_boundary_points :,
                        self.num_boundary_points + i,
                        : self.num_boundary_points,
                    ] = input_map[
                        [-1],
                        i,
                        [0],
                    ]

                    output_map[
                        -self.num_boundary_points :,
                        -self.num_boundary_points - (i + 1),
                        : self.num_boundary_points,
                    ] = input_map[
                        [-1],
                        -(i + 1),
                        [0],
                    ]
                    output_map[
                        -self.num_boundary_points :,
                        -self.num_boundary_points - (i + 1),
                        -self.num_boundary_points :,
                    ] = input_map[
                        [-1],
                        -(i + 1),
                        [-1],
                    ]
                for i in range(input_map.shape[2]):
                    output_map[
                        : self.num_boundary_points,
                        : self.num_boundary_points,
                        self.num_boundary_points + i,
                    ] = input_map[
                        [0],
                        [0],
                        i,
                    ]
                    output_map[
                        : self.num_boundary_points,
                        -self.num_boundary_points :,
                        self.num_boundary_points + i,
                    ] = input_map[
                        [0],
                        [-1],
                        i,
                    ]
                    output_map[
                        : self.num_boundary_points,
                        : self.num_boundary_points,
                        -self.num_boundary_points - (i + 1),
                    ] = input_map[
                        [0],
                        [0],
                        -(i + 1),
                    ]
                    output_map[
                        : self.num_boundary_points,
                        -self.num_boundary_points :,
                        -self.num_boundary_points - (i + 1),
                    ] = input_map[
                        [0],
                        [-1],
                        -(i + 1),
                    ]
                    output_map[
                        -self.num_boundary_points :,
                        : self.num_boundary_points,
                        self.num_boundary_points + i,
                    ] = input_map[
                        [-1],
                        [0],
                        i,
                    ]

                    output_map[
                        -self.num_boundary_points :,
                        : self.num_boundary_points,
                        -self.num_boundary_points - (i + 1),
                    ] = input_map[
                        [-1],
                        [0],
                        -(i + 1),
                    ]
                    output_map[
                        -self.num_boundary_points :,
                        -self.num_boundary_points :,
                        -self.num_boundary_points - (i + 1),
                    ] = input_map[
                        [-1],
                        [-1],
                        -(i + 1),
                    ]
        else:
            output_map = np.zeros(
                (
                    input_map.shape[0] + 2 * self.num_boundary_points,
                    input_map.shape[1] + 2 * self.num_boundary_points,
                ),
            )
            # center
            output_map[
                self.num_boundary_points : -self.num_boundary_points,
                self.num_boundary_points : -self.num_boundary_points,
            ] = input_map

            # edges
            if fill_edge:
                output_map[
                    self.num_boundary_points : -self.num_boundary_points,
                    : self.num_boundary_points,
                ] = input_map[:, [0]]
                output_map[
                    self.num_boundary_points : -self.num_boundary_points,
                    -self.num_boundary_points :,
                ] = input_map[:, [-1]]

                output_map[
                    : self.num_boundary_points,
                    self.num_boundary_points : -self.num_boundary_points,
                ] = input_map[[0], :]
                output_map[
                    -self.num_boundary_points :,
                    self.num_boundary_points : -self.num_boundary_points,
                ] = input_map[[-1], :]

                # corners
                output_map[: self.num_boundary_points, : self.num_boundary_points] = input_map[0, 0]
                output_map[-self.num_boundary_points :, -self.num_boundary_points :] = input_map[
                    -1,
                    -1,
                ]
                output_map[-self.num_boundary_points :, : self.num_boundary_points] = input_map[
                    -1,
                    0,
                ]
                output_map[: self.num_boundary_points, -self.num_boundary_points :] = input_map[
                    0,
                    -1,
                ]

        return output_map

    def _extend_relaxation_param_dict(
        self,
        relaxation_param_dict: dict[str, NDArray[np.float64 | np.int64 | np.bool]],
    ) -> dict[str, NDArray[np.float64 | np.int64 | np.bool]]:
        output_dict = {}
        for key, value in relaxation_param_dict.items():
            output_dict[key] = self._extend_map_for_pml(value)
        return output_dict

    def _localize_pml_region(self) -> tuple[NDArray[np.float64], ...]:
        pml_mask_x: NDArray[np.float64]
        pml_mask_y: NDArray[np.float64]
        pml_mask_z: NDArray[np.float64]
        if self.is_3d:
            n_x_extended, n_y_extended, n_z_extended = self.extended_medium.sound_speed.shape

            pml_mask_x = np.zeros((n_x_extended, n_y_extended, n_z_extended))
            pml_mask_y = np.zeros((n_x_extended, n_y_extended, n_z_extended))
            pml_mask_z = np.zeros((n_x_extended, n_y_extended, n_z_extended))
            for i in range(self.n_pml_layer):
                pml_mask_x[
                    i + (n_x_extended - self.m_spatial_order - self.n_pml_layer),
                    :,
                    :,
                ] = i / self.n_pml_layer

                pml_mask_x[self.m_spatial_order + self.n_pml_layer - i - 1, :, :] = (
                    i / self.n_pml_layer
                )

                pml_mask_y[
                    :,
                    i + (n_y_extended - self.m_spatial_order - self.n_pml_layer),
                    :,
                ] = i / self.n_pml_layer

                pml_mask_y[:, self.m_spatial_order + self.n_pml_layer - i - 1, :] = (
                    i / self.n_pml_layer
                )

                pml_mask_z[
                    :,
                    :,
                    i + (n_z_extended - self.m_spatial_order - self.n_pml_layer),
                ] = i / self.n_pml_layer

                pml_mask_z[:, :, self.m_spatial_order + self.n_pml_layer - i - 1] = (
                    i / self.n_pml_layer
                )

            pml_mask_x[0 : self.m_spatial_order, :, :] = 1
            pml_mask_x[n_x_extended - self.m_spatial_order : n_x_extended, :, :] = 1

            pml_mask_y[:, 0 : self.m_spatial_order, :] = 1
            pml_mask_y[:, n_y_extended - self.m_spatial_order : n_y_extended, :] = 1

            pml_mask_z[:, :, 0 : self.m_spatial_order] = 1
            pml_mask_z[:, :, n_z_extended - self.m_spatial_order : n_z_extended] = 1
            return pml_mask_x, pml_mask_y, pml_mask_z

        n_x_extended, n_y_extended = self.extended_medium.sound_speed.shape

        pml_mask_x = np.zeros((n_x_extended, n_y_extended))
        pml_mask_y = np.zeros((n_x_extended, n_y_extended))

        for i in range(self.n_pml_layer):
            pml_mask_x[
                i + (n_x_extended - self.m_spatial_order - self.n_pml_layer),
                :,
            ] = i / self.n_pml_layer

            pml_mask_x[self.m_spatial_order + self.n_pml_layer - i - 1, :] = i / self.n_pml_layer

            pml_mask_y[
                :,
                i + (n_y_extended - self.m_spatial_order - self.n_pml_layer),
            ] = i / self.n_pml_layer

            pml_mask_y[:, self.m_spatial_order + self.n_pml_layer - i - 1] = i / self.n_pml_layer

        pml_mask_x[0 : self.m_spatial_order, :] = 1
        pml_mask_x[n_x_extended - self.m_spatial_order : n_x_extended, :] = 1

        pml_mask_y[:, 0 : self.m_spatial_order] = 1
        pml_mask_y[:, n_y_extended - self.m_spatial_order : n_y_extended] = 1

        return pml_mask_x, pml_mask_y

    @staticmethod
    def _calc_a_and_b(
        d_x: NDArray[np.float64] | float,
        kappa_x: NDArray[np.float64] | float,
        alpha_x: NDArray[np.float64] | float,
        dt: NDArray[np.float64] | float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # function [a b] = ab(dx,kappax,alphax,dT)
        d_x = np.array(d_x)
        kappa_x = np.array(kappa_x)
        alpha_x = np.array(alpha_x)
        dt = np.array(dt)

        b = np.exp(-(d_x / kappa_x + alpha_x) * dt)
        eps = 1e-10
        a = d_x / (kappa_x * (d_x + kappa_x * alpha_x) + eps) * (b - 1)
        return a, b

    def run(self, *, use_pml: bool = True) -> fullwave.MediumRelaxationMaps:
        """Generate perfect matched layer (PML) relaxation parameters.

        It generates the relaxation parameters
        for the PML region considering the given medium and PML parameters.

        Returns
        -------
        Medium
            A Medium instance with the constructed domain properties.

        """
        if use_pml:
            extended_medium: fullwave.MediumRelaxationMaps = self.extended_medium.build()
            if self.is_3d:
                return self._apply_pml_3d(
                    extended_medium=extended_medium,
                    theoritical_reflection_coefficient=self.theoritical_reflection_coefficient,
                    n_polynomial=self.n_polynomial,
                )

            return self._apply_pml(
                extended_medium=extended_medium,
                theoritical_reflection_coefficient=self.theoritical_reflection_coefficient,
                n_polynomial=self.n_polynomial,
            )

        extended_medium: fullwave.MediumRelaxationMaps = self.extended_medium.build()
        return extended_medium

    def _apply_pml(
        self,
        extended_medium: fullwave.MediumRelaxationMaps,
        theoritical_reflection_coefficient: float,
        n_polynomial: float,
    ) -> fullwave.MediumRelaxationMaps:
        """Apply PML to the extended medium relaxation parameters.

        ref: Komatitsch, D., & Martin, R. (2007).
        An unsplit convolutional perfectly matched layer improved
        at grazing incidence for the seismic wave equation.
        Geophysics, 72(5), SM155-SM167. https://doi.org/10.1190/1.2757586

        Parameters
        ----------
        extended_medium : fullwave.MediumRelaxationMaps
            The extended medium relaxation parameters.
        n_polynomial : float
            The polynomial order for the PML damping parameter.
            it changes the transition function shape from the medium to the PML.
        theoritical_reflection_coefficient : float
            The theoretical reflection coefficient for the PML.
            it changes the PML strength. it gets unstable if it is too low.

        Returns
        -------
        fullwave.MediumRelaxationMaps
            The extended medium relaxation parameters with PML applied.

        """
        # alpha=0 and d=0 will make a and b in the PML be 0
        # this procedure shrinks the multiple relaxation mechanisms to a single one
        alpha_target_pml = 0
        alpha_target_higher_nu = 0
        d_target_higher_nu = 0

        # see Komatitsch, D., & Martin, R. (2007), SM160
        d_target_pml = (
            -(n_polynomial + 1)
            * self.extended_grid.c0
            * np.log(theoritical_reflection_coefficient)
            / (2 * (self.pml_layer_m + self.transition_layer_m))
            # / (2 * (self.pml_layer_m))
        )
        # alpha_pml_entrance = np.pi * self.extended_grid.f0

        out_dict = {}
        relaxation_param_dict = extended_medium.relaxation_param_dict
        rename_dict = _obtain_relax_var_rename_dict(
            n_relaxation_mechanisms=self.extended_medium.n_relaxation_mechanisms,
            is_3d=self.is_3d,
            use_isotropic_relaxation=self.use_isotropic_relaxation,
        )
        for key_fw2, key_py in rename_dict.items():
            if key_fw2 in ["kappa_x", "kappa_u", "kappa_y", "kappa_w"]:
                out_dict[key_fw2] = relaxation_param_dict[key_py].copy()
            elif (
                ("alpha_u_nu" in key_fw2 and "nu1" not in key_fw2)
                or ("alpha_x_nu" in key_fw2 and "nu1" not in key_fw2)
                or ("alpha_w_nu" in key_fw2 and "nu1" not in key_fw2)
                or ("alpha_y_nu" in key_fw2 and "nu1" not in key_fw2)
            ):
                # out_dict[key_fw2] = relaxation_param_dict[key_py].copy()
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    relaxation_param_dict[key_py].copy(),
                    value_target=alpha_target_higher_nu,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=0,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=self.is_3d,
                )
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    out_dict[key_fw2],
                    # relaxation_param_dict[key_py].copy(),
                    value_target=alpha_target_higher_nu,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=1,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=self.is_3d,
                )
            elif (
                ("d_u_nu" in key_fw2 and "nu1" not in key_fw2)
                or ("d_x_nu" in key_fw2 and "nu1" not in key_fw2)
                or ("d_w_nu" in key_fw2 and "nu1" not in key_fw2)
                or ("d_y_nu" in key_fw2 and "nu1" not in key_fw2)
            ):
                # out_dict[key_fw2] = relaxation_param_dict[key_py].copy()
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    relaxation_param_dict[key_py].copy(),
                    value_target=d_target_higher_nu,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=0,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=self.is_3d,
                )
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    out_dict[key_fw2],
                    # relaxation_param_dict[key_py].copy(),
                    value_target=d_target_higher_nu,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=1,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=self.is_3d,
                )
            elif (
                ("alpha_u_nu" in key_fw2 and "nu1" in key_fw2)
                or ("alpha_x_nu" in key_fw2 and "nu1" in key_fw2)
                or ("alpha_w_nu" in key_fw2 and "nu1" in key_fw2)
                or ("alpha_y_nu" in key_fw2 and "nu1" in key_fw2)
            ):
                # out_dict[key_fw2] = relaxation_param_dict[key_py].copy()
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    relaxation_param_dict[key_py].copy(),
                    value_target=alpha_target_pml,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=0,
                    transition_type="linear",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=self.is_3d,
                )
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    out_dict[key_fw2],
                    # relaxation_param_dict[key_py].copy(),
                    value_target=alpha_target_pml,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=1,
                    transition_type="linear",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=self.is_3d,
                )
            elif (
                ("d_u_nu" in key_fw2 and "nu1" in key_fw2)
                or ("d_x_nu" in key_fw2 and "nu1" in key_fw2)
                or ("d_w_nu" in key_fw2 and "nu1" in key_fw2)
                or ("d_y_nu" in key_fw2 and "nu1" in key_fw2)
            ):
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    relaxation_param_dict[key_py].copy(),
                    value_target=d_target_pml,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=0,
                    n_polynomial=n_polynomial,
                    transition_type="polynomial",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=self.is_3d,
                )
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    out_dict[key_fw2],
                    # relaxation_param_dict[key_py].copy(),
                    value_target=d_target_pml,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=1,
                    n_polynomial=n_polynomial,
                    transition_type="polynomial",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=self.is_3d,
                )

        axis_list = ["u", "x"] if self.use_isotropic_relaxation else ["u", "w", "x", "y"]
        for nu in range(1, extended_medium.n_relaxation_mechanisms + 1):
            for axis in axis_list:
                (
                    out_dict[f"a_pml_{axis}{nu}"],
                    out_dict[f"b_pml_{axis}{nu}"],
                ) = self._calc_a_and_b(
                    d_x=out_dict[f"d_{axis}_nu{nu}"],
                    kappa_x=out_dict[f"kappa_{axis}"],
                    alpha_x=out_dict[f"alpha_{axis}_nu{nu}"],
                    dt=extended_medium.grid.dt,
                )

        extended_medium.relaxation_param_dict_for_fw2.update(
            out_dict,
        )

        return extended_medium

    def _apply_pml_3d(
        self,
        extended_medium: fullwave.MediumRelaxationMaps,
        theoritical_reflection_coefficient: float,
        n_polynomial: float,
    ) -> fullwave.MediumRelaxationMaps:
        """Apply PML to the extended medium relaxation parameters.

        ref: Komatitsch, D., & Martin, R. (2007).
        An unsplit convolutional perfectly matched layer improved
        at grazing incidence for the seismic wave equation.
        Geophysics, 72(5), SM155-SM167. https://doi.org/10.1190/1.2757586

        Parameters
        ----------
        extended_medium : fullwave.MediumRelaxationMaps
            The extended medium relaxation parameters.
        n_polynomial : float
            The polynomial order for the PML damping parameter.
            it changes the transition function shape from the medium to the PML.
        theoritical_reflection_coefficient : float
            The theoretical reflection coefficient for the PML.
            it changes the PML strength. it gets unstable if it is too low.

        Returns
        -------
        fullwave.MediumRelaxationMaps
            The extended medium relaxation parameters with PML applied.

        """
        # alpha=0 and d=0 will make a and b in the PML be 0
        # this procedure shrinks the multiple relaxation mechanisms to a single one
        alpha_target_pml = 0
        alpha_target_higher_nu = 0
        d_target_higher_nu = 0

        # see Komatitsch, D., & Martin, R. (2007), SM160
        d_target_pml = (
            -(n_polynomial + 1)
            * self.extended_grid.c0
            * np.log(theoritical_reflection_coefficient)
            / (2 * (self.pml_layer_m + self.transition_layer_m))
            # / (2 * self.pml_layer_m)
        )

        out_dict = {}
        relaxation_param_dict = extended_medium.relaxation_param_dict
        rename_dict = _obtain_relax_var_rename_dict(
            n_relaxation_mechanisms=self.extended_medium.n_relaxation_mechanisms,
            is_3d=self.is_3d,
            use_isotropic_relaxation=self.use_isotropic_relaxation,
        )
        for key_fw2, key_py in rename_dict.items():
            if (
                key_fw2 in ["kappa_x", "kappa_u"]
                or key_fw2 in ["kappa_y", "kappa_v"]
                or key_fw2 in ["kappa_z", "kappa_w"]
            ):
                out_dict[key_fw2] = relaxation_param_dict[key_py].copy()
            elif (
                ("alpha_u_nu" in key_fw2 and "nu1" not in key_fw2)
                or ("alpha_v_nu" in key_fw2 and "nu1" not in key_fw2)
                or ("alpha_w_nu" in key_fw2 and "nu1" not in key_fw2)
                or ("alpha_x_nu" in key_fw2 and "nu1" not in key_fw2)
                or ("alpha_y_nu" in key_fw2 and "nu1" not in key_fw2)
                or ("alpha_z_nu" in key_fw2 and "nu1" not in key_fw2)
            ):
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    relaxation_param_dict[key_py].copy(),
                    value_target=alpha_target_higher_nu,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=0,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=self.is_3d,
                )
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    out_dict[key_fw2],
                    value_target=alpha_target_higher_nu,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=1,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=self.is_3d,
                )
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    out_dict[key_fw2],
                    value_target=alpha_target_higher_nu,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=2,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=self.is_3d,
                )
            elif (
                ("d_u_nu" in key_fw2 and "nu1" not in key_fw2)
                or ("d_v_nu" in key_fw2 and "nu1" not in key_fw2)
                or ("d_w_nu" in key_fw2 and "nu1" not in key_fw2)
                or ("d_x_nu" in key_fw2 and "nu1" not in key_fw2)
                or ("d_y_nu" in key_fw2 and "nu1" not in key_fw2)
                or ("d_z_nu" in key_fw2 and "nu1" not in key_fw2)
            ):
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    relaxation_param_dict[key_py].copy(),
                    value_target=d_target_higher_nu,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=0,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=self.is_3d,
                )
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    out_dict[key_fw2],
                    value_target=d_target_higher_nu,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=1,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=self.is_3d,
                )
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    out_dict[key_fw2],
                    value_target=d_target_higher_nu,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=2,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=self.is_3d,
                )
            elif (
                ("alpha_u_nu" in key_fw2 and "nu1" in key_fw2)
                or ("alpha_v_nu" in key_fw2 and "nu1" in key_fw2)
                or ("alpha_w_nu" in key_fw2 and "nu1" in key_fw2)
                or ("alpha_x_nu" in key_fw2 and "nu1" in key_fw2)
                or ("alpha_y_nu" in key_fw2 and "nu1" in key_fw2)
                or ("alpha_z_nu" in key_fw2 and "nu1" in key_fw2)
            ):
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    relaxation_param_dict[key_py].copy(),
                    value_target=alpha_target_pml,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=0,
                    transition_type="linear",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=self.is_3d,
                )
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    out_dict[key_fw2],
                    value_target=alpha_target_pml,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=1,
                    transition_type="linear",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=self.is_3d,
                )
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    out_dict[key_fw2],
                    value_target=alpha_target_pml,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=2,
                    transition_type="linear",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=self.is_3d,
                )
            elif (
                ("d_u_nu" in key_fw2 and "nu1" in key_fw2)
                or ("d_v_nu" in key_fw2 and "nu1" in key_fw2)
                or ("d_w_nu" in key_fw2 and "nu1" in key_fw2)
                or ("d_x_nu" in key_fw2 and "nu1" in key_fw2)
                or ("d_y_nu" in key_fw2 and "nu1" in key_fw2)
                or ("d_z_nu" in key_fw2 and "nu1" in key_fw2)
            ):
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    relaxation_param_dict[key_py].copy(),
                    value_target=d_target_pml,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=0,
                    n_polynomial=n_polynomial,
                    transition_type="polynomial",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=self.is_3d,
                )
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    out_dict[key_fw2],
                    value_target=d_target_pml,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=1,
                    n_polynomial=n_polynomial,
                    transition_type="polynomial",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=self.is_3d,
                )
                out_dict[key_fw2] = self._apply_transition_and_pml(
                    out_dict[key_fw2],
                    value_target=d_target_pml,
                    array_shape=relaxation_param_dict[key_py].shape,
                    axis=2,
                    n_polynomial=n_polynomial,
                    transition_type="polynomial",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=self.is_3d,
                )

        axis_list = ["u", "x"] if self.use_isotropic_relaxation else ["u", "v", "w", "x", "y", "z"]

        for nu in range(1, extended_medium.n_relaxation_mechanisms + 1):
            for axis in axis_list:
                (
                    out_dict[f"a_pml_{axis}{nu}"],
                    out_dict[f"b_pml_{axis}{nu}"],
                ) = self._calc_a_and_b(
                    d_x=out_dict[f"d_{axis}_nu{nu}"],
                    kappa_x=out_dict[f"kappa_{axis}"],
                    alpha_x=out_dict[f"alpha_{axis}_nu{nu}"],
                    dt=extended_medium.grid.dt,
                )

        extended_medium.relaxation_param_dict_for_fw2.update(
            out_dict,
        )

        return extended_medium

    def _apply_transition_and_pml(  # noqa: PLR0912 C901, PLR0915
        self,
        input_array: NDArray[np.float64],
        value_target: float,
        array_shape: tuple[int, ...],
        axis: int = 0,
        *,
        transition_type: str = "smooth",
        n_polynomial: float = 2,
        transit_within_transition_layer: bool = False,
        transit_within_pml_layer: bool = False,
        disable_the_transition_and_pml: bool = False,
        is_3d: bool = False,
    ) -> NDArray[np.float64]:
        if transit_within_transition_layer and transit_within_pml_layer:
            error_msg = (
                "Both transit_within_transition_layer and transit_within_pml_layer "
                "cannot be True at the same time."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if disable_the_transition_and_pml:
            return input_array

        if transit_within_transition_layer and self.n_transition_layer == 0:
            error_msg = (
                "Transition layer is not defined. "
                "Set transit_within_transition_layer to False or define n_transition_layer."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if transit_within_transition_layer:
            layer_thickness = self.n_transition_layer
            layer_offset = self.n_pml_layer
        elif transit_within_pml_layer:
            layer_thickness = self.n_pml_layer
            layer_offset = 0
        else:
            layer_thickness = self.n_pml_layer + self.n_transition_layer
            layer_offset = 0

        if transition_type == "smooth":
            transition_function = _smooth_transition_function(
                np.linspace(
                    0,
                    1,
                    layer_thickness + 1,
                ),
            )
        elif transition_type == "linear":
            transition_function = _linear_transition_function(
                np.linspace(
                    0,
                    1,
                    layer_thickness + 1,
                ),
            )
        elif transition_type == "polynomial":
            transition_function = _n_th_deg_polynomial_function(
                np.linspace(
                    0,
                    1,
                    layer_thickness + 1,
                ),
                n=n_polynomial,
            )
        elif transition_type == "cosine":
            transition_function = _cosine_transition_function(
                np.linspace(
                    0,
                    1,
                    layer_thickness + 1,
                ),
            )
        else:
            error_msg = (
                f"Invalid transition type: {transition_type}. "
                "Choose from 'smooth', 'linear', or 'polynomial'."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        n_axis_extended = array_shape[axis]

        if axis == 0:
            input_array[: self.m_spatial_order + layer_offset + layer_thickness] = value_target
            input_array[
                n_axis_extended - self.m_spatial_order - layer_thickness - layer_offset :,
            ] = value_target
            if is_3d:
                up_start = self.m_spatial_order + layer_offset - 1
                up_end = self.m_spatial_order + layer_offset + layer_thickness
                up_slice = slice(up_start, up_end)

                down_start = (
                    n_axis_extended - self.m_spatial_order - layer_thickness - layer_offset - 1
                )
                down_end = n_axis_extended - self.m_spatial_order - layer_offset
                down_slice = slice(down_start, down_end)

                # fetch the "mid" and "down" face values and expand dims for broadcasting
                up_vals = input_array[up_end, :, :][None, :, :]
                down_vals = input_array[down_start, :, :][None, :, :]

                # top transition (use reversed transition function)
                input_array[up_slice, :, :] = up_vals - transition_function[::-1][
                    :,
                    None,
                    None,
                ] * (up_vals - value_target)

                # bottom transition (forward transition function)
                input_array[down_slice, :, :] = down_vals - transition_function[:, None, None] * (
                    down_vals - value_target
                )
            else:
                up_start = self.m_spatial_order + layer_offset - 1
                up_end = self.m_spatial_order + layer_offset + layer_thickness
                up_slice = slice(up_start, up_end)

                down_start = (
                    n_axis_extended - self.m_spatial_order - layer_thickness - layer_offset - 1
                )
                down_end = n_axis_extended - self.m_spatial_order - layer_offset
                down_slice = slice(down_start, down_end)

                # fetch the "mid" and "down" face values and expand dims for broadcasting
                up_vals = input_array[up_end, :][None, :]
                down_vals = input_array[down_start, :][None, :]

                # top transition (use reversed transition function)
                input_array[up_slice, :] = up_vals - transition_function[::-1][:, None] * (
                    up_vals - value_target
                )

                # bottom transition (forward transition function)
                input_array[down_slice, :] = down_vals - transition_function[:, None] * (
                    down_vals - value_target
                )
        elif axis == 1:
            input_array[:, : self.m_spatial_order + layer_offset + layer_thickness] = value_target
            input_array[
                :,
                n_axis_extended - self.m_spatial_order - layer_thickness - layer_offset :,
            ] = value_target
            if is_3d:
                up_start = self.m_spatial_order + layer_offset - 1
                up_end = self.m_spatial_order + layer_offset + layer_thickness
                up_slice = slice(up_start, up_end)

                down_start = (
                    n_axis_extended - self.m_spatial_order - layer_thickness - layer_offset - 1
                )
                down_end = n_axis_extended - self.m_spatial_order - layer_offset
                down_slice = slice(down_start, down_end)

                # fetch the "mid" and "down" face values and expand dims for broadcasting
                up_vals = input_array[:, up_end, :][:, None, :]
                down_vals = input_array[:, down_start, :][:, None, :]

                # top transition (use reversed transition function)
                input_array[:, up_slice, :] = up_vals - transition_function[::-1][
                    None,
                    :,
                    None,
                ] * (up_vals - value_target)

                # bottom transition (forward transition function)
                input_array[:, down_slice, :] = down_vals - transition_function[None, :, None] * (
                    down_vals - value_target
                )
            else:
                up_start = self.m_spatial_order + layer_offset - 1
                up_end = self.m_spatial_order + layer_offset + layer_thickness
                up_slice = slice(up_start, up_end)

                down_start = (
                    n_axis_extended - self.m_spatial_order - layer_thickness - layer_offset - 1
                )
                down_end = n_axis_extended - self.m_spatial_order - layer_offset
                down_slice = slice(down_start, down_end)

                # fetch the "mid" and "down" face values and expand dims for broadcasting
                up_vals = input_array[:, up_end][:, None]
                down_vals = input_array[:, down_start][:, None]

                # top transition (use reversed transition function)
                input_array[:, up_slice] = up_vals - transition_function[::-1][None, :] * (
                    up_vals - value_target
                )

                # bottom transition (forward transition function)
                input_array[:, down_slice] = down_vals - transition_function[None, :] * (
                    down_vals - value_target
                )
        elif axis == 2:
            input_array[:, :, : self.m_spatial_order + layer_offset + layer_thickness] = (
                value_target
            )
            input_array[
                :,
                :,
                n_axis_extended - self.m_spatial_order - layer_thickness - layer_offset :,
            ] = value_target
            if is_3d:
                up_start = self.m_spatial_order + layer_offset - 1
                up_end = self.m_spatial_order + layer_offset + layer_thickness
                up_slice = slice(up_start, up_end)

                down_start = (
                    n_axis_extended - self.m_spatial_order - layer_thickness - layer_offset - 1
                )
                down_end = n_axis_extended - self.m_spatial_order - layer_offset
                down_slice = slice(down_start, down_end)

                # fetch the “mid” and “down” face values and expand dims for broadcasting
                up_vals = input_array[:, :, up_end][..., None]
                down_vals = input_array[:, :, down_start][..., None]

                # top transition (use reversed transition function)
                input_array[:, :, up_slice] = up_vals - transition_function[::-1][
                    None,
                    None,
                    :,
                ] * (up_vals - value_target)

                # bottom transition (forward transition function)
                input_array[:, :, down_slice] = down_vals - transition_function[None, None, :] * (
                    down_vals - value_target
                )
            else:
                error_msg = (
                    "axis=2 is not supported for 2D cases. Please set is_3d=True to use axis=2."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        else:
            error_msg = f"Invalid axis value. Expected 0, 1, but got {axis}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        return input_array

    @staticmethod
    def _calc_time_constants(
        dx: NDArray[np.float64],
        kappa: NDArray[np.float64],
        alpha: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return dx / kappa + alpha

    def _sort_relaxation_param_dict(
        self,
        relaxation_param_dict: dict[str, NDArray[np.float64]],
        relaxation_param_updates: dict[str, NDArray[np.float64]],
        n_relaxation_mechanisms: int,
    ) -> dict:
        kappa_x1 = relaxation_param_updates["kappa_x1"]
        kappa_x2 = relaxation_param_updates["kappa_x2"]

        d_x1 = []
        alpha_x1 = []
        d_x2 = []
        alpha_x2 = []
        time_const_x1 = []
        time_const_x2 = []
        for nu in range(1, n_relaxation_mechanisms + 1):
            d_x1_nu = relaxation_param_updates[f"d_x1_nu{nu}"]
            alpha_x1_nu = relaxation_param_updates[f"alpha_x1_nu{nu}"]
            d_x2_nu = relaxation_param_updates[f"d_x2_nu{nu}"]
            alpha_x2_nu = relaxation_param_updates[f"alpha_x2_nu{nu}"]

            d_x1.append(d_x1_nu)
            alpha_x1.append(alpha_x1_nu)
            d_x2.append(d_x2_nu)
            alpha_x2.append(alpha_x2_nu)

            time_const_x1_nu = self._calc_time_constants(
                dx=d_x1_nu,
                kappa=kappa_x1,
                alpha=alpha_x1_nu,
            )
            time_const_x2_nu = self._calc_time_constants(
                dx=d_x2_nu,
                kappa=kappa_x2,
                alpha=alpha_x2_nu,
            )
            time_const_x1.append(time_const_x1_nu)
            time_const_x2.append(time_const_x2_nu)

        time_const_x1 = np.stack(time_const_x1, axis=-1)
        time_const_x2 = np.stack(time_const_x2, axis=-1)
        d_x1 = np.stack(d_x1, axis=-1)
        alpha_x1 = np.stack(alpha_x1, axis=-1)
        d_x2 = np.stack(d_x2, axis=-1)
        alpha_x2 = np.stack(alpha_x2, axis=-1)

        # sort the nu values based on the time constants
        sorted_indices_x1 = np.argsort(time_const_x1, axis=-1)
        sorted_indices_x2 = np.argsort(time_const_x2, axis=-1)
        relaxation_param_dict["kappa_x1"] = np.atleast_2d(kappa_x1)
        relaxation_param_dict["kappa_x2"] = np.atleast_2d(kappa_x2)

        for nu in range(1, n_relaxation_mechanisms + 1):
            relaxation_param_dict[f"d_x1_nu{nu}"] = np.atleast_2d(
                np.take_along_axis(
                    d_x1,
                    np.expand_dims(sorted_indices_x1[..., nu - 1], axis=-1),
                    axis=-1,
                ).squeeze(-1),
            )
            relaxation_param_dict[f"alpha_x1_nu{nu}"] = np.atleast_2d(
                np.take_along_axis(
                    alpha_x1,
                    np.expand_dims(sorted_indices_x1[..., nu - 1], axis=-1),
                    axis=-1,
                ).squeeze(-1),
            )
            relaxation_param_dict[f"d_x2_nu{nu}"] = np.atleast_2d(
                np.take_along_axis(
                    d_x2,
                    np.expand_dims(sorted_indices_x2[..., nu - 1], axis=-1),
                    axis=-1,
                ).squeeze(-1),
            )
            relaxation_param_dict[f"alpha_x2_nu{nu}"] = np.atleast_2d(
                np.take_along_axis(
                    alpha_x2,
                    np.expand_dims(sorted_indices_x2[..., nu - 1], axis=-1),
                    axis=-1,
                ).squeeze(-1),
            )
        return relaxation_param_dict

    def plot(
        self,
        export_path: Path | str | None = Path("./temp/temp.png"),
        *,
        show: bool = False,
    ) -> None:
        """Plot the medium fields using matplotlib."""
        relaxation_param_dict_keys = initialize_relaxation_param_dict().keys()

        target_map_dict: OrderedDict = OrderedDict(
            [
                ("Sound speed", self.extended_medium.sound_speed),
                ("Density", self.extended_medium.density),
                ("Beta", self.extended_medium.beta),
                ("Air map", self.extended_medium.air_map),
            ],
        )
        for key in relaxation_param_dict_keys:
            target_map_dict[key] = self.extended_medium.relaxation_param_dict[key]

        target_map_dict.update(
            [
                ("PML mask x", self.pml_mask_x),
                ("PML mask y", self.pml_mask_y),
                ("Source mask", self.extended_source.mask),
                ("Sensor mask", self.extended_sensor.mask),
            ],
        )

        num_plots = len(target_map_dict)
        # calculate subplot shape to make a square
        n_rows = int(np.sqrt(num_plots))
        n_cols = int(np.ceil(num_plots / n_rows))
        # adjust the fig size
        fig_size = (n_cols * 5, n_rows * 5)

        plt.close("all")
        _, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)

        for ax, (title, map_data) in zip(
            axes.flatten(),
            target_map_dict.items(),
            strict=False,
        ):
            plot_utils.plot_array_on_ax(
                ax,
                map_data,
                title=title,
                xlim=(-5, self.extended_grid.ny + 5),
                ylim=(-5, self.extended_grid.nx + 5),
                reverse_y_axis=True,
            )
        plt.tight_layout()

        if export_path is not None:
            plt.savefig(export_path, dpi=300)
        if show:
            plt.show()
        plt.close("all")


@dataclass
class PMLBuilderExponentialAttenuation(PMLBuilder):
    """A class to set up PML for exponential attenuation media."""

    def __init__(
        self,
        grid: fullwave.Grid,
        medium: fullwave.Medium,
        source: fullwave.Source,
        sensor: fullwave.Sensor,
        *,
        m_spatial_order: int = 8,
        n_pml_layer: int = 40,
        # n_transition_layer: int = 40,
        # pml_alpha_target: float = 1.1,
        # pml_alpha_power_target: float = 1.6,
        # pml_strength_factor: float = 2.0,
        # use_2_relax_mechanisms: bool = False,
    ) -> None:
        """Initialize the PMLSetup with the given medium, source, sensor, and PML parameters.

        Parameters
        ----------
        grid: fullwave.Grid
            The grid configuration.
        medium : fullwave.Medium)
            The medium relaxation maps.
        source : fullwave.Source
            The source configuration.
        sensor : fullwave.Sensor
            The sensor configuration.
        m_spatial_order : int, optional
            fullwave simulation's spatial order (default is 8).
            It depends on the fullwave simulation binary version.
            Fullwave simulation has 2M th order spatial accuracy and fourth order accuracy in time.
            see Pinton, G. (2021) http://arxiv.org/abs/2106.11476 for more detail.
        n_pml_layer : int, optional
            PML layer thickness (default is 40).
        n_transition_layer : int, optional
            Number of transition layers (default is 40).
        pml_alpha_target : float, optional
            Target alpha value for PML (default is 0.5).
            This value is used to calculate the transition layer values.
        pml_alpha_power_target : float, optional
            Target alpha power value for PML (default is 1.0).
            This value is used to calculate the transition layer values.
        pml_strength_factor : float, optional
            Strength factor for PML (default is 2.0).
            This value is used to calculate the PML target values.
        use_2_relax_mechanisms : bool, optional
            If True, use 2 relaxation mechanisms for PML for stability (default is False).
            if True, pml_alpha_target, pml_alpha_power_target, and pml_strength_factor are ignored.

        """
        check_functions.check_instance(
            grid,
            fullwave.Grid,
        )
        check_functions.check_instance(
            medium,
            fullwave.Medium,
        )
        check_functions.check_instance(
            source,
            fullwave.Source,
        )
        check_functions.check_instance(
            sensor,
            fullwave.Sensor,
        )

        self.grid_org = grid
        self.medium_org = medium
        self.source_org = source
        self.sensor_org = sensor
        self.is_3d = grid.is_3d

        self.m_spatial_order = m_spatial_order
        self.n_pml_layer = n_pml_layer
        # self.n_transition_layer = n_transition_layer
        # self.pml_alpha_target = pml_alpha_target
        # self.pml_alpha_power_target = pml_alpha_power_target
        # self.pml_strength_factor = pml_strength_factor
        # self.use_2_relax_mechanisms = use_2_relax_mechanisms

        domain_size: tuple[float, ...]
        if self.is_3d:
            domain_size = (
                (self.medium_org.sound_speed.shape[0] + 2 * self.num_boundary_points)
                * self.grid_org.dx,
                (self.medium_org.sound_speed.shape[1] + 2 * self.num_boundary_points)
                * self.grid_org.dy,
                (self.medium_org.sound_speed.shape[2] + 2 * self.num_boundary_points)
                * self.grid_org.dz,
            )
        else:
            domain_size = (
                (self.medium_org.sound_speed.shape[0] + 2 * self.num_boundary_points)
                * self.grid_org.dx,
                (self.medium_org.sound_speed.shape[1] + 2 * self.num_boundary_points)
                * self.grid_org.dy,
            )
        self.extended_grid = fullwave.Grid(
            domain_size=domain_size,
            f0=self.grid_org.f0,
            duration=self.grid_org.duration,
            c0=self.grid_org.c0,
            ppw=self.grid_org.ppw,
            cfl=self.grid_org.cfl,
        )

        self.extended_medium = fullwave.Medium(
            grid=self.extended_grid,
            sound_speed=self._extend_map_for_pml(self.medium_org.sound_speed),
            density=self._extend_map_for_pml(self.medium_org.density),
            beta=self._extend_map_for_pml(self.medium_org.beta),
            alpha_coeff=self._extend_map_for_pml(self.medium_org.alpha_coeff),
            alpha_power=self._extend_map_for_pml(self.medium_org.alpha_power),
            air_map=self._extend_map_for_pml(self.medium_org.air_map),
            n_relaxation_mechanisms=self.medium_org.n_relaxation_mechanisms,
            path_relaxation_parameters_database=self.medium_org.path_relaxation_parameters_database,
            attenuation_builder=self.medium_org.attenuation_builder,
        )

        self.extended_source = fullwave.Source(
            p0=self.source_org.p0,
            mask=self._extend_map_for_pml(self.source_org.mask, fill_edge=False),
        )
        self.extended_sensor = fullwave.Sensor(
            mask=self._extend_map_for_pml(self.sensor_org.mask, fill_edge=False),
            sampling_modulus_time=self.sensor_org.sampling_modulus_time,
        )
        if self.is_3d:
            self.pml_mask_x, self.pml_mask_y, self.pml_mask_z = self._localize_pml_region()
        else:
            self.pml_mask_x, self.pml_mask_y = self._localize_pml_region()

        self.pml_layer_m = self.extended_grid.dx * self.n_pml_layer
        # self.transition_layer_m = self.extended_grid.dx * self.n_transition_layer

        self.n_polynomial = 2
        self.theoritical_reflection_coefficient = 10 ** (-30)

        if self.n_pml_layer == 0:
            self.n_transition_layer = 0

    @cached_property
    def num_boundary_points(self) -> int:
        """Returns the number of the boundary points.

        Number of PML layer and ghost cells.
        """
        return self.n_pml_layer + self.m_spatial_order

    def run(self, *, use_pml: bool = True) -> fullwave.MediumExponentialAttenuation:
        """Generate perfect matched layer (PML) relaxation parameters.

        It generates the relaxation parameters
        for the PML region considering the given medium and PML parameters.

        Returns
        -------
        Medium
            A Medium instance with the constructed domain properties.

        """
        if use_pml:
            extended_medium: fullwave.MediumExponentialAttenuation = (
                self.extended_medium.build_exponential()
            )
            if self.is_3d:
                return self._apply_pml_3d(
                    extended_medium=extended_medium,
                )

            return self._apply_pml_2d(
                extended_medium=extended_medium,
            )

        extended_medium: fullwave.MediumExponentialAttenuation = (
            self.extended_medium.build_exponential()
        )
        return extended_medium

    @staticmethod
    def _mask_body_2d(nx: int, ny: int, n_body: int) -> NDArray[np.float64]:
        """Create a mask for the PML region.

        Parameters
        ----------
        nx : int
            Number of grid points in the x-direction.
        ny : int
            Number of grid points in the y-direction.
        n_body : int
            Thickness of the body region (non-PML region).

        Returns
        -------
        NDArray[np.float64]
            A 3D numpy array representing the PML mask.

        """
        # Create coordinate grids (1-based indices like MATLAB)
        x = np.arange(1, nx + 1)[:, None]
        y = np.arange(1, ny + 1)[None, :]

        # Distances from each side boundary
        ri = np.where(x <= n_body, n_body - x + 1, np.where(x > nx - n_body, x - (nx - n_body), 0))
        rj = np.where(y <= n_body, n_body - y + 1, np.where(y > ny - n_body, y - (ny - n_body), 0))

        # Compute mask
        mask = np.sqrt(ri**2 + rj**2)

        # Normalize
        if mask.max() > 0:
            mask /= mask.max()

        return mask

    @staticmethod
    def _mask_body_3d(nx: int, ny: int, nz: int, n_body: int) -> NDArray[np.float64]:
        """Create a mask for the PML region.

        Parameters
        ----------
        nx : int
            Number of grid points in the x-direction.
        ny : int
            Number of grid points in the y-direction.
        nz : int
            Number of grid points in the z-direction.
        n_body : int
            Thickness of the body region (non-PML region).

        Returns
        -------
        NDArray[np.float64]
            A 3D numpy array representing the PML mask.

        """
        # Create coordinate grids (1-based indices like MATLAB)
        x = np.arange(1, nx + 1)[:, None, None]
        y = np.arange(1, ny + 1)[None, :, None]
        z = np.arange(1, nz + 1)[None, None, :]

        # Distances from each side boundary
        ri = np.where(x <= n_body, n_body - x + 1, np.where(x > nx - n_body, x - (nx - n_body), 0))
        rj = np.where(y <= n_body, n_body - y + 1, np.where(y > ny - n_body, y - (ny - n_body), 0))
        rk = np.where(z <= n_body, n_body - z + 1, np.where(z > nz - n_body, z - (nz - n_body), 0))

        # Compute mask
        mask = np.sqrt(ri**2 + rj**2 + rk**2)

        # Normalize
        if mask.max() > 0:
            mask /= mask.max()

        return mask

    def _apply_pml_3d(
        self,
        extended_medium: fullwave.MediumExponentialAttenuation,
    ) -> fullwave.MediumExponentialAttenuation:
        a_mask = self._mask_body_3d(
            nx=extended_medium.alpha_exp.shape[0],
            ny=extended_medium.alpha_exp.shape[1],
            nz=extended_medium.alpha_exp.shape[2],
            n_body=self.num_boundary_points,
        )
        extended_medium.alpha_exp *= 1 - a_mask
        return extended_medium

    def _apply_pml_2d(
        self,
        extended_medium: fullwave.MediumExponentialAttenuation,
    ) -> fullwave.MediumExponentialAttenuation:
        a_mask = self._mask_body_2d(
            nx=extended_medium.alpha_exp.shape[0],
            ny=extended_medium.alpha_exp.shape[1],
            n_body=self.num_boundary_points,
        )
        extended_medium.alpha_exp *= 1 - a_mask
        return extended_medium
