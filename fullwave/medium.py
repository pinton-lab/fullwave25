"""Medium class for Fullwave."""

import logging
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from fullwave import Grid
from fullwave.solver.utils import initialize_relaxation_param_dict
from fullwave.utils import check_functions, plot_utils
from fullwave.utils.relaxation_parameters import generate_relaxation_params

logger = logging.getLogger("__main__." + __name__)


@dataclass
class MediumRelaxationMaps:
    """Medium class for Fullwave."""

    grid: Grid
    sound_speed: NDArray[np.float64]
    density: NDArray[np.float64]
    beta: NDArray[np.float64]
    air_map: NDArray[np.int64]
    relaxation_param_dict: dict[str, NDArray[np.float64]]
    relaxation_param_dict_for_fw2: dict[str, NDArray[np.float64]]
    use_regression: bool = False

    def __init__(
        self,
        grid: Grid,
        sound_speed: NDArray[np.float64],
        density: NDArray[np.float64],
        beta: NDArray[np.float64],
        relaxation_param_dict: dict[str, NDArray[np.float64]],
        *,
        air_map: NDArray[np.int64] | None = None,
        n_relaxation_mechanisms: int = 2,
        use_isotropic_relaxation: bool = True,
    ) -> None:
        """Medium class for Fullwave.

        Parameters
        ----------
        grid : Grid
            Grid instance.
        sound_speed : NDArray[np.float64]
            Sound speed in the medium [m/s].
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D
        density : NDArray[np.float64]
            Density of the medium [kg/m^3].
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D
        beta : NDArray[np.float64]
            nonlinearity [unitless].
            beta = 1 + B/A / 2
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D
        relaxation_param_dict: dict[str, NDArray[np.float64]]
            relaxation parameter map dict.
            key: kappa_x1, kappa_x2, d_x1_nu{i}, alpha_x1_nu{i}, d_x2_nu{i}, alpha_x2_nu{i}
            value.shape: [nx, ny] for 2D, [nx, ny, nz] for 3D for each value
            see Pinton, G. (2021) http://arxiv.org/abs/2106.11476 for more detail.
        air_map: NDArray[np.int64], optional
            Binary matrix where the medium is air.
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D
        n_relaxation_mechanisms : int, optional
            Number of relaxation mechanisms, by default 2
        use_isotropic_relaxation : bool, optional
            Whether to use isotropic relaxation mechanisms for attenuation modeling
            to reduce memory usage while retaining accuracy.
            For 2D it will reduce the memory usage by approximately 15%.
            For 3D it will reduce the memory usage by approximately 25%.
            This option omits the anisotropic relaxation mechanisms to model the attenuation.
            We usually recommend using isotropic relaxation mechanisms
            unless the anisotropic attenuation is required for the simulation.

        """
        check_functions.check_compatible_value(
            n_relaxation_mechanisms,
            [2],
            "Only n_relaxation_mechanisms=2 are supported currently.",
        )
        self.n_relaxation_mechanisms = n_relaxation_mechanisms
        self.relaxation_param_dict = initialize_relaxation_param_dict(
            n_relaxation_mechanisms=n_relaxation_mechanisms,
            value=np.zeros_like(sound_speed),
        )
        self.grid = grid
        self.is_3d = grid.is_3d

        self.sound_speed = sound_speed
        self.density = density
        self.beta = beta

        if air_map is None:
            self.air_map = np.zeros_like(self.sound_speed, dtype=bool)
        else:
            self.air_map = air_map

        self.__post_init__()

        self._update_relaxation_param_dict(
            relaxation_param_updates=relaxation_param_dict,
        )
        self.use_isotropic_relaxation = use_isotropic_relaxation
        self.relaxation_param_dict_for_fw2 = self._calc_relaxation_param_dict_for_fw2(
            use_isotropic_relaxation=self.use_isotropic_relaxation,
        )
        self.check_fields()
        logger.debug("MediumRelaxationMaps instance created.")

    def __post_init__(self) -> None:
        """Post-initialization processing for Medium."""
        self.sound_speed = np.atleast_2d(self.sound_speed)
        self.density = np.atleast_2d(self.density)
        self.beta = np.atleast_2d(self.beta)

    def _update_relaxation_param_dict(
        self,
        relaxation_param_updates: dict[str, NDArray[np.float64]],
    ) -> None:
        self.check_relaxation_param_dict(
            relaxation_param_dict=relaxation_param_updates,
            contents_shape=self.sound_speed.shape,
            n_relaxation_mechanisms=self.n_relaxation_mechanisms,
        )

        # nu should be sorted considering the values of the time constants. (for PML implementation)
        # The sorting must be done between maps.
        # The order of the nu has no meaning because of the summation feature of Fullwave2.

        kappa_x1 = relaxation_param_updates["kappa_x1"]
        kappa_x2 = relaxation_param_updates["kappa_x2"]

        d_x1 = []
        alpha_x1 = []
        d_x2 = []
        alpha_x2 = []
        time_const_x1 = []
        time_const_x2 = []
        for nu in range(1, self.n_relaxation_mechanisms + 1):
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
        self.relaxation_param_dict["kappa_x1"] = np.atleast_2d(kappa_x1)
        self.relaxation_param_dict["kappa_x2"] = np.atleast_2d(kappa_x2)

        for nu in range(1, self.n_relaxation_mechanisms + 1):
            self.relaxation_param_dict[f"d_x1_nu{nu}"] = np.atleast_2d(
                np.take_along_axis(
                    d_x1,
                    np.expand_dims(sorted_indices_x1[..., nu - 1], axis=-1),
                    axis=-1,
                ).squeeze(-1),
            )
            self.relaxation_param_dict[f"alpha_x1_nu{nu}"] = np.atleast_2d(
                np.take_along_axis(
                    alpha_x1,
                    np.expand_dims(sorted_indices_x1[..., nu - 1], axis=-1),
                    axis=-1,
                ).squeeze(-1),
            )
            self.relaxation_param_dict[f"d_x2_nu{nu}"] = np.atleast_2d(
                np.take_along_axis(
                    d_x2,
                    np.expand_dims(sorted_indices_x2[..., nu - 1], axis=-1),
                    axis=-1,
                ).squeeze(-1),
            )
            self.relaxation_param_dict[f"alpha_x2_nu{nu}"] = np.atleast_2d(
                np.take_along_axis(
                    alpha_x2,
                    np.expand_dims(sorted_indices_x2[..., nu - 1], axis=-1),
                    axis=-1,
                ).squeeze(-1),
            )

        # check keys
        desired_dict = initialize_relaxation_param_dict(
            n_relaxation_mechanisms=self.n_relaxation_mechanisms,
        )
        key_set = set(self.relaxation_param_dict.keys())
        desired_key_set = set(desired_dict.keys())
        if key_set != desired_key_set:
            error_msg = f"Unknown relaxation parameter keys: {key_set - desired_key_set}"
            raise ValueError(error_msg)

        # for key, value in relaxation_param_updates.items():
        #     self.relaxation_param_dict[key] = np.atleast_2d(value)

    def check_relaxation_param_dict(
        self,
        relaxation_param_dict: dict[str, NDArray[np.float64]],
        contents_shape: NDArray[np.int64] | tuple[int, ...],
        n_relaxation_mechanisms: int = 2,
    ) -> None:
        """Check if the relaxation parameter updates have valid keys and matching shapes.

        Raises:
            ValueError: If the keys do not match the desired keys or
            if the shapes of the values do not match the domain shape.

        """
        desired_dict = initialize_relaxation_param_dict(
            n_relaxation_mechanisms=n_relaxation_mechanisms,
        )
        # key check
        key_set = set(relaxation_param_dict.keys())
        desired_key_set = set(desired_dict.keys())
        if key_set != desired_key_set:
            error_msg = f"Unknown relaxation parameter keys: {key_set - desired_key_set}"
            raise ValueError(error_msg)

        for value in relaxation_param_dict.values():
            if value.shape != contents_shape:
                error_msg = (
                    "Relaxation parameter map shape error: "
                    f"{value.shape} != {self.sound_speed.shape} (domain shape)"
                )
                raise ValueError(error_msg)

    @property
    def bulk_modulus(self) -> NDArray[np.float64]:
        """Return the bulk_modulus."""
        return np.multiply(self.sound_speed**2, self.density)

    @property
    def n_coords_zero(self) -> int:
        """Return the number of air coordinates.

        (alias for self.n_air)
        """
        return self.n_air

    @property
    def n_air(self) -> int:
        """Return the number of air coordinates."""
        return self.air_map.sum()

    @staticmethod
    def _calc_a_and_b(
        dx: NDArray[np.float64] | float,
        kappa_x: NDArray[np.float64] | float,
        alpha_x: NDArray[np.float64] | float,
        dt: NDArray[np.float64] | float,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        # function [a b] = ab(dx,kappax,alphax,dT)
        dx = np.array(dx)
        kappa_x = np.array(kappa_x)
        alpha_x = np.array(alpha_x)
        dt = np.array(dt)

        b = np.exp(-(dx / kappa_x + alpha_x) * dt)
        eps = 1e-10
        a = dx / (kappa_x * (dx + kappa_x * alpha_x) + eps) * (b - 1)
        return a, b

    @staticmethod
    def _calc_time_constants(
        dx: NDArray[np.float64],
        kappa: NDArray[np.float64],
        alpha: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return dx / kappa + alpha

    def _calculate_relaxation_coefficients(self) -> dict[str, NDArray[np.float64]]:
        """Calculate relaxation coefficients for all mechanisms.

        Returns
        -------
            dict[str, NDArray[np.float64]]: Dictionary with calculated coefficients.

        """
        relaxation_coefficients = {}
        relaxation_coefficients["kappa_x1"] = self.relaxation_param_dict["kappa_x1"]
        relaxation_coefficients["kappa_x2"] = self.relaxation_param_dict["kappa_x2"]

        for nu in range(1, self.n_relaxation_mechanisms + 1):
            (
                relaxation_coefficients[f"a_pml_x1_nu{nu}"],
                relaxation_coefficients[f"b_pml_x1_nu{nu}"],
            ) = self._calc_a_and_b(
                dx=self.relaxation_param_dict[f"d_x1_nu{nu}"],
                kappa_x=self.relaxation_param_dict["kappa_x1"],
                alpha_x=self.relaxation_param_dict[f"alpha_x1_nu{nu}"],
                dt=self.grid.dt,
            )
            (
                relaxation_coefficients[f"a_pml_x2_nu{nu}"],
                relaxation_coefficients[f"b_pml_x2_nu{nu}"],
            ) = self._calc_a_and_b(
                dx=self.relaxation_param_dict[f"d_x2_nu{nu}"],
                kappa_x=self.relaxation_param_dict["kappa_x2"],
                alpha_x=self.relaxation_param_dict[f"alpha_x2_nu{nu}"],
                dt=self.grid.dt,
            )
        return relaxation_coefficients

    def _calc_relaxation_param_dict_for_fw2(
        self,
        *,
        use_isotropic_relaxation: bool = True,
    ) -> dict[str, NDArray[np.float64]]:
        """Return the relaxation parameter dict for Fullwave2.

        Parameters
        ----------
        use_isotropic_relaxation : bool, optional
            Whether to use isotropic relaxation mechanisms for attenuation modeling
            to reduce memory usage while retaining accuracy.
            For 2D it will reduce the GPU memory usage by approximately 15%.
            For 3D it will reduce the GPU memory usage by approximately 30%
            and CPU memory usage by approximately 20%.
            This option omits the anisotropic relaxation mechanisms to model the attenuation.
            We usually recommend using isotropic relaxation mechanisms
            unless the anisotropic attenuation is required for the simulation.

        Returns
        -------
            dict[str, NDArray[np.float64]]: A dictionary with the calculated relaxation parameters
            formatted for Fullwave2.

        """
        if use_isotropic_relaxation:
            rename_dict = {
                "kappa_x": "kappa_x2",
                "kappa_u": "kappa_x1",
            }
            for nu in range(1, self.n_relaxation_mechanisms + 1):
                rename_dict[f"a_pml_u{nu}"] = f"a_pml_x1_nu{nu}"
                rename_dict[f"b_pml_u{nu}"] = f"b_pml_x1_nu{nu}"
                rename_dict[f"a_pml_x{nu}"] = f"a_pml_x2_nu{nu}"
                rename_dict[f"b_pml_x{nu}"] = f"b_pml_x2_nu{nu}"

            relaxation_coefficients = self._calculate_relaxation_coefficients()
            out_dict = {}
            for new_key, key in rename_dict.items():
                out_dict[new_key] = relaxation_coefficients[key].copy()
            return out_dict

        rename_dict = {
            "kappa_x": "kappa_x2",
            "kappa_y": "kappa_x2",
            "kappa_u": "kappa_x1",
            "kappa_w": "kappa_x1",
        }
        if self.is_3d:
            rename_dict["kappa_z"] = "kappa_x2"
            rename_dict["kappa_v"] = "kappa_x1"
        for nu in range(1, self.n_relaxation_mechanisms + 1):
            rename_dict[f"a_pml_u{nu}"] = f"a_pml_x1_nu{nu}"
            rename_dict[f"b_pml_u{nu}"] = f"b_pml_x1_nu{nu}"
            rename_dict[f"a_pml_w{nu}"] = f"a_pml_x1_nu{nu}"
            rename_dict[f"b_pml_w{nu}"] = f"b_pml_x1_nu{nu}"
            if self.is_3d:
                rename_dict[f"a_pml_v{nu}"] = f"a_pml_x1_nu{nu}"
                rename_dict[f"b_pml_v{nu}"] = f"b_pml_x1_nu{nu}"

            rename_dict[f"a_pml_x{nu}"] = f"a_pml_x2_nu{nu}"
            rename_dict[f"b_pml_x{nu}"] = f"b_pml_x2_nu{nu}"
            rename_dict[f"a_pml_y{nu}"] = f"a_pml_x2_nu{nu}"
            rename_dict[f"b_pml_y{nu}"] = f"b_pml_x2_nu{nu}"
            if self.is_3d:
                rename_dict[f"a_pml_z{nu}"] = f"a_pml_x2_nu{nu}"
                rename_dict[f"b_pml_z{nu}"] = f"b_pml_x2_nu{nu}"

        relaxation_coefficients = self._calculate_relaxation_coefficients()

        # extend it to x and y directions and rename the keys to Fullwave2 format
        out_dict = {}
        for new_key, key in rename_dict.items():
            out_dict[new_key] = relaxation_coefficients[key].copy()
        return out_dict

    def check_fields(self) -> None:
        """Check if the fields have the correct shape."""
        grid_shape = (
            (self.grid.nx, self.grid.ny, self.grid.nz)
            if self.is_3d
            else (self.grid.nx, self.grid.ny)
        )

        def _error_msg(
            field: NDArray[np.float64 | np.int64],
            grid_shape: NDArray[np.int64] | tuple[int, ...],
        ) -> str:
            return f"map shape error: {field.shape} != {grid_shape}"

        assert self.sound_speed.shape == grid_shape, _error_msg(self.sound_speed, grid_shape)
        assert self.density.shape == grid_shape, _error_msg(self.density, grid_shape)
        assert self.beta.shape == grid_shape, _error_msg(self.beta, grid_shape)
        for value in self.relaxation_param_dict.values():
            assert value.shape == grid_shape, _error_msg(value, grid_shape)

    def plot(
        self,
        export_path: Path | str | None = Path("./temp/temp.png"),
        *,
        show: bool = False,
        dpi: int = 300,
        plot_fw2_params: bool = False,
    ) -> None:
        """Plot the medium fields using matplotlib."""
        if self.is_3d:
            error_msg = "3D plotting is not implemented yet."
            raise NotImplementedError(error_msg)

        if plot_fw2_params:
            target_map_dict: OrderedDict = OrderedDict(
                [
                    ("Sound speed", self.sound_speed),
                    ("Density", self.density),
                    ("Beta", self.beta),
                    ("Air map", self.air_map),
                ],
            )
            for key in self.relaxation_param_dict_for_fw2:
                target_map_dict[key] = self.relaxation_param_dict_for_fw2[key]
        else:
            relaxation_param_dict_keys = initialize_relaxation_param_dict(
                n_relaxation_mechanisms=self.n_relaxation_mechanisms,
            ).keys()

            target_map_dict: OrderedDict = OrderedDict(
                [
                    ("Sound speed", self.sound_speed),
                    ("Density", self.density),
                    ("Beta", self.beta),
                    ("Air map", self.air_map),
                ],
            )
            for key in relaxation_param_dict_keys:
                target_map_dict[key] = self.relaxation_param_dict[key]

        num_plots = len(target_map_dict)
        # calculate subplot shape to make a square
        n_rows = int(np.sqrt(num_plots)) + 1
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
            plot_utils.plot_array_on_ax(ax, map_data, title=title)
        plt.tight_layout()

        if export_path is not None:
            export_path = Path(export_path)
            export_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(export_path, dpi=dpi)
        if show:
            plt.show()
        plt.close("all")

    def print_info(self) -> None:
        """Print grid information."""
        print(str(self))

    def summary(self) -> None:
        """Alias for print_info."""
        self.print_info()

    def __str__(self) -> str:
        """Return a string representation of the Medium.

        Returns
        -------
        str
            A string summarizing the Medium properties.

        """
        return (
            f"Relaxation Medium:\n"
            f"  Grid: {self.grid}\n"
            "\n"
            f"  Sound speed: min {np.min(self.sound_speed):.2f} m/s, "
            f"max {np.max(self.sound_speed):.2f} m/s\n"
            f"  Density: min {np.min(self.density):.2f} kg/m^3, "
            f"max {np.max(self.density):.2f} kg/m^3\n"
            f"  Beta: min {np.min(self.beta):.2f}, max {np.max(self.beta):.2f}\n"
            f"  Number of air coordinates: {self.n_air}\n"
            f"  Number of relaxation mechanisms: {self.n_relaxation_mechanisms}\n"
            f"  Relaxation parameters:\n"
        ) + "".join(
            [
                f"    {key}: min {np.min(value):.4e}, max {np.max(value):.4e}\n"
                for key, value in self.relaxation_param_dict.items()
            ],
        )

    def __repr__(self) -> str:
        """Return a detailed string representation of the Medium.

        Returns
        -------
        str
            A detailed string representation of the Medium instance.

        """
        return str(self)

    def build(self) -> "MediumRelaxationMaps":
        """Build the MediumRelaxationMaps instance.

        It returns self for compatibility with Solver class.

        We can pass the MediumRelaxationMaps instance
        directly to Solver instead of Medium.

        We can pass the relaxation_param_dict directly to the simulation
        bypassing the Medium.build() step.

        Returns
        -------
        MediumRelaxationMaps

        """
        return self


@dataclass
class MediumExponentialAttenuation:
    """Medium class for Fullwave with exponential attenuation."""

    grid: Grid
    sound_speed: NDArray[np.float64]
    density: NDArray[np.float64]
    alpha_exp: NDArray[np.float64]
    beta: NDArray[np.float64]
    air_map: NDArray[np.int64]

    def __init__(
        self,
        grid: Grid,
        sound_speed: NDArray[np.float64],
        density: NDArray[np.float64],
        alpha_exp: NDArray[np.float64],
        beta: NDArray[np.float64],
        *,
        air_map: NDArray[np.int64] | None = None,
    ) -> None:
        """Medium class for Fullwave.

        Parameters
        ----------
        grid: Grid
            Grid instance.
        sound_speed : NDArray[np.float64]
            Sound speed in the medium [m/s].
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D
        density : NDArray[np.float64]
            Density of the medium [kg/m^3].
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D
        alpha_exp : NDArray[np.float64]
            Exponential attenuation coefficient converted from alpha coeff.
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D
        beta : NDArray[np.float64]
            nonlinearity [unitless].
            beta = 1 + B/A / 2
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D
        air_map: NDArray[np.int64], optional
            Binary matrix where the medium is air.
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D

        """
        check_functions.check_instance(grid, Grid)
        self.grid = grid
        self.is_3d = grid.is_3d

        self.sound_speed = sound_speed
        self.density = density
        self.alpha_exp = alpha_exp
        self.beta = beta

        if air_map is None:
            self.air_map = np.zeros_like(self.sound_speed, dtype=bool)
        else:
            self.air_map = air_map

        self.__post_init__()
        self.check_fields()

    def __post_init__(self) -> None:
        """Post-initialization processing for Medium."""
        self.sound_speed = np.atleast_2d(self.sound_speed)
        self.density = np.atleast_2d(self.density)
        self.alpha_exp = np.atleast_2d(self.alpha_exp)
        self.beta = np.atleast_2d(self.beta)

    def check_fields(self) -> None:
        """Check if the fields have the correct shape."""
        grid_shape: tuple[int, ...]
        if self.is_3d:
            grid_shape = (self.grid.nx, self.grid.ny, self.grid.nz)
        else:
            grid_shape = (self.grid.nx, self.grid.ny)

        def _error_msg(
            field: NDArray[np.float64 | np.int64],
            grid_shape: NDArray[np.int64] | tuple[int, ...],
        ) -> str:
            return f"map shape error: {field.shape} != {grid_shape}"

        assert self.sound_speed.shape == grid_shape, _error_msg(self.sound_speed, grid_shape)
        assert self.density.shape == grid_shape, _error_msg(self.density, grid_shape)
        assert self.alpha_exp.shape == grid_shape, _error_msg(self.alpha_exp, grid_shape)
        assert self.beta.shape == grid_shape, _error_msg(self.beta, grid_shape)

    @property
    def bulk_modulus(self) -> NDArray[np.float64]:
        """Return the bulk_modulus."""
        return np.multiply(self.sound_speed**2, self.density)

    @property
    def n_coords_zero(self) -> int:
        """Return the number of air coordinates.

        (alias for self.n_air)
        """
        return self.n_air

    @property
    def n_air(self) -> int:
        """Return the number of air coordinates."""
        return self.air_map.sum()

    def plot(
        self,
        export_path: Path | str | None = Path("./temp/temp.png"),
        *,
        show: bool = False,
        cmap: str = "turbo",
        dpi: int = 300,
    ) -> None:
        """Plot the medium fields using matplotlib."""
        if self.is_3d:
            error_msg = "3D plotting is not implemented yet."
            raise NotImplementedError(error_msg)
        plt.close("all")
        _, axes = plt.subplots(2, 3, figsize=(15, 10))

        for ax, map_data, title in zip(
            axes.flatten(),
            [
                self.sound_speed,
                self.density,
                self.alpha_exp,
                self.beta,
                self.air_map,
            ],
            ["Sound speed", "Density", "Alpha exp", "Beta", "Air map"],
            strict=False,
        ):
            plot_utils.plot_array_on_ax(
                ax,
                map_data,
                title=title,
                xlim=(0 - 10, self.grid.ny + 10),
                ylim=(0 - 10, self.grid.nx + 10),
                reverse_y_axis=True,
                cmap=cmap,
            )
        plt.tight_layout()

        if export_path is not None:
            plt.savefig(export_path, dpi=dpi)
        if show:
            plt.show()
        plt.close("all")

    def print_info(self) -> None:
        """Print grid information."""
        print(str(self))

    def summary(self) -> None:
        """Alias for print_info."""
        self.print_info()

    def __str__(self) -> str:
        """Return a string representation of the Medium.

        Returns
        -------
        str
            A string summarizing the Medium properties.

        """
        return (
            f"Relaxation Medium:\n"
            f"  Grid: {self.grid}\n"
            "\n"
            f"  Sound speed: min {np.min(self.sound_speed):.2f} m/s, "
            f"max {np.max(self.sound_speed):.2f} m/s\n"
            f"  Density: min {np.min(self.density):.2f} kg/m^3, "
            f"max {np.max(self.density):.2f} kg/m^3\n"
            f"  Beta: min {np.min(self.beta):.2f}, max {np.max(self.beta):.2f}\n"
            f"  Number of air coordinates: {self.n_air}\n"
            f"  Exponential attenuation coefficient: min {np.min(self.alpha_exp):.2f}, "
            f"max {np.max(self.alpha_exp):.2f}\n"
        )

    def __repr__(self) -> str:
        """Return a detailed string representation of the Medium.

        Returns
        -------
        str
            A detailed string representation of the Medium instance.

        """
        return str(self)


@dataclass
class Medium:
    """Medium class for Fullwave."""

    grid: Grid
    sound_speed: NDArray[np.float64]
    density: NDArray[np.float64]
    alpha_coeff: NDArray[np.float64]
    alpha_power: NDArray[np.float64]
    beta: NDArray[np.float64]
    air_map: NDArray[np.int64]
    attenuation_builder: str = "lookup"

    def __init__(
        self,
        grid: Grid,
        sound_speed: NDArray[np.float64],
        density: NDArray[np.float64],
        alpha_coeff: NDArray[np.float64],
        alpha_power: NDArray[np.float64],
        beta: NDArray[np.float64],
        *,
        air_map: NDArray[np.int64] | None = None,
        path_relaxation_parameters_database: Path = Path(__file__).parent
        / "solver"
        / "bins"
        / "database"
        / "relaxation_params_database_num_relax=2_20251027_1437.mat",
        n_relaxation_mechanisms: int = 2,
        attenuation_builder: str = "lookup",
        use_isotropic_relaxation: bool = True,
    ) -> None:
        """Medium class for Fullwave.

        Parameters
        ----------
        grid: Grid
            Grid instance.
        sound_speed : NDArray[np.float64]
            Sound speed in the medium [m/s].
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D
        density : NDArray[np.float64]
            Density of the medium [kg/m^3].
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D
        alpha_coeff : NDArray[np.float64]
            Attenuation coefficient [dB/cm/MHz^gamma].
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D
        alpha_power : NDArray[np.float64]
            Attenuation power [unitless].
            gamma in the attenuation coefficient (power law)
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D
        beta : NDArray[np.float64]
            nonlinearity [unitless].
            beta = 1 + B/A / 2
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D
        air_map: NDArray[np.int64], optional
            Binary matrix where the medium is air.
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D
        path_relaxation_parameters_database : Path, optional
            Path to the relaxation parameters database.
        n_relaxation_mechanisms : int, optional
            Number of relaxation mechanisms, by default 4
        attenuation_builder : str, optional
            Attenuation builder method, by default "lookup".
            Options are "lookup", "interpolation", and "regression".
        use_isotropic_relaxation : bool, optional
            Whether to use isotropic relaxation mechanisms for attenuation modeling
            to reduce memory usage while retaining accuracy.
            For 2D it will reduce the memory usage by approximately 15%.
            For 3D it will reduce the memory usage by approximately 25%.
            This option omits the anisotropic relaxation mechanisms to model the attenuation.
            We usually recommend using isotropic relaxation mechanisms
            unless the anisotropic attenuation is required for the simulation.

        """
        check_functions.check_compatible_value(
            n_relaxation_mechanisms,
            [2],
            "Only n_relaxation_mechanisms=2 are supported currently.",
        )
        check_functions.check_instance(grid, Grid)
        check_functions.check_path_exists(path_relaxation_parameters_database)
        self.grid = grid
        self.is_3d = grid.is_3d

        self.sound_speed = sound_speed
        self.density = density
        self.alpha_coeff = alpha_coeff
        self.alpha_power = alpha_power
        self.beta = beta
        if air_map is None:
            self.air_map = np.zeros_like(self.sound_speed, dtype=bool)
        else:
            self.air_map = air_map
        self.path_relaxation_parameters_database = path_relaxation_parameters_database
        self.n_relaxation_mechanisms = n_relaxation_mechanisms
        self.use_isotropic_relaxation = use_isotropic_relaxation

        if self.n_relaxation_mechanisms != 2 and self.air_map.sum() > 0:
            warning_msg = (
                "Warning: Currently, only n_relaxation_mechanisms=2 supports air regions. "
                "Setting air regions to zero for other n_relaxation_mechanisms."
            )
            logger.warning(warning_msg)
            self.air_map = np.zeros_like(self.sound_speed, dtype=bool)

        self.attenuation_builder = attenuation_builder
        self.__post_init__()
        self.check_fields()
        logger.debug("Medium instance created.")

    def __post_init__(self) -> None:
        """Post-initialization processing for Medium."""
        self.sound_speed = np.atleast_2d(self.sound_speed)
        self.density = np.atleast_2d(self.density)
        self.alpha_coeff = np.atleast_2d(self.alpha_coeff)
        self.alpha_power = np.atleast_2d(self.alpha_power)
        self.beta = np.atleast_2d(self.beta)

    def check_fields(self) -> None:
        """Check if the fields have the correct shape."""
        grid_shape: tuple[int, ...]
        if self.is_3d:
            grid_shape = (self.grid.nx, self.grid.ny, self.grid.nz)
        else:
            grid_shape = (self.grid.nx, self.grid.ny)

        def _error_msg(
            field: NDArray[np.float64 | np.int64],
            grid_shape: NDArray[np.int64] | tuple[int, ...],
        ) -> str:
            return f"map shape error: {field.shape} != {grid_shape}"

        assert self.sound_speed.shape == grid_shape, _error_msg(self.sound_speed, grid_shape)
        assert self.density.shape == grid_shape, _error_msg(self.density, grid_shape)
        assert self.alpha_coeff.shape == grid_shape, _error_msg(self.alpha_coeff, grid_shape)
        assert self.alpha_power.shape == grid_shape, _error_msg(self.alpha_power, grid_shape)
        assert self.beta.shape == grid_shape, _error_msg(self.beta, grid_shape)
        logger.debug("All medium fields have correct shapes.")

    @property
    def bulk_modulus(self) -> NDArray[np.float64]:
        """Return the bulk_modulus."""
        return np.multiply(self.sound_speed**2, self.density)

    @property
    def n_coords_zero(self) -> int:
        """Return the number of air coordinates.

        (alias for self.n_air)
        """
        return self.n_air

    @property
    def n_air(self) -> int:
        """Return the number of air coordinates."""
        return self.air_map.sum()

    def plot(
        self,
        export_path: Path | str | None = Path("./temp/temp.png"),
        *,
        show: bool = False,
        cmap: str = "turbo",
        figsize: tuple = (20, 6),
        fontsize_title: int = 20,
        dpi: int = 300,
    ) -> None:
        """Plot the medium fields using matplotlib."""
        if self.is_3d:
            plt.close("all")
            _, axes = plt.subplots(2, 6, figsize=figsize)
            # plot the x-y axis and x-z axis slices
            for ax, map_data, title in zip(
                axes.flatten(),
                [
                    self.sound_speed[:, :, self.grid.nz // 2],
                    self.sound_speed[:, self.grid.ny // 2, :],
                    self.density[:, :, self.grid.nz // 2],
                    self.density[:, self.grid.ny // 2, :],
                    self.alpha_coeff[:, :, self.grid.nz // 2],
                    self.alpha_coeff[:, self.grid.ny // 2, :],
                    self.alpha_power[:, :, self.grid.nz // 2],
                    self.alpha_power[:, self.grid.ny // 2, :],
                    self.beta[:, :, self.grid.nz // 2],
                    self.beta[:, self.grid.ny // 2, :],
                    self.air_map[:, :, self.grid.nz // 2],
                    self.air_map[:, self.grid.ny // 2, :],
                ],
                [
                    "Sound speed (x-y slice)",
                    "Sound speed (x-z slice)",
                    "Density (x-y slice)",
                    "Density (x-z slice)",
                    "Alpha coeff (x-y slice)",
                    "Alpha coeff (x-z slice)",
                    "Alpha power (x-y slice)",
                    "Alpha power (x-z slice)",
                    "Beta (x-y slice)",
                    "Beta (x-z slice)",
                    "Air map (x-y slice)",
                    "Air map (x-z slice)",
                ],
                strict=False,
            ):
                plot_utils.plot_array_on_ax(
                    ax,
                    map_data,
                    title=title,
                    cmap=cmap,
                )
            plt.tight_layout()
            if export_path is not None:
                plt.savefig(export_path, dpi=dpi)
            if show:
                plt.show()
            plt.close("all")
        else:
            plt.close("all")
            _, axes = plt.subplots(2, 3, figsize=(15, 10))

            for ax, map_data, title in zip(
                axes.flatten(),
                [
                    self.sound_speed,
                    self.density,
                    self.alpha_coeff,
                    self.alpha_power,
                    self.beta,
                    self.air_map,
                ],
                [
                    (
                        "Sound speed\n"
                        r"$c$"
                    ),
                    (
                        "Density\n"
                        r"$\rho$"
                    ),
                    (
                        "Alpha coefficient\n"
                        r"$\alpha_0$"
                    ),
                    (
                        "Power law exponent\n"
                        r"$\gamma$"
                    ),
                    (
                        "Nonlinearity\n"
                        r"$\beta=1+\frac{B}{2A}$"
                    ),
                    "Air map",
                ],
                strict=False,
            ):
                plot_utils.plot_array_on_ax(
                    ax,
                    map_data,
                    title=title,
                    xlim=(0 - 10, self.grid.ny + 10),
                    ylim=(0 - 10, self.grid.nx + 10),
                    reverse_y_axis=True,
                    cmap=cmap,
                )
                ax.title.set_fontsize(fontsize_title)
            plt.tight_layout()

            if export_path is not None:
                plt.savefig(export_path, dpi=dpi)
            if show:
                plt.show()
            plt.close("all")

    # ---

    def build(self) -> MediumRelaxationMaps:
        """Retrieve the relaxation parameters from alpha and power maps.

        it uses the relaxation parameters look up table
        to generate the relaxation parameters.

        Returns:
            MediumRelaxationMaps: An instance of MediumRelaxationMaps
            built from the retrieved relaxation parameters.

        Raises:
            ValueError: If an unknown attenuation_builder is specified.

        """
        logger.debug("Building MediumRelaxationMaps from alpha and power maps.")
        if self.attenuation_builder == "lookup":
            relaxation_param_dict = generate_relaxation_params(
                n_relaxation_mechanisms=self.n_relaxation_mechanisms,
                alpha_coeff=self.alpha_coeff,
                alpha_power=self.alpha_power,
                path_database=self.path_relaxation_parameters_database,
            )
        else:
            error_msg = (
                f"Unknown attenuation_builder: {self.attenuation_builder}. "
                'Only "lookup" is supported currently.'
            )
            raise ValueError(error_msg)
        return MediumRelaxationMaps(
            grid=self.grid,
            sound_speed=self.sound_speed,
            density=self.density,
            beta=self.beta,
            relaxation_param_dict=relaxation_param_dict,
            air_map=self.air_map,
            n_relaxation_mechanisms=self.n_relaxation_mechanisms,
            use_isotropic_relaxation=self.use_isotropic_relaxation,
        )

    def _db_mhz_cm_to_a_exp(
        self,
        alpha_coeff: np.ndarray,
    ) -> np.ndarray:
        """Convert alpha in [dB/cm/MHz] to a_exp for Fullwave.

        Parameters
        ----------
        alpha_coeff : np.ndarray
            Attenuation coefficient [dB/cm/MHz].
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D

        Returns
        -------
        np.ndarray
            a_exp for Fullwave.
            shape: [nx, ny] for 2D, [nx, ny, nz] for 3D

        """
        np_factor = -10 * np.log10(np.exp(-1))  # equivalent to -db(exp(-1)) in MATLAB
        f0 = self.grid.omega / (2.0 * np.pi * 1e6)
        texp = alpha_coeff / 2.0 * f0 * self.grid.c0 / (1e-2 * np_factor)
        return np.exp(-self.grid.dt * texp)

    def build_exponential(self) -> MediumExponentialAttenuation:
        """Build MediumExponentialAttenuation from alpha and power maps.

        Returns:
            MediumExponentialAttenuation: An instance of MediumExponentialAttenuation
            built from the alpha and power maps.

        """
        logger.debug("Building MediumExponentialAttenuation from alpha and power maps.")
        alpha_exp = self._db_mhz_cm_to_a_exp(
            self.alpha_coeff,
        )
        return MediumExponentialAttenuation(
            grid=self.grid,
            sound_speed=self.sound_speed,
            density=self.density,
            alpha_exp=alpha_exp,
            beta=self.beta,
            air_map=self.air_map,
        )

    def print_info(self) -> None:
        """Print grid information."""
        print(str(self))

    def summary(self) -> None:
        """Alias for print_info."""
        self.print_info()

    def __str__(self) -> str:
        """Return a string representation of the Medium.

        Returns
        -------
        str
            A string summarizing the Medium properties.

        """
        return (
            f"Medium: \n"
            f"  Grid: {self.grid}\n"
            "\n"
            f"  Sound speed: min={np.min(self.sound_speed):.2f}, "
            f"max={np.max(self.sound_speed):.2f}\n"
            f"  Density: min={np.min(self.density):.2f}, "
            f"max={np.max(self.density):.2f}\n"
            f"  Alpha coeff: min={np.min(self.alpha_coeff):.2f}, "
            f"max={np.max(self.alpha_coeff):.2f}\n"
            f"  Alpha power: min={np.min(self.alpha_power):.2f}, "
            f"max={np.max(self.alpha_power):.2f}\n"
            f"  Beta: min={np.min(self.beta):.2f}, max={np.max(self.beta):.2f}\n"
            f"  Number of air coords: {self.n_air}\n"
            f"  Attenuation builder: {self.attenuation_builder}\n"
        )

    def __repr__(self) -> str:
        """Return a detailed string representation of the Medium.

        Returns
        -------
        str
            A detailed string representation of the Medium instance.

        """
        return str(self)
