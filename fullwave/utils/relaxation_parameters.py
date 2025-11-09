"""Module for generating relaxation parameters.

using a precomputed lookup table and input attenuation values.
"""

import logging
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from fullwave.solver.utils import initialize_relaxation_param_dict

logger = logging.getLogger("__main__." + __name__)


def _map_parameters_search(
    input_tensor: NDArray[np.float64],
    look_up_table: NDArray[np.float64],
    alpha_list: NDArray[np.float64],
    power_list: NDArray[np.float64],
    invalid_matrix: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Map (nx, ny, 2) input tensor to (nx, ny, 11) using LUT.

    Parameters
    ----------
    input_tensor: NDArray[np.float64]
        Normalized input tensor [0, 1]^2
    look_up_table: NDArray[np.float64]
        Precomputed parameter table shape (B1, B2, 4 * n_relaxation + 2)
    alpha_list: NDArray[np.float64]
        List of alpha values for the lookup table.
    power_list: NDArray[np.float64]
        List of power values for the lookup table.
    invalid_matrix: NDArray[np.bool_]
        Matrix indicating invalid (alpha, power) combinations.

    Returns
    -------
    NDArray[np.float64]
    Output tensor with shape (nx, ny, 4 * n_relaxation + 2)

    """
    # search nearest in alpha_list and power_list.
    # alpha is in input_tensor[:, :, 0]
    # power is in input_tensor[:, :, 1]
    # the index corresponds to lookup table
    alpha_index = np.searchsorted(alpha_list[0].round(10), input_tensor[..., 0])
    power_index = np.searchsorted(power_list[0].round(10), input_tensor[..., 1])
    # Clip indices to valid range
    alpha_index = np.clip(alpha_index, 0, len(alpha_list[0]) - 1)
    power_index = np.clip(power_index, 0, len(power_list[0]) - 1)
    # check invalid indices
    invalid_indices = invalid_matrix[alpha_index, power_index]
    if np.any(invalid_indices):
        invalid_alpha_power = np.unique(
            input_tensor[:, :, [0, 1]][np.where(invalid_indices)],
            axis=0,
        )
        invalid_attenuation = ", ".join(
            [f"({a:.4f}, {p:.4f})" for a, p in invalid_alpha_power],
        )
        message = (
            "Warning: Some attenuation values correspond to invalid relaxation parameters. "
            "This is due to the limitations of the precomputed lookup table. "
            "Please change the attenuation values.\n"
            f"Number of invalid points: {np.sum(invalid_indices)}.\n"
            f"Invalid attenuation values (alpha, power): {invalid_attenuation}\n"
        )
        logger.warning(message)

    # Advanced indexing for 2D parameter space
    return look_up_table[alpha_index, power_index, :]


def generate_relaxation_params(
    alpha_coeff: NDArray[np.float64],
    alpha_power: NDArray[np.float64],
    n_relaxation_mechanisms: int = 2,
    path_database: Path = Path(__file__).parent
    / "bins"
    / "relaxation_params_database_num_relax=2_20251027_1437.mat",
) -> dict[str, NDArray[np.float64]]:
    """Generate relaxation parameters using a precomputed lookup table and input attenuation values.

    The binning of the attenuation value depends
    on the number of bins used to generate the lookup table.

    Parameters
    ----------
    alpha_coeff : NDArray[np.float64]
        Array of attenuation coefficients.
    alpha_power : NDArray[np.float64]
        Array of attenuation power values.
    n_relaxation_mechanisms : int, optional
        Number of relaxation mechanisms (default is 4).
    path_database : Path, optional
        Path to the relaxation parameters database.

    Returns
    -------
    dict[str, NDArray[np.float64]]
        A dictionary containing the computed relaxation parameters.

    """
    relaxation_parameters_generator = RelaxationParametersGenerator(
        n_relaxation_mechanisms=n_relaxation_mechanisms,
        path_database=path_database,
    )
    return relaxation_parameters_generator.generate(alpha_coeff, alpha_power)


class RelaxationParametersGenerator:
    """Class for generating relaxation parameters."""

    def __init__(
        self,
        *,
        n_relaxation_mechanisms: int = 2,
        path_database: Path = Path(__file__).parent
        / "bins"
        / "database"
        / "relaxation_params_database_num_relax=2_20251027_1437.mat",
    ) -> None:
        """Initialize the relaxation parameters generator.

        Parameters
        ----------
        n_relaxation_mechanisms : int, optional
            Number of relaxation mechanisms (default is 4).
        path_database : Path, optional
            Path to the relaxation parameters database.

        Raises
        ------
        FileNotFoundError
            If the relaxation parameters database is not found at the specified path.

        """
        if not path_database.exists():
            error_msg = f"Relaxation parameters database not found at {path_database}."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)

        self.n_relaxation_mechanisms = n_relaxation_mechanisms
        self.path_database = path_database

        self.database = loadmat(self.path_database)
        self.look_up_table = self.database["database"]
        self.alpha_list = self.database["alpha_0_list"]
        self.power_list = self.database["power_list"]
        self.invalid_matrix = self.database["invalid_matrix"]
        self.alpha_min = self.alpha_list.min()
        self.alpha_max = self.alpha_list.max()
        self.power_min = self.power_list.min()
        self.power_max = self.power_list.max().round(4)

        self._check_database()

    def _check_database(self) -> None:
        """Check the integrity of the lookup table.

        Raises
        ------
        ValueError: If the lookup table is not 3-dimensional.
        ValueError: If the lookup table does not have (4 * n_relaxation_mechanisms + 2) columns.
        ValueError: If the lookup table contains NaN values.

        """
        if self.look_up_table.ndim != 3:
            error_msg = "look_up_table must have 3 dimensions."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if self.look_up_table.shape[2] != 4 * self.n_relaxation_mechanisms + 2:
            error_msg = "look_up_table must have 4 * n_relaxation_mechanisms + 2 columns."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if np.isnan(self.look_up_table).any():
            error_msg = "look_up_table must not contain NaN values."
            logger.error(error_msg)
            raise ValueError(error_msg)

    def generate(
        self,
        alpha_coeff: NDArray[np.float64],
        alpha_power: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """Generate relaxation parameters based on attenuation values.

        Parameters
        ----------
        alpha_coeff : NDArray[np.float64]
            Array of attenuation coefficients.
        alpha_power : NDArray[np.float64]
            Array of attenuation power values.

        Returns
        -------
        dict[str, NDArray[np.float64]]
            A dictionary containing the computed relaxation parameters.

        """
        if np.any(alpha_coeff < self.alpha_min) or np.any(alpha_power < self.power_min):
            error_msg = (
                "attenuation is out of range."
                "the out-of-range values will be clipped to the min value."
                f"alpha minimum: {self.alpha_min}, "
                f"power minimum: {self.power_min}"
            )
            logger.warning(error_msg)
        if np.any(alpha_coeff > self.alpha_max) or np.any(alpha_power > self.power_max):
            error_msg = (
                "attenuation is out of range."
                "the out-of-range values will be clipped to the max value."
                f"alpha maximum: {self.alpha_max}, "
                f"power maximum: {self.power_max}"
            )
            logger.warning(error_msg)

        alpha_coeff = np.clip(alpha_coeff, self.alpha_min, self.alpha_max)
        alpha_power = np.clip(alpha_power, self.power_min, self.power_max)

        # # Normalize to [0, 1] for the lookup table
        # alpha_coeff = (alpha_coeff - self.alpha_min) / (self.alpha_max - self.alpha_min)
        # alpha_power = (alpha_power - self.power_min) / (self.power_max - self.power_min)

        input_data = np.stack([alpha_coeff, alpha_power], axis=-1)
        # output = _map_parameters(input_data, self.look_up_table, self.alpha_list, self.power_list)
        output = _map_parameters_search(
            input_data,
            self.look_up_table,
            self.alpha_list,
            self.power_list,
            self.invalid_matrix,
        )

        relaxation_param_dict = initialize_relaxation_param_dict(self.n_relaxation_mechanisms)
        for i, key in enumerate(relaxation_param_dict.keys()):
            relaxation_param_dict[key] = output[..., i]
        return relaxation_param_dict
