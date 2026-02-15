"""Module for generating relaxation parameters.

using a precomputed lookup table and input attenuation values.
"""

import logging
import time
from pathlib import Path

import numba as nb
import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from fullwave.solver.utils import initialize_relaxation_param_dict

logger = logging.getLogger("__main__." + __name__)


@nb.njit(parallel=True, fastmath=True)
def _searchsorted_parallel_sorted_a(
    a_sorted: NDArray[np.float64],
    v_flat: NDArray[np.float64],
    *,
    side_is_right: bool,
) -> NDArray[np.int64]:
    n = a_sorted.size
    m = v_flat.size
    out = np.empty(m, dtype=np.int64)

    if side_is_right:
        for i in nb.prange(m):
            x = v_flat[i]
            lo = 0
            hi = n
            while lo < hi:
                mid = (lo + hi) >> 1
                if a_sorted[mid] <= x:
                    lo = mid + 1
                else:
                    hi = mid
            out[i] = lo
    else:
        for i in nb.prange(m):
            x = v_flat[i]
            lo = 0
            hi = n
            while lo < hi:
                mid = (lo + hi) >> 1
                if a_sorted[mid] < x:
                    lo = mid + 1
                else:
                    hi = mid
            out[i] = lo

    return out


def searchsorted_parallel(
    a: NDArray[np.float64],
    v: NDArray[np.float64],
    *,
    side: str = "left",
    sorter: NDArray[np.int64] | None = None,
) -> NDArray[np.int64]:
    """Make np.searchsorted parallel using Numba.

    A drop-in parallel version of np.searchsorted using Numba.

    Parameters
    ----------
    a : NDArray[np.float64]
        1-D sorted array.
    v : NDArray[np.float64]
        Array of values to search.
    side : str, optional
        'left' or 'right', optional. Default is 'left'.
        If 'left', the index of the first suitable location found is given.
        If 'right', return the last such index.
    sorter : NDArray[np.int64] | None, optional
        Optional array of indices that sort 'a'.

    Returns
    -------
    NDArray[np.int64]
        Indices into 'a' such that, if the corresponding elements in 'v' were
        inserted before the indices, the order of 'a' would be preserved.

    """
    a = np.asarray(a)
    v_arr = np.asarray(v)

    # Handle sorter: NumPy defines that indices refer to sorted(a) not original a. [page:2]
    if sorter is not None:
        sorter = np.asarray(sorter)
        a_sorted = a[sorter]
    else:
        a_sorted = a

    side_is_right = side == "right"
    v_flat = v_arr.ravel()
    out_flat = _searchsorted_parallel_sorted_a(a_sorted, v_flat, side_is_right)
    out = out_flat.reshape(v_arr.shape)

    # Scalar-in -> scalar-out, like NumPy. [page:2]
    if np.isscalar(v) or v_arr.shape == ():
        return int(out.reshape(()))
    return out


@nb.njit(parallel=True, fastmath=True)
def _map_parameters_fused_kernel(
    input_flat: NDArray[np.float64],
    alpha_sorted: NDArray[np.float64],
    power_sorted: NDArray[np.float64],
    look_up_table: NDArray[np.float64],
    invalid_matrix: NDArray[np.bool_],
    output_flat: NDArray[np.float64],
    has_invalid: NDArray[np.bool_],
) -> None:
    """Fused kernel: searchsorted + clip + invalid check + LUT lookup in one pass."""
    n = input_flat.shape[0]
    n_alpha = alpha_sorted.shape[0]
    n_power = power_sorted.shape[0]
    n_params = look_up_table.shape[2]
    max_alpha_idx = n_alpha - 1
    max_power_idx = n_power - 1

    for idx in nb.prange(n):
        # Binary search for alpha (left side)
        a_val = input_flat[idx, 0]
        lo = np.int64(0)
        hi = np.int64(n_alpha)
        while lo < hi:
            mid = (lo + hi) >> 1
            if alpha_sorted[mid] < a_val:
                lo = mid + 1
            else:
                hi = mid
        ai = lo
        ai = min(ai, max_alpha_idx)

        # Binary search for power (left side)
        p_val = input_flat[idx, 1]
        lo = np.int64(0)
        hi = np.int64(n_power)
        while lo < hi:
            mid = (lo + hi) >> 1
            if power_sorted[mid] < p_val:
                lo = mid + 1
            else:
                hi = mid
        pi = lo
        pi = min(pi, max_power_idx)

        # Check invalid (race on has_invalid[0] is fine — only sets True)
        if invalid_matrix[ai, pi]:
            has_invalid[0] = True

        # Direct LUT lookup — avoids large intermediate index arrays
        for k in range(n_params):
            output_flat[idx, k] = look_up_table[ai, pi, k]


def _map_parameters_search(
    input_tensor: NDArray[np.float64],
    look_up_table: NDArray[np.float64],
    alpha_list: NDArray[np.float64],
    power_list: NDArray[np.float64],
    invalid_matrix: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """Map (nx, ny, 2) input tensor to (nx, ny, n_params) using LUT.

    Fuses searchsorted, clip, invalid check, and LUT lookup into a single
    parallel pass to avoid allocating large intermediate arrays.

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
    logger.debug("Mapping parameters using fused kernel.")
    spatial_shape = input_tensor.shape[:-1]
    n_elements = 1
    for s in spatial_shape:
        n_elements *= s
    n_params = look_up_table.shape[2]

    alpha_sorted = np.ascontiguousarray(alpha_list[0].round(10))
    power_sorted = np.ascontiguousarray(power_list[0].round(10))

    # Reshape to (N, 2) for the fused kernel
    input_flat = np.ascontiguousarray(input_tensor.reshape(n_elements, input_tensor.shape[-1]))
    output_flat = np.empty((n_elements, n_params), dtype=look_up_table.dtype)
    has_invalid = np.array([False])

    time_start = time.time()
    _map_parameters_fused_kernel(
        input_flat,
        alpha_sorted,
        power_sorted,
        look_up_table,
        invalid_matrix,
        output_flat,
        has_invalid,
    )
    time_end = time.time()
    logger.debug("Fused kernel time: %.4f seconds.", time_end - time_start)

    if has_invalid[0]:
        # Recompute indices only for the warning path (rarely taken)
        alpha_index = searchsorted_parallel(alpha_sorted, input_flat[:, 0])
        power_index = searchsorted_parallel(power_sorted, input_flat[:, 1])
        alpha_index = np.clip(alpha_index, 0, len(alpha_sorted) - 1)
        power_index = np.clip(power_index, 0, len(power_sorted) - 1)
        invalid_indices = invalid_matrix[alpha_index, power_index].reshape(spatial_shape)
        invalid_alpha_power = np.unique(
            input_tensor[..., :2][np.where(invalid_indices)],
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

    return output_flat.reshape(*spatial_shape, n_params)


def generate_relaxation_params(
    alpha_coeff: NDArray[np.float64],
    alpha_power: NDArray[np.float64],
    n_relaxation_mechanisms: int = 2,
    path_database: Path = Path(__file__).parent
    / "bins"
    / "relaxation_params_database_num_relax=2_20260113_0957.mat",
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
        / "relaxation_params_database_num_relax=2_20260113_0957.mat",
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
