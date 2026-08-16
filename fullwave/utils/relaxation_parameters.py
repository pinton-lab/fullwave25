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


def _array_module(array: NDArray) -> object:
    if isinstance(array, np.ndarray):
        return np
    import cupy as cp  # noqa: PLC0415

    return cp


def transfer_relaxation_params_to_sound_speed(
    relaxation_param_dict: dict[str, NDArray[np.float64]],
    sound_speed: NDArray[np.float64] | float,
    n_relaxation_mechanisms: int = 2,
    reference_sound_speed: float = 1540.0,
) -> dict[str, NDArray[np.float64]]:
    """Make a lookup entry deliver its requested attenuation law at another sound speed.

    The wavenumber is ``omega / c`` times a bracket built from the relaxation
    parameters alone, so an entry calibrated at ``reference_sound_speed`` and
    used at ``c`` attenuates by the wrong factor ``reference_sound_speed / c``.
    Scaling the bracket's departure from unity by ``c / reference_sound_speed``
    corrects it. The relaxation frequencies are left alone so that no mechanism
    moves through the evaluation band. An array ``sound_speed`` transfers per
    voxel. The input dictionary is not modified.
    """
    xp = _array_module(next(iter(relaxation_param_dict.values())))
    ratio = xp.asarray(sound_speed, dtype=xp.float64) / reference_sound_speed
    if ratio.ndim == 0 and float(ratio) == 1.0:
        return dict(relaxation_param_dict)

    transferred = dict(relaxation_param_dict)
    for direction in ("x1", "x2"):
        stretching = f"kappa_{direction}"
        transferred[stretching] = 1.0 + ratio * (relaxation_param_dict[stretching] - 1.0)
        for i_relax in range(n_relaxation_mechanisms):
            strength = f"d_{direction}_nu{i_relax + 1}"
            transferred[strength] = ratio * relaxation_param_dict[strength]
    return transferred


def transfer_relaxation_params_to_band(
    relaxation_param_dict: dict[str, NDArray[np.float64]],
    band_scale: float,
    n_relaxation_mechanisms: int = 2,
) -> dict[str, NDArray[np.float64]]:
    """Multiply every relaxation rate by ``band_scale``, the rate half of the band transfer.

    Scaling the rates and the evaluation frequency together leaves each
    relaxation term unchanged, so the transfer is exact. The stretching factors
    are already non-dimensional and do not move. The key half is
    ``band_scaled_alpha_coeff`` and runs before the table is read. The input
    dictionary is not modified.
    """
    if band_scale == 1.0:
        return dict(relaxation_param_dict)

    transferred = dict(relaxation_param_dict)
    for direction in ("x1", "x2"):
        for i_relax in range(n_relaxation_mechanisms):
            suffix = f"{direction}_nu{i_relax + 1}"
            transferred[f"d_{suffix}"] = band_scale * relaxation_param_dict[f"d_{suffix}"]
            transferred[f"alpha_{suffix}"] = band_scale * relaxation_param_dict[f"alpha_{suffix}"]
    return transferred


def band_scaled_alpha_coeff(
    alpha_coeff: NDArray[np.float64],
    alpha_power: NDArray[np.float64],
    band_scale: float,
) -> NDArray[np.float64]:
    """Return the lookup key for a calibration band scaled by ``band_scale``.

    The key half of the band transfer. The exponent carries over unchanged and
    only the coefficient moves, with ``alpha_power == 1`` as the fixed point.
    """
    if band_scale == 1.0:
        return alpha_coeff
    return alpha_coeff * band_scale ** (alpha_power - 1.0)


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


def _map_parameters_search_gpu(
    input_tensor: NDArray[np.float64],
    look_up_table: NDArray[np.float64],
    alpha_list: NDArray[np.float64],
    power_list: NDArray[np.float64],
    invalid_matrix: NDArray[np.bool_],
) -> NDArray[np.float64]:
    """GPU version of _map_parameters_search using CuPy searchsorted + fancy indexing.

    Parameters
    ----------
    input_tensor : cp.ndarray
        Input tensor with shape (..., 2) where last dim is (alpha, power).
    look_up_table : NDArray[np.float64]
        Precomputed parameter table shape (B1, B2, 4 * n_relaxation + 2).
    alpha_list : NDArray[np.float64]
        List of alpha values for the lookup table.
    power_list : NDArray[np.float64]
        List of power values for the lookup table.
    invalid_matrix : NDArray[np.bool_]
        Matrix indicating invalid (alpha, power) combinations.

    Returns
    -------
    cp.ndarray
        Output tensor with shape (..., 4 * n_relaxation + 2).

    """
    import cupy as cp  # noqa: PLC0415

    logger.debug("Mapping parameters using CuPy GPU kernel.")
    time_start = time.time()

    spatial_shape = input_tensor.shape[:-1]

    # Transfer small LUT arrays to GPU (these are tiny, ~KB)
    alpha_sorted = cp.asarray(alpha_list[0].round(10))
    power_sorted = cp.asarray(power_list[0].round(10))
    lut_gpu = cp.asarray(look_up_table)

    # Flatten spatial dims
    n_elements = int(cp.prod(cp.asarray(list(spatial_shape))))
    input_flat = input_tensor.reshape(n_elements, 2)

    # Searchsorted on GPU
    alpha_indices = cp.searchsorted(alpha_sorted, input_flat[:, 0], side="left")
    power_indices = cp.searchsorted(power_sorted, input_flat[:, 1], side="left")

    # Clip to valid range
    alpha_indices = cp.clip(alpha_indices, 0, len(alpha_sorted) - 1)
    power_indices = cp.clip(power_indices, 0, len(power_sorted) - 1)

    # LUT lookup via fancy indexing
    output_flat = lut_gpu[alpha_indices, power_indices, :]

    time_end = time.time()
    logger.debug("CuPy GPU kernel time: %.4f seconds.", time_end - time_start)

    # Check for invalid combinations (transfer only the boolean result)
    invalid_gpu = cp.asarray(invalid_matrix)
    has_invalid = bool(cp.any(invalid_gpu[alpha_indices, power_indices]))
    if has_invalid:
        invalid_flags = invalid_gpu[alpha_indices, power_indices].reshape(spatial_shape)
        # Transfer only the small set of invalid points for the warning message
        invalid_indices_np = cp.asnumpy(invalid_flags)
        input_np = cp.asnumpy(input_tensor)
        invalid_alpha_power = np.unique(
            input_np[..., :2][np.where(invalid_indices_np)],
            axis=0,
        )
        invalid_attenuation = ", ".join(
            [f"({a:.4f}, {p:.4f})" for a, p in invalid_alpha_power],
        )
        message = (
            "Warning: Some attenuation values correspond to invalid relaxation parameters. "
            "This is due to the limitations of the precomputed lookup table. "
            "Please change the attenuation values.\n"
            f"Number of invalid points: {int(cp.sum(invalid_flags))}.\n"
            f"Invalid attenuation values (alpha, power): {invalid_attenuation}\n"
        )
        logger.warning(message)

    return output_flat.reshape(*spatial_shape, look_up_table.shape[2])


def generate_relaxation_params(
    alpha_coeff: NDArray[np.float64],
    alpha_power: NDArray[np.float64],
    n_relaxation_mechanisms: int = 2,
    path_database: Path = Path(__file__).parent.parent
    / "solver"
    / "bins"
    / "database"
    / "relaxation_params_database_num_relax=2_20260113_0957.mat",
    *,
    band_scale: float = 1.0,
    sound_speed: NDArray[np.float64] | float | None = None,
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
    band_scale : float, optional
        Transfer the calibration to a frequency band scaled by this factor.
        1.0, the default, is the identity and reads the table as calibrated.
    sound_speed : NDArray[np.float64] | float | None, optional
        Transfer the calibration to this sound speed. None, the default, is no
        transfer and reads the table as calibrated.

    Returns
    -------
    dict[str, NDArray[np.float64]]
        A dictionary containing the computed relaxation parameters.

    """
    relaxation_parameters_generator = RelaxationParametersGenerator(
        n_relaxation_mechanisms=n_relaxation_mechanisms,
        path_database=path_database,
    )
    return relaxation_parameters_generator.generate(
        alpha_coeff,
        alpha_power,
        band_scale=band_scale,
        sound_speed=sound_speed,
    )


class RelaxationParametersGenerator:
    """Class for generating relaxation parameters."""

    def __init__(
        self,
        *,
        n_relaxation_mechanisms: int = 2,
        path_database: Path = Path(__file__).parent.parent
        / "solver"
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
        *,
        band_scale: float = 1.0,
        sound_speed: NDArray[np.float64] | float | None = None,
    ) -> dict[str, NDArray[np.float64]]:
        """Generate relaxation parameters based on attenuation values.

        Dispatches to CuPy GPU path when inputs are CuPy arrays,
        otherwise uses the Numba CPU path.

        The band transfer straddles the table read, since its key half chooses
        which entry is looked up and its rate half scales what comes back. The
        sound-speed transfer only changes what comes back. Both default to the
        identity.

        Parameters
        ----------
        alpha_coeff : NDArray[np.float64]
            Array of attenuation coefficients.
        alpha_power : NDArray[np.float64]
            Array of attenuation power values.
        band_scale : float, optional
            Transfer the calibration to a frequency band scaled by this factor.
        sound_speed : NDArray[np.float64] | float | None, optional
            Transfer the calibration to this sound speed.

        Returns
        -------
        dict[str, NDArray[np.float64]]
            A dictionary containing the computed relaxation parameters.

        """
        alpha_coeff = band_scaled_alpha_coeff(alpha_coeff, alpha_power, band_scale)

        use_gpu = not isinstance(alpha_coeff, np.ndarray)
        if use_gpu:
            relaxation_param_dict = self._generate_gpu(alpha_coeff, alpha_power)
        else:
            relaxation_param_dict = self._generate_cpu(alpha_coeff, alpha_power)

        relaxation_param_dict = transfer_relaxation_params_to_band(
            relaxation_param_dict,
            band_scale,
            self.n_relaxation_mechanisms,
        )
        if sound_speed is not None:
            relaxation_param_dict = transfer_relaxation_params_to_sound_speed(
                relaxation_param_dict,
                sound_speed,
                self.n_relaxation_mechanisms,
            )
        return relaxation_param_dict

    def _generate_cpu(
        self,
        alpha_coeff: NDArray[np.float64],
        alpha_power: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """CPU path using Numba fused kernel."""
        self._warn_out_of_range(alpha_coeff, alpha_power, xp=np)

        alpha_coeff = np.clip(alpha_coeff, self.alpha_min, self.alpha_max)
        alpha_power = np.clip(alpha_power, self.power_min, self.power_max)

        input_data = np.stack([alpha_coeff, alpha_power], axis=-1)
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

    def _generate_gpu(
        self,
        alpha_coeff: NDArray[np.float64],
        alpha_power: NDArray[np.float64],
    ) -> dict[str, NDArray[np.float64]]:
        """GPU path using CuPy searchsorted + fancy indexing."""
        import cupy as cp  # noqa: PLC0415

        self._warn_out_of_range(alpha_coeff, alpha_power, xp=cp)

        alpha_coeff = cp.clip(alpha_coeff, self.alpha_min, self.alpha_max)
        alpha_power = cp.clip(alpha_power, self.power_min, self.power_max)

        input_data = cp.stack([alpha_coeff, alpha_power], axis=-1)
        output = _map_parameters_search_gpu(
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

    def _warn_out_of_range(self, alpha_coeff: NDArray, alpha_power: NDArray, *, xp: object) -> None:
        """Log warnings if attenuation values are out of LUT range."""
        if xp.any(alpha_coeff < self.alpha_min) or xp.any(alpha_power < self.power_min):
            error_msg = (
                "attenuation is out of range."
                "the out-of-range values will be clipped to the min value."
                f"alpha minimum: {self.alpha_min}, "
                f"power minimum: {self.power_min}"
            )
            logger.warning(error_msg)
        if xp.any(alpha_coeff > self.alpha_max) or xp.any(alpha_power > self.power_max):
            error_msg = (
                "attenuation is out of range."
                "the out-of-range values will be clipped to the max value."
                f"alpha maximum: {self.alpha_max}, "
                f"power maximum: {self.power_max}"
            )
            logger.warning(error_msg)
