"""Module for generating relaxation parameters.

using a precomputed lookup table and input attenuation values.
"""

import json
import logging
import time
from pathlib import Path

import numba as nb
import numpy as np
from numpy.typing import NDArray
from scipy.io import loadmat

from fullwave.solver.shipped_database import ShippedDatabase
from fullwave.solver.utils import initialize_relaxation_param_dict

logger = logging.getLogger("__main__." + __name__)


def _array_module(array: NDArray) -> object:
    if isinstance(array, np.ndarray):
        return np
    import cupy as cp  # noqa: PLC0415

    return cp


def _on_the_host(array: NDArray) -> NDArray:
    """Return an array on the host, whichever device it is on."""
    if isinstance(array, np.ndarray):
        return array
    return _array_module(array).asnumpy(array)


def scale_relaxation_attenuation(
    relaxation_param_dict: dict[str, NDArray[np.float64]],
    factor: NDArray[np.float64] | float,
    n_relaxation_mechanisms: int = ShippedDatabase.mechanisms,
    *,
    exact: bool = False,
) -> dict[str, NDArray[np.float64]]:
    """Scale the attenuation coefficient of the relaxation parameters.

    The attenuation coefficient is multiplied by ``factor``, and the exponent
    does not move. Kramers-Kronig dispersion is proportional to the
    coefficient, so the dispersion scales with it.

    The wavenumber is ``omega / c`` times ``(S_x1 * S_x2) ** (-1/2)``, where each
    ``S = 1 / kappa - gamma`` is the stretched spatial operator of one direction.
    An ``S`` of one is a lossless, non-dispersive medium, and the whole
    attenuation and dispersion of the model sit in how far ``S`` is from one.
    Both rules below act on that quantity.

    The default rule moves ``kappa`` toward one and scales every strength, and
    leaves every rate. It is first order.

    The exact rule holds ``S - 1`` exactly proportional to ``factor`` at every
    frequency. It is refused where it would make a rate negative or carry a
    stretching factor through zero, since the memory variable recursion is then
    unstable. Those voxels take the default rule and a warning is logged.

    An array ``factor`` scales per voxel. The input dictionary is not modified.

    Parameters
    ----------
    relaxation_param_dict : dict[str, NDArray[np.float64]]
        Relaxation parameters, as the lookup returns them.
    factor : NDArray[np.float64] | float
        What the attenuation coefficient is multiplied by.
    n_relaxation_mechanisms : int, optional
        Number of relaxation mechanisms.
    exact : bool, optional
        Use the exact rule. False, the default, is the first-order rule.

    Returns
    -------
    dict[str, NDArray[np.float64]]
        A new dictionary. A ``factor`` of 1.0 returns the input values unchanged.

    """
    xp = _array_module(next(iter(relaxation_param_dict.values())))
    ratio = xp.asarray(factor, dtype=xp.float64)
    if ratio.ndim == 0 and float(ratio) == 1.0:
        return dict(relaxation_param_dict)

    if not exact:
        return _first_order_attenuation(
            relaxation_param_dict, ratio, n_relaxation_mechanisms, xp=xp
        )

    candidate, admissible = _exact_attenuation(
        relaxation_param_dict, ratio, n_relaxation_mechanisms, xp=xp
    )
    if bool(xp.all(admissible)):
        return candidate

    first_order = _first_order_attenuation(
        relaxation_param_dict, ratio, n_relaxation_mechanisms, xp=xp
    )
    if not bool(xp.any(admissible)):
        logger.warning(
            "the exact attenuation scaling is inadmissible everywhere, "
            "so the first-order rule is used instead.",
        )
        return first_order
    logger.warning(
        "the exact attenuation scaling is inadmissible at some voxels, "
        "which take the first-order rule instead.",
    )
    return {
        key: xp.where(admissible, candidate[key], first_order[key]) for key in relaxation_param_dict
    }


def _first_order_attenuation(
    relaxation_param_dict: dict[str, NDArray[np.float64]],
    ratio: NDArray[np.float64],
    n_relaxation_mechanisms: int,
    *,
    xp: object,
) -> dict[str, NDArray[np.float64]]:
    """Move kappa toward one and scale every strength, leaving every rate.

    A factor at or above ``1 / (1 - kappa)`` carries the stretching factor to zero
    or below. The memory variable decay ``exp(-(d / kappa + rate) * dt)`` then
    exceeds one and the recursion grows without bound, so that is refused.
    """
    scaled = dict(relaxation_param_dict)
    for direction in ("x1", "x2"):
        stretching = f"kappa_{direction}"
        moved = 1.0 + ratio * (relaxation_param_dict[stretching] - 1.0)
        if bool(xp.any(moved <= 0.0)):
            error_msg = (
                f"scaling the attenuation by this factor carries {stretching} to "
                f"{float(xp.min(moved)):.6g}, at or below zero, which makes the memory "
                "variable recursion unstable. The exact rule cannot reach zero, so a "
                "smaller factor is the only route where it has already been refused."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        scaled[stretching] = moved
        for i_relax in range(n_relaxation_mechanisms):
            strength = f"d_{direction}_nu{i_relax + 1}"
            scaled[strength] = ratio * relaxation_param_dict[strength]
    return scaled


def _exact_attenuation(
    relaxation_param_dict: dict[str, NDArray[np.float64]],
    ratio: NDArray[np.float64],
    n_relaxation_mechanisms: int,
    *,
    xp: object,
) -> tuple[dict[str, NDArray[np.float64]], NDArray[np.bool_]]:
    """Hold each stretched operator exactly proportional, and say where that is admissible.

    Each of the three lines is forced by matching one term of
    ``S = 1 / kappa - gamma`` so that ``S - 1`` carries the factor exactly.
    """
    scaled = dict(relaxation_param_dict)
    admissible = xp.ones_like(ratio + relaxation_param_dict["kappa_x1"], dtype=bool)
    for direction in ("x1", "x2"):
        stretching = f"kappa_{direction}"
        before = relaxation_param_dict[stretching]
        reciprocal = 1.0 + ratio * (1.0 / before - 1.0)
        admissible = admissible & (reciprocal > 0.0)
        after = 1.0 / xp.where(reciprocal > 0.0, reciprocal, 1.0)
        scaled[stretching] = after
        for i_relax in range(n_relaxation_mechanisms):
            strength = f"d_{direction}_nu{i_relax + 1}"
            rate = f"alpha_{direction}_nu{i_relax + 1}"
            moved = ratio * relaxation_param_dict[strength] * (after / before) ** 2
            shifted = (
                relaxation_param_dict[rate]
                + relaxation_param_dict[strength] / before
                - moved / after
            )
            admissible = admissible & (shifted >= 0.0)
            scaled[strength] = moved
            scaled[rate] = shifted
    return scaled, admissible


def transfer_relaxation_params_to_sound_speed(
    relaxation_param_dict: dict[str, NDArray[np.float64]],
    sound_speed: NDArray[np.float64] | float,
    n_relaxation_mechanisms: int = ShippedDatabase.mechanisms,
    reference_sound_speed: float = 1540.0,
) -> dict[str, NDArray[np.float64]]:
    """Hold a lookup entry's attenuation law fixed at another sound speed.

    The wavenumber is ``omega / c`` times a bracket built from the relaxation
    parameters alone, so an entry calibrated at ``reference_sound_speed`` and
    used at ``c`` attenuates by the wrong factor ``reference_sound_speed / c``.
    Scaling the attenuation by ``c / reference_sound_speed`` corrects it, which
    is ``scale_relaxation_attenuation``. The relaxation
    frequencies are left alone so that no mechanism moves through the evaluation
    band. An array ``sound_speed`` transfers per voxel. The input dictionary is
    not modified.
    """
    xp = _array_module(next(iter(relaxation_param_dict.values())))
    ratio = xp.asarray(sound_speed, dtype=xp.float64) / reference_sound_speed
    return scale_relaxation_attenuation(relaxation_param_dict, ratio, n_relaxation_mechanisms)


def transfer_relaxation_params_to_band(
    relaxation_param_dict: dict[str, NDArray[np.float64]],
    band_scale: float,
    n_relaxation_mechanisms: int = ShippedDatabase.mechanisms,
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


def band_scaled_sound_speed(
    sound_speed: NDArray[np.float64] | float,
    alpha_coeff: NDArray[np.float64] | float,
    alpha_power: NDArray[np.float64] | float,
    band_scale: float,
    reference_frequency_hz: float = 5.0e6,
) -> NDArray[np.float64] | float:
    """Return the sound speed a band-transferred entry must be built and used at.

    A transferred entry quotes its sound speed at ``band_scale`` times
    ``reference_frequency_hz`` rather than at ``reference_frequency_hz``, so its
    whole phase velocity curve sits off by one constant in slowness. Building the
    medium at the speed returned here puts the curve back.

    Pass the same value to the medium and to the ``sound_speed`` argument of
    ``generate_relaxation_params``, or the attenuation moves with the phase
    velocity. ``Medium`` does both.

    The constant belongs to the requested attenuation coefficient, so it is wrong
    by the same factor wherever the lookup quantizes or clips that request.

    Parameters
    ----------
    sound_speed : NDArray[np.float64] | float
        Sound speed the medium is to carry at ``reference_frequency_hz`` [m/s].
    alpha_coeff : NDArray[np.float64] | float
        Attenuation coefficient [dB/cm/MHz^gamma].
    alpha_power : NDArray[np.float64] | float
        Attenuation power [unitless].
    band_scale : float
        Transfer the calibration to a frequency band scaled by this factor.
    reference_frequency_hz : float, optional
        Frequency the calibration quotes its sound speed at [Hz]. The shipped
        table was fitted at 5 MHz over a 1 to 20 MHz band. The table file does
        not record it, so it is carried here.

    Returns
    -------
    NDArray[np.float64] | float
        Sound speed to build the medium with [m/s]. A ``band_scale`` of 1.0
        returns ``sound_speed`` unchanged.

    """
    if band_scale == 1.0:
        return sound_speed

    xp = np if np.isscalar(alpha_coeff) else _array_module(alpha_coeff)
    coefficient = xp.asarray(alpha_coeff, dtype=xp.float64)
    exponent = xp.asarray(alpha_power, dtype=xp.float64)

    megahertz = reference_frequency_hz / 1.0e6
    nepers_per_metre = coefficient * 100.0 * megahertz**exponent / (20.0 / np.log(10.0))
    slope = nepers_per_metre / (2.0 * np.pi * reference_frequency_hz)

    tangent_shift = xp.tan(exponent * np.pi / 2.0) * slope * (band_scale ** (exponent - 1.0) - 1.0)
    logarithmic_shift = -(2.0 / np.pi) * slope * np.log(band_scale)
    takes_the_logarithmic_branch = xp.mod(exponent, 2.0) == 1.0
    shift = xp.where(takes_the_logarithmic_branch, logarithmic_shift, tangent_shift)

    transferred = 1.0 / (1.0 / xp.asarray(sound_speed, dtype=xp.float64) + shift)
    if np.isscalar(alpha_coeff) and np.isscalar(sound_speed):
        return float(transferred)
    return transferred


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
    n_relaxation_mechanisms: int = ShippedDatabase.mechanisms,
    path_database: Path = ShippedDatabase.table,
    *,
    path_invalid_cells: Path | None = None,
    band_scale: float = 1.0,
    sound_speed: NDArray[np.float64] | float | None = None,
    scale_to_requested_alpha_coeff: bool = False,
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
    path_invalid_cells : Path, optional
        Path to the JSON record an evaluation wrote, which names every invalid
        cell of the table and why it is invalid. A request that lands on one is
        warned about, and the lookup still serves it.
    band_scale : float, optional
        Transfer the calibration to a frequency band scaled by this factor.
        1.0, the default, is the identity and reads the table as calibrated.
    sound_speed : NDArray[np.float64] | float | None, optional
        Transfer the calibration to this sound speed. None, the default, is no
        transfer and reads the table as calibrated.
    scale_to_requested_alpha_coeff : bool, optional
        Give the requested attenuation coefficient rather than the calibrated
        level the lookup serves. False, the default, reproduces every result
        produced before this existed.

    Returns
    -------
    dict[str, NDArray[np.float64]]
        A dictionary containing the computed relaxation parameters.

    """
    relaxation_parameters_generator = RelaxationParametersGenerator(
        n_relaxation_mechanisms=n_relaxation_mechanisms,
        path_database=path_database,
        path_invalid_cells=path_invalid_cells,
    )
    return relaxation_parameters_generator.generate(
        alpha_coeff,
        alpha_power,
        band_scale=band_scale,
        sound_speed=sound_speed,
        scale_to_requested_alpha_coeff=scale_to_requested_alpha_coeff,
    )


class RelaxationParametersGenerator:
    """Class for generating relaxation parameters."""

    def __init__(
        self,
        *,
        n_relaxation_mechanisms: int = ShippedDatabase.mechanisms,
        path_database: Path = ShippedDatabase.table,
        path_invalid_cells: Path | None = None,
    ) -> None:
        """Initialize the relaxation parameters generator.

        Parameters
        ----------
        n_relaxation_mechanisms : int, optional
            Number of relaxation mechanisms (default is 4).
        path_database : Path, optional
            Path to the relaxation parameters database.
        path_invalid_cells : Path, optional
            Path to the JSON record an evaluation wrote, which names every
            invalid cell of the table and why it is invalid. A request that lands
            on one is warned about. Without it nothing is marked and the
            behaviour is unchanged. It is a separate file because the table is
            pinned by hash and an evaluation must not change that hash.

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
        self.invalid_cells = self._load_invalid_cells(
            self._record_beside_the_table(path_database, path_invalid_cells)
        )
        self._invalid_mask = self._invalid_mask_of(self.invalid_cells)

        self._check_database()

    @staticmethod
    def _record_beside_the_table(
        path_database: Path, path_invalid_cells: Path | None
    ) -> Path | None:
        """Return the record to read, defaulting the shipped table to its own record.

        The record describes one grid, so it pairs with one table. A caller that
        names its own table and no record therefore gets no record, rather than
        a refusal about a grid it never asked about.

        Parameters
        ----------
        path_database : Path
            The table the caller named.
        path_invalid_cells : Path or None
            The record the caller named, or None.

        Returns
        -------
        Path or None
            The record to read, or None where there is none.

        """
        if path_invalid_cells is not None:
            return path_invalid_cells
        if Path(path_database) == ShippedDatabase.table:
            return ShippedDatabase.invalid_cells
        return None

    def _load_invalid_cells(self, path: Path | None) -> dict | None:
        """Return the record of invalid cells an evaluation wrote, checked against the table.

        The record is JSON rather than a matrix, because it carries the reason
        each cell is invalid beside the cell itself.

        Parameters
        ----------
        path : Path or None
            Where the record is, or None where the caller named none.

        Returns
        -------
        dict or None
            The record, or None where the caller named none.

        Raises
        ------
        FileNotFoundError
            The caller named a file that is not there.
        ValueError
            The record does not lie on the same grid as the table.

        """
        if path is None:
            return None
        if not Path(path).exists():
            error_msg = f"Invalid-cell record not found at {path}."
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        grid = record.get("grid") or {}
        for key, mine in (
            ("alpha_coeff", self.alpha_list),
            ("alpha_power", self.power_list),
        ):
            held = np.asarray(grid.get(key, []), dtype=np.float64).ravel()
            axis = np.asarray(mine, dtype=np.float64).ravel()
            if held.shape != axis.shape or not np.allclose(held, axis):
                error_msg = (
                    f"The invalid-cell record at {path} carries a different {key} axis from "
                    f"the table at {self.path_database}, so it describes a different grid."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
        return record

    def _invalid_mask_of(self, record: dict | None) -> NDArray[np.uint8] | None:
        """Return the record laid out over the grid, 1 where a cell is invalid.

        Parameters
        ----------
        record : dict or None
            The record, or None where the caller named none.

        Returns
        -------
        NDArray[np.uint8] or None
            The mask, or None where the caller named none.

        """
        if record is None:
            return None
        mask = np.zeros((self.alpha_list.size, self.power_list.size), dtype=np.uint8)
        for cell in record.get("invalid", []):
            mask[int(cell["row"]), int(cell["column"])] = 1
        return mask

    def invalid_reasons(self) -> dict[tuple[float, float], list[str]]:
        """Return why each invalid cell is invalid, keyed by the cell.

        Returns
        -------
        dict[tuple[float, float], list[str]]
            The attenuation coefficient and power of each invalid cell, against
            its reasons. It is empty where the caller named no record.

        """
        if self.invalid_cells is None:
            return {}
        return {
            (float(cell["alpha_coeff"]), float(cell["alpha_power"])): list(cell["reasons"])
            for cell in self.invalid_cells.get("invalid", [])
        }

    def _check_database(self) -> None:
        """Check the integrity of the lookup table.

        Raises
        ------
        ValueError: If the lookup table is not 3-dimensional.
        ValueError: If the table's column count names a different mechanism count
            from the one this run asked for.
        ValueError: If the lookup table contains NaN values.

        """
        if self.look_up_table.ndim != 3:
            error_msg = "look_up_table must have 3 dimensions."
            logger.error(error_msg)
            raise ValueError(error_msg)
        columns = self.look_up_table.shape[2]
        wanted = 4 * self.n_relaxation_mechanisms + 2
        if columns != wanted:
            held = (columns - 2) / 4
            error_msg = (
                f"The table at {self.path_database} holds {columns} columns, which is "
                f"{held:g} relaxation mechanisms, and this run asked for "
                f"{self.n_relaxation_mechanisms}, which needs {wanted}. A table serves one "
                "mechanism count. Either ask for the count this table holds, or name the "
                "table for the count you want with path_relaxation_parameters_database. "
                f"The counts this package ships are {sorted(ShippedDatabase.stems)}."
            )
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
        scale_to_requested_alpha_coeff: bool = False,
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
        scale_to_requested_alpha_coeff : bool, optional
            Give the requested attenuation coefficient rather than the
            calibrated level the lookup serves.

        Returns
        -------
        dict[str, NDArray[np.float64]]
            A dictionary containing the computed relaxation parameters.

        """
        alpha_coeff = band_scaled_alpha_coeff(alpha_coeff, alpha_power, band_scale)
        self._warn_invalid(alpha_coeff, alpha_power)

        use_gpu = not isinstance(alpha_coeff, np.ndarray)
        if use_gpu:
            relaxation_param_dict = self._generate_gpu(alpha_coeff, alpha_power)
        else:
            relaxation_param_dict = self._generate_cpu(alpha_coeff, alpha_power)

        if scale_to_requested_alpha_coeff:
            relaxation_param_dict = scale_relaxation_attenuation(
                relaxation_param_dict,
                self.alpha_coeff_shortfall(alpha_coeff),
                self.n_relaxation_mechanisms,
                exact=True,
            )

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

    def alpha_coeff_shortfall(
        self,
        alpha_coeff: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Return the requested attenuation coefficient over the one the table serves.

        The lookup takes the first calibrated level at or above the request, and
        clips a request past either end of the axis. Pass the result to
        ``scale_relaxation_attenuation`` to reach the request instead. A request
        of zero returns zero, so a lossless voxel is served a bracket of one.

        Parameters
        ----------
        alpha_coeff : NDArray[np.float64]
            Attenuation coefficients as the lookup reads them, so already
            through ``band_scaled_alpha_coeff`` where a band transfer applies.

        Returns
        -------
        NDArray[np.float64]
            The request divided by what the lookup serves, per voxel.

        """
        xp = np if isinstance(alpha_coeff, np.ndarray) else _array_module(alpha_coeff)
        axis = xp.asarray(np.asarray(self.alpha_list).ravel().round(10))
        return (
            alpha_coeff / axis[self._axis_index(alpha_coeff, axis, self.alpha_min, self.alpha_max)]
        )

    def is_calibrated(
        self,
        alpha_coeff: NDArray[np.float64],
        alpha_power: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Return whether the calibration produced a usable entry for each request.

        The optimization did not converge at every cell of the grid, and the
        table flags those rather than leaving them out. The lookup still serves a
        flagged cell's parameters, so a caller that wants to exclude them has to
        ask.

        Parameters
        ----------
        alpha_coeff : NDArray[np.float64]
            Attenuation coefficients as the lookup reads them, so already
            through ``band_scaled_alpha_coeff`` where a band transfer applies.
        alpha_power : NDArray[np.float64]
            Attenuation power values.

        Returns
        -------
        NDArray[np.bool_]
            True where the calibration produced a usable entry, per voxel.

        """
        xp = np if isinstance(alpha_coeff, np.ndarray) else _array_module(alpha_coeff)
        alpha_axis = xp.asarray(np.asarray(self.alpha_list).ravel().round(10))
        power_axis = xp.asarray(np.asarray(self.power_list).ravel().round(10))
        alpha_index = self._axis_index(alpha_coeff, alpha_axis, self.alpha_min, self.alpha_max)
        power_index = self._axis_index(alpha_power, power_axis, self.power_min, self.power_max)
        return ~xp.asarray(self.invalid_matrix).astype(bool)[alpha_index, power_index]

    def is_usable(
        self,
        alpha_coeff: NDArray[np.float64],
        alpha_power: NDArray[np.float64],
    ) -> NDArray[np.bool_]:
        """Return whether an evaluation found each request usable.

        This reads the record an evaluation wrote, which carries every reason a
        cell is invalid. The reasons are the optimization that did not converge,
        the absorbing layer reflection, the delivered attenuation, the delivered
        phase velocity and the stability of the solver. ``is_calibrated`` answers
        the first of those from the table alone, without a record.

        Every request is usable where the caller named no record.

        Parameters
        ----------
        alpha_coeff : NDArray[np.float64]
            Attenuation coefficients as the lookup reads them, so already
            through ``band_scaled_alpha_coeff`` where a band transfer applies.
        alpha_power : NDArray[np.float64]
            Attenuation power values.

        Returns
        -------
        NDArray[np.bool_]
            True where the evaluation found the cell usable, per voxel.

        """
        xp = np if isinstance(alpha_coeff, np.ndarray) else _array_module(alpha_coeff)
        if self._invalid_mask is None:
            return xp.ones_like(xp.asarray(alpha_coeff), dtype=bool)
        alpha_axis = xp.asarray(np.asarray(self.alpha_list).ravel().round(10))
        power_axis = xp.asarray(np.asarray(self.power_list).ravel().round(10))
        alpha_index = self._axis_index(alpha_coeff, alpha_axis, self.alpha_min, self.alpha_max)
        power_index = self._axis_index(alpha_power, power_axis, self.power_min, self.power_max)
        return ~xp.asarray(self._invalid_mask).astype(bool)[alpha_index, power_index]

    def _warn_invalid(
        self,
        alpha_coeff: NDArray[np.float64],
        alpha_power: NDArray[np.float64],
    ) -> None:
        """Warn where a request lands on a cell an evaluation marked invalid.

        The warning names each cell with the reasons the record gives, so the
        caller sees which gate the cell failed.

        Parameters
        ----------
        alpha_coeff : NDArray[np.float64]
            Attenuation coefficients as the lookup reads them.
        alpha_power : NDArray[np.float64]
            Attenuation power values.

        """
        if self._invalid_mask is None:
            return
        usable = self.is_usable(alpha_coeff, alpha_power)
        marked = ~np.asarray(_on_the_host(usable))
        if not marked.any():
            return
        pairs = np.unique(
            np.stack(
                [
                    np.asarray(_on_the_host(alpha_coeff))[marked].ravel(),
                    np.asarray(_on_the_host(alpha_power))[marked].ravel(),
                ],
                axis=-1,
            ),
            axis=0,
        )
        reasons = self.invalid_reasons()
        logger.warning(
            "an evaluation marked %d voxels invalid, over %d cells of the lookup table. "
            "The lookup still serves them. The cells and their reasons are %s",
            int(marked.sum()),
            len(pairs),
            [
                (float(one), float(other), reasons.get((float(one), float(other)), []))
                for one, other in pairs
            ],
        )

    @staticmethod
    def _axis_index(
        value: NDArray[np.float64],
        axis: NDArray[np.float64],
        lowest: float,
        highest: float,
    ) -> NDArray[np.int64]:
        """Return the axis index the lookup kernel selects, which clips at both ends."""
        xp = np if isinstance(value, np.ndarray) else _array_module(value)
        clipped = xp.clip(value, lowest, highest)
        return xp.minimum(xp.searchsorted(axis, clipped, side="left"), axis.size - 1)

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
