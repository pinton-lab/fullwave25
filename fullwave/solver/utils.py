"""utils module for Fullwave solver."""

import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numpy.typing import DTypeLike, NDArray

from fullwave.solver.shipped_database import ShippedDatabase

logger = logging.getLogger("__main__." + __name__)


def load_dat_data(dat_file_path: Path, dtype: DTypeLike = np.float32) -> NDArray[np.float64]:
    """Load data from a .dat file given its file path.

    Args:
        dat_file_path (Path): Path to the .dat file.
        dtype: Data type to use when reading the file.

    Raises
    ------
        ValueError: if dat_file_path does not exist.

    Returns
    -------
        NDArray[np.float64]: Array of data read from the file.

    """
    if not dat_file_path.exists():
        error_msg = f"dat_file_path {dat_file_path} does not exist"
        logger.error(error_msg)
        raise ValueError(error_msg)

    sim_result = np.fromfile(dat_file_path, dtype=dtype)
    if np.isnan(sim_result).any():
        message = (
            "The simulation contains NaN values. Check the simulation domains or PML settings."
        )
        logger.warning(
            message,
        )
    if np.isinf(sim_result).any():
        message = (
            "The simulation contains Inf values. Check the simulation domains or PML settings."
        )
        logger.warning(message)

    return sim_result


def load_dat_and_reshape(
    dat_file_path: Path,
    n_sensors: int,
    dtype: DTypeLike = np.float32,
) -> NDArray[np.float64]:
    """Load data from a .dat file given its file path.

    Args:
        dat_file_path (Path): Path to the .dat file.
        n_sensors: Number of sensors
        dtype: Data type to use when reading the file.

    Returns
    -------
        NDArray[np.float64]: Array of data read from the file.

    """
    data = load_dat_data(dat_file_path, dtype=dtype)
    return data.reshape(-1, n_sensors).T


def _relaxation_param_keys(n_relaxation_mechanisms: int) -> list[str]:
    """Return the ordered list of relaxation parameter keys."""
    keys = ["kappa_x1", "kappa_x2"]
    for i_relax in range(n_relaxation_mechanisms):
        suffix = f"nu{i_relax + 1}"
        keys.extend(
            [
                f"d_x1_{suffix}",
                f"alpha_x1_{suffix}",
                f"d_x2_{suffix}",
                f"alpha_x2_{suffix}",
            ],
        )
    return keys


def initialize_relaxation_param_dict(
    n_relaxation_mechanisms: int = ShippedDatabase.mechanisms,
    value: NDArray[np.float64] | None = None,
) -> dict[str, NDArray[np.float64]]:
    """Initialize a dictionary with relaxation parameters.

    Returns
    -------
        dict[str, NDArray[np.float64]]: Dictionary of relaxation parameters.

    """
    logger.debug("Initializing relaxation parameter dictionary.")
    keys = _relaxation_param_keys(n_relaxation_mechanisms)

    if value is None:
        out_dict = dict.fromkeys(keys)
    else:
        # Parallelize array copies across threads. NumPy releases the GIL
        # during memory operations so this scales with memory bandwidth.
        with ThreadPoolExecutor(max_workers=len(keys)) as executor:
            copies = list(executor.map(lambda _: value.copy(), keys))
        out_dict = dict(zip(keys, copies, strict=False))

    logger.debug("Relaxation parameter dictionary initialized.")
    return out_dict


def load_data_with_time_step(
    file_path: Path,
    n_sensors: int,
    time_step: int,
    dtype: DTypeLike = np.float32,
) -> NDArray[np.float64]:
    """Load data from a file using memory mapping with a specific time step.

    This function allows loading data for a specific time step
    without loading the entire dataset into memory.

    Args:
        file_path (Path): Path to the file.
        n_sensors: Number of sensors
        dtype: Data type to use when reading the file.
        time_step: Time step index to load.

    Returns
    -------
        NDArray[np.float64]: Array of data read from the file.

    Raises
    ------
        ValueError: if file_path does not exist.

    """
    if not file_path.exists():
        error_msg = f"file_path {file_path} does not exist"
        logger.error(error_msg)
        raise ValueError(error_msg)

    sim_result_memmap = np.memmap(file_path, dtype=dtype, mode="r")
    time_step_data = sim_result_memmap[time_step * n_sensors : (time_step + 1) * n_sensors]
    return np.array(time_step_data)  # Convert memmap to ndarray


def load_data_with_sensor_index(
    file_path: Path,
    n_sensors: int,
    sensor_index: int,
    dtype: DTypeLike = np.float32,
) -> NDArray[np.float64]:
    """Load data from a file using memory mapping with sensor index.

    This function allows loading data for a specific sensor index
    without loading the entire dataset into memory.

    Args:
        file_path (Path): Path to the file.
        n_sensors: Number of sensors
        sensor_index: Sensor index to load.
        dtype: Data type to use when reading the file.

    Returns
    -------
        NDArray[np.float64]: Array of data read from the file.

    Raises
    ------
        ValueError: if file_path does not exist.

    """
    if not file_path.exists():
        error_msg = f"file_path {file_path} does not exist"
        logger.error(error_msg)
        raise ValueError(error_msg)

    sim_result_memmap = np.memmap(file_path, dtype=dtype, mode="r")
    sensor_data = sim_result_memmap[sensor_index::n_sensors]
    return np.array(sensor_data)  # Convert memmap to ndarray
