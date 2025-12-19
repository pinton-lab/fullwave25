import tempfile
from pathlib import Path

import numpy as np
import pytest

from fullwave.solver.utils import load_data_with_sensor_index, load_data_with_time_step


def test_load_data_with_time_step_success():
    """Test loading data for a specific time step."""
    n_sensors = 3
    n_time_steps = 4
    dtype = np.float32

    # Create test data: shape (n_time_steps * n_sensors,)
    test_data = np.arange(n_time_steps * n_sensors, dtype=dtype)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp_file:
        tmp_path = Path(tmp_file.name)
        test_data.tofile(tmp_file)

    try:
        # Load data for time step 2
        time_step = 2
        result = load_data_with_time_step(tmp_path, n_sensors, time_step, dtype)

        # Expected data: indices [6, 7, 8] for time_step=2, n_sensors=3
        expected = np.array([6.0, 7.0, 8.0], dtype=dtype)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == dtype
    finally:
        tmp_path.unlink()


def test_load_data_with_time_step_first():
    """Test loading data for the first time step."""
    n_sensors = 5
    n_time_steps = 3
    dtype = np.float32

    test_data = np.arange(n_time_steps * n_sensors, dtype=dtype)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp_file:
        tmp_path = Path(tmp_file.name)
        test_data.tofile(tmp_file)

    try:
        result = load_data_with_time_step(tmp_path, n_sensors, 0, dtype)
        expected = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=dtype)
        np.testing.assert_array_equal(result, expected)
    finally:
        tmp_path.unlink()


def test_load_data_with_time_step_last():
    """Test loading data for the last time step."""
    n_sensors = 4
    n_time_steps = 3
    dtype = np.float32

    test_data = np.arange(n_time_steps * n_sensors, dtype=dtype)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp_file:
        tmp_path = Path(tmp_file.name)
        test_data.tofile(tmp_file)

    try:
        result = load_data_with_time_step(tmp_path, n_sensors, 2, dtype)
        expected = np.array([8.0, 9.0, 10.0, 11.0], dtype=dtype)
        np.testing.assert_array_equal(result, expected)
    finally:
        tmp_path.unlink()


def test_load_data_with_time_step_file_not_exists():
    """Test that ValueError is raised when file does not exist."""
    non_existent_path = Path("/non/existent/path/file.dat")

    with pytest.raises(ValueError, match="file_path .* does not exist"):
        load_data_with_time_step(non_existent_path, 10, 0)


def test_load_data_with_time_step_different_dtype():
    """Test loading data with different dtype."""
    n_sensors = 2
    dtype = np.float64

    test_data = np.array([1.5, 2.5, 3.5, 4.5], dtype=dtype)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp_file:
        tmp_path = Path(tmp_file.name)
        test_data.tofile(tmp_file)

    try:
        result = load_data_with_time_step(tmp_path, n_sensors, 1, dtype)
        expected = np.array([3.5, 4.5], dtype=dtype)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == dtype
    finally:
        tmp_path.unlink()


def test_load_data_with_sensor_index_success():
    """Test loading data for a specific sensor index."""
    n_sensors = 3
    n_time_steps = 4
    dtype = np.float32

    # Create test data: shape (n_time_steps * n_sensors,)
    test_data = np.arange(n_time_steps * n_sensors, dtype=dtype)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp_file:
        tmp_path = Path(tmp_file.name)
        test_data.tofile(tmp_file)

    try:
        # Load data for sensor index 1
        sensor_index = 1
        result = load_data_with_sensor_index(tmp_path, n_sensors, sensor_index, dtype)

        # Expected data: indices [1, 4, 7, 10] for sensor_index=1, n_sensors=3
        expected = np.array([1.0, 4.0, 7.0, 10.0], dtype=dtype)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == dtype
    finally:
        tmp_path.unlink()


def test_load_data_with_sensor_index_first():
    """Test loading data for the first sensor."""
    n_sensors = 4
    n_time_steps = 3
    dtype = np.float32

    test_data = np.arange(n_time_steps * n_sensors, dtype=dtype)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp_file:
        tmp_path = Path(tmp_file.name)
        test_data.tofile(tmp_file)

    try:
        result = load_data_with_sensor_index(tmp_path, n_sensors, 0, dtype)
        expected = np.array([0.0, 4.0, 8.0], dtype=dtype)
        np.testing.assert_array_equal(result, expected)
    finally:
        tmp_path.unlink()


def test_load_data_with_sensor_index_last():
    """Test loading data for the last sensor."""
    n_sensors = 5
    n_time_steps = 3
    dtype = np.float32

    test_data = np.arange(n_time_steps * n_sensors, dtype=dtype)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp_file:
        tmp_path = Path(tmp_file.name)
        test_data.tofile(tmp_file)

    try:
        result = load_data_with_sensor_index(tmp_path, n_sensors, 4, dtype)
        expected = np.array([4.0, 9.0, 14.0], dtype=dtype)
        np.testing.assert_array_equal(result, expected)
    finally:
        tmp_path.unlink()


def test_load_data_with_sensor_index_file_not_exists():
    """Test that ValueError is raised when file does not exist."""
    non_existent_path = Path("/non/existent/path/file.dat")

    with pytest.raises(ValueError, match="file_path .* does not exist"):
        load_data_with_sensor_index(non_existent_path, 10, 0)


def test_load_data_with_sensor_index_different_dtype():
    """Test loading data with different dtype."""
    n_sensors = 2
    dtype = np.float64

    test_data = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5], dtype=dtype)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".dat") as tmp_file:
        tmp_path = Path(tmp_file.name)
        test_data.tofile(tmp_file)

    try:
        result = load_data_with_sensor_index(tmp_path, n_sensors, 1, dtype)
        expected = np.array([2.5, 4.5, 6.5], dtype=dtype)
        np.testing.assert_array_equal(result, expected)
        assert result.dtype == dtype
    finally:
        tmp_path.unlink()
