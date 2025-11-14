import logging
import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from fullwave.utils.relaxation_parameters import _map_parameters_search, generate_relaxation_params


# Dummy replacement for initialize_relaxation_param_dict to have a predictable dict.
def dummy_initialize_relaxation_param_dict(n_relaxation_mechanisms: int) -> dict:
    n_params = 4 * n_relaxation_mechanisms + 2
    # Use predictable ordered keys (as Python 3.7+ dict preserves insertion order)
    return {str(i): None for i in range(n_params)}


# Helper function to create a temporary .mat database file.
def create_mat_database(lookup_table, alpha_list, power_list, invalid_matrix):
    db = {
        "database": lookup_table,
        "alpha_0_list": alpha_list,
        "power_list": power_list,
        "invalid_matrix": invalid_matrix,
    }
    with tempfile.NamedTemporaryFile(suffix=".mat", delete=False) as tmp_file:
        savemat(tmp_file.name, db)
    return Path(tmp_file.name)


def test_file_not_found(tmp_path):
    # Provide a path that does not exist.
    fake_path = tmp_path / "nonexistent.mat"
    alpha_coeff = np.array([[0.5]], dtype=np.float64)
    alpha_power = np.array([[0.5]], dtype=np.float64)
    with pytest.raises(FileNotFoundError):
        generate_relaxation_params(
            alpha_coeff,
            alpha_power,
            n_relaxation_mechanisms=4,
            path_database=fake_path,
        )


def test_lookup_table_wrong_ndim(monkeypatch):
    # Create a lookup table with wrong ndim (2D instead of 3D).
    table = np.arange(25).reshape(5, 5).astype(np.float64)
    alpha_list = np.array([[0.0, 1.0]], dtype=np.float64)
    power_list = np.array([[0.0, 1.0]], dtype=np.float64)
    invalid_matrix = np.zeros((len(alpha_list), len(power_list)), dtype=bool)
    db_path = create_mat_database(table, alpha_list, power_list, invalid_matrix)
    monkeypatch.setattr(
        "fullwave.utils.relaxation_parameters.initialize_relaxation_param_dict",
        dummy_initialize_relaxation_param_dict,
    )

    alpha_coeff = np.array([[0.5]], dtype=np.float64)
    alpha_power = np.array([[0.5]], dtype=np.float64)
    with pytest.raises(ValueError, match="3 dimensions"):
        generate_relaxation_params(
            alpha_coeff,
            alpha_power,
            n_relaxation_mechanisms=4,
            path_database=db_path,
        )
    db_path.unlink()


def test_lookup_table_wrong_shape(monkeypatch):
    # Create a lookup table with wrong number of columns in the 3rd dimension.
    # For n_relaxation_mechanisms=4, expected shape[2] is 18; here we use 17.
    table = np.arange(5 * 5 * 17).reshape(5, 5, 17).astype(np.float64)
    alpha_list = np.array([[0.0, 1.0]], dtype=np.float64)
    power_list = np.array([[0.0, 1.0]], dtype=np.float64)
    invalid_matrix = np.zeros((len(alpha_list), len(power_list)), dtype=bool)
    db_path = create_mat_database(table, alpha_list, power_list, invalid_matrix)
    monkeypatch.setattr(
        "fullwave.utils.relaxation_parameters.initialize_relaxation_param_dict",
        dummy_initialize_relaxation_param_dict,
    )

    alpha_coeff = np.array([[0.5]], dtype=np.float64)
    alpha_power = np.array([[0.5]], dtype=np.float64)
    with pytest.raises(ValueError, match="4 \\* n_relaxation_mechanisms \\+ 2 columns"):
        generate_relaxation_params(
            alpha_coeff,
            alpha_power,
            n_relaxation_mechanisms=4,
            path_database=db_path,
        )
    db_path.unlink()


def test_lookup_table_contains_nan(monkeypatch):
    # Create a lookup table that contains a NaN value.
    table = np.arange(5 * 5 * 18).reshape(5, 5, 18).astype(np.float64)
    table[0, 0, 0] = np.nan
    alpha_list = np.array([[0.0, 1.0]], dtype=np.float64)
    power_list = np.array([[0.0, 1.0]], dtype=np.float64)
    invalid_matrix = np.zeros((len(alpha_list), len(power_list)), dtype=bool)
    db_path = create_mat_database(table, alpha_list, power_list, invalid_matrix)
    monkeypatch.setattr(
        "fullwave.utils.relaxation_parameters.initialize_relaxation_param_dict",
        dummy_initialize_relaxation_param_dict,
    )

    alpha_coeff = np.array([[0.5]], dtype=np.float64)
    alpha_power = np.array([[0.5]], dtype=np.float64)
    with pytest.raises(ValueError, match="must not contain NaN"):
        generate_relaxation_params(
            alpha_coeff,
            alpha_power,
            n_relaxation_mechanisms=4,
            path_database=db_path,
        )
    db_path.unlink()


def test_map_parameters_search_basic(monkeypatch):
    # Test basic functionality of _map_parameters_search

    # Create a simple 3x3 lookup table with 6 parameters (n_relaxation=1)
    lookup_table = np.arange(3 * 3 * 6).reshape(3, 3, 6).astype(np.float64)
    alpha_list = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    power_list = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    invalid_matrix = np.zeros((len(alpha_list[0]), len(power_list[0])), dtype=bool)

    # Input tensor with exact matches
    input_tensor = np.array([[[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]]], dtype=np.float64)

    result = _map_parameters_search(
        input_tensor,
        lookup_table,
        alpha_list,
        power_list,
        invalid_matrix,
    )

    assert result.shape == (1, 3, 6)
    # Check that we get the correct lookup values
    np.testing.assert_array_equal(result[0, 0, :], lookup_table[0, 0, :])
    np.testing.assert_array_equal(result[0, 1, :], lookup_table[1, 1, :])
    np.testing.assert_array_equal(result[0, 2, :], lookup_table[2, 2, :])


def test_map_parameters_search_interpolation(monkeypatch):
    # Test that searchsorted finds nearest neighbors

    lookup_table = np.arange(3 * 3 * 6).reshape(3, 3, 6).astype(np.float64)
    alpha_list = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    power_list = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    invalid_matrix = np.zeros((len(alpha_list[0]), len(power_list[0])), dtype=bool)

    # Input values between grid points
    input_tensor = np.array([[[0.25, 0.25], [0.75, 0.75]]], dtype=np.float64)

    result = _map_parameters_search(
        input_tensor,
        lookup_table,
        alpha_list,
        power_list,
        invalid_matrix,
    )

    assert result.shape == (1, 2, 6)


def test_map_parameters_search_clipping(monkeypatch):
    # Test that out-of-bounds values are clipped

    lookup_table = np.arange(3 * 3 * 6).reshape(3, 3, 6).astype(np.float64)
    alpha_list = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    power_list = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    invalid_matrix = np.zeros((len(alpha_list[0]), len(power_list[0])), dtype=bool)

    # Input values beyond valid range
    input_tensor = np.array([[[-0.5, -0.5], [1.5, 1.5]]], dtype=np.float64)

    result = _map_parameters_search(
        input_tensor,
        lookup_table,
        alpha_list,
        power_list,
        invalid_matrix,
    )

    assert result.shape == (1, 2, 6)
    # Should clip to first and last indices
    np.testing.assert_array_equal(result[0, 0, :], lookup_table[0, 0, :])
    np.testing.assert_array_equal(result[0, 1, :], lookup_table[2, 2, :])


def test_map_parameters_search_with_invalid_points(monkeypatch, caplog):
    # Test warning when invalid points are encountered

    lookup_table = np.arange(3 * 3 * 6).reshape(3, 3, 6).astype(np.float64)
    alpha_list = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    power_list = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    invalid_matrix = np.zeros((len(alpha_list[0]), len(power_list[0])), dtype=bool)
    invalid_matrix[1, 1] = True  # Mark center point as invalid

    input_tensor = np.array([[[0.5, 0.5]]], dtype=np.float64)

    with caplog.at_level(logging.WARNING):
        result = _map_parameters_search(
            input_tensor,
            lookup_table,
            alpha_list,
            power_list,
            invalid_matrix,
        )

    assert result.shape == (1, 1, 6)
    assert "invalid relaxation parameters" in caplog.text
    assert "Number of invalid points: 1" in caplog.text


def test_map_parameters_search_multiple_invalid_points(monkeypatch, caplog):
    # Test with multiple invalid points

    lookup_table = np.arange(4 * 4 * 6).reshape(4, 4, 6).astype(np.float64)
    alpha_list = np.array([[0.0, 0.3, 0.6, 1.0]], dtype=np.float64)
    power_list = np.array([[0.0, 0.3, 0.6, 1.0]], dtype=np.float64)
    invalid_matrix = np.zeros((len(alpha_list[0]), len(power_list[0])), dtype=bool)
    invalid_matrix[1, 1] = True
    invalid_matrix[2, 2] = True

    input_tensor = np.array([[[0.3, 0.3], [0.6, 0.6], [0.0, 0.0]]], dtype=np.float64)

    with caplog.at_level(logging.WARNING):
        result = _map_parameters_search(
            input_tensor,
            lookup_table,
            alpha_list,
            power_list,
            invalid_matrix,
        )

    assert result.shape == (1, 3, 6)
    assert "Number of invalid points: 2" in caplog.text


def test_map_parameters_search_no_invalid_points(monkeypatch, caplog):
    # Test that no warning is raised when all points are valid

    lookup_table = np.arange(3 * 3 * 6).reshape(3, 3, 6).astype(np.float64)
    alpha_list = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    power_list = np.array([[0.0, 0.5, 1.0]], dtype=np.float64)
    invalid_matrix = np.zeros((len(alpha_list[0]), len(power_list[0])), dtype=bool)

    input_tensor = np.array([[[0.0, 0.0], [0.5, 0.5]]], dtype=np.float64)

    with caplog.at_level(logging.WARNING):
        result = _map_parameters_search(
            input_tensor,
            lookup_table,
            alpha_list,
            power_list,
            invalid_matrix,
        )

    assert result.shape == (1, 2, 6)
    assert "invalid relaxation parameters" not in caplog.text
