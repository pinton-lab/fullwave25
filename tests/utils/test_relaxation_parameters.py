import tempfile
from pathlib import Path

import numpy as np
import pytest
from scipy.io import savemat

from fullwave.utils.relaxation_parameters import generate_relaxation_params


# Dummy replacement for initialize_relaxation_param_dict to have a predictable dict.
def dummy_initialize_relaxation_param_dict(n_relaxation_mechanisms: int) -> dict:
    n_params = 4 * n_relaxation_mechanisms + 2
    # Use predictable ordered keys (as Python 3.7+ dict preserves insertion order)
    return {str(i): None for i in range(n_params)}


# Helper function to create a temporary .mat database file.
def create_mat_database(lookup_table, alpha_list, power_list):
    db = {
        "database": lookup_table,
        "alpha_0_list": alpha_list,
        "power_list": power_list,
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
    db_path = create_mat_database(table, alpha_list, power_list)
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
    db_path = create_mat_database(table, alpha_list, power_list)
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
    db_path = create_mat_database(table, alpha_list, power_list)
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
