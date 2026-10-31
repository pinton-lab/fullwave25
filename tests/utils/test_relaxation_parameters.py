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
    # The message must name the table, the count it holds and the count asked
    # for, because the column count alone does not say which of the two is
    # wrong.
    with pytest.raises(ValueError, match="holds 17 columns") as raised:
        generate_relaxation_params(
            alpha_coeff,
            alpha_power,
            n_relaxation_mechanisms=4,
            path_database=db_path,
        )
    said = str(raised.value)
    assert str(db_path) in said, "the message must name the table"
    assert "asked for 4" in said, "the message must name the count asked for"
    assert "needs 18" in said, "the message must name the count that table needs"
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


def _cupy_with_gpu():
    try:
        import cupy as cp

        cp.cuda.runtime.getDeviceCount()
    except Exception:
        return None
    return cp


def _same_parameters(first: dict, second: dict) -> bool:
    return all(np.array_equal(np.asarray(first[key]), np.asarray(second[key])) for key in first)


def _on_host(parameters: dict) -> dict:
    return {
        key: np.asarray(getattr(value, "get", lambda v=value: v)())
        for key, value in parameters.items()
    }


@pytest.fixture(scope="module")
def generator():
    from fullwave.utils.relaxation_parameters import RelaxationParametersGenerator

    return RelaxationParametersGenerator()


def _parameters_for(generator, alpha_coeff, alpha_power, *, dtype, xp=np):
    alpha_coeff_map = xp.full((3, 2), alpha_coeff, dtype=dtype)
    alpha_power_map = xp.full((3, 2), alpha_power, dtype=dtype)
    return _on_host(generator.generate(alpha_coeff_map, alpha_power_map))


@pytest.fixture(scope="module")
def invalid_nodes(generator):
    invalid = np.asarray(generator.invalid_matrix).astype(bool)
    alpha_nodes = np.asarray(generator.alpha_list).ravel().round(10)
    power_nodes = np.asarray(generator.power_list).ravel().round(10)
    rows, columns = np.nonzero(invalid)
    assert rows.size > 0
    return invalid, alpha_nodes, power_nodes, rows, columns


def test_float32_power_node_serves_its_own_cell(generator):
    own_cell = _parameters_for(generator, 0.5, 1.2, dtype=np.float64)
    next_cell = _parameters_for(generator, 0.5, 1.3, dtype=np.float64)
    served = _parameters_for(generator, 0.5, 1.2, dtype=np.float32)
    assert _same_parameters(served, own_cell)
    assert not _same_parameters(served, next_cell)


def test_float32_alpha_node_serves_its_own_cell(generator):
    own_cell = _parameters_for(generator, 0.1, 1.5, dtype=np.float64)
    next_cell = _parameters_for(generator, 0.11, 1.5, dtype=np.float64)
    served = _parameters_for(generator, 0.1, 1.5, dtype=np.float32)
    assert _same_parameters(served, own_cell)
    assert not _same_parameters(served, next_cell)


def test_off_node_requests_keep_the_cell_at_or_above_rule(generator):
    cell_above = _parameters_for(generator, 0.5, 1.3, dtype=np.float64)
    assert _same_parameters(_parameters_for(generator, 0.5, 1.25, dtype=np.float64), cell_above)
    assert _same_parameters(
        _parameters_for(generator, 0.5, 1.2 + 1e-4, dtype=np.float64), cell_above
    )
    assert _same_parameters(
        _parameters_for(generator, 0.5, 1.2 + 1e-4, dtype=np.float32), cell_above
    )
    assert not _same_parameters(_parameters_for(generator, 0.5, 1.2, dtype=np.float64), cell_above)


def test_float32_node_on_the_gpu_matches_the_cpu(generator):
    cp = _cupy_with_gpu()
    if cp is None:
        pytest.skip("cupy and a GPU are required")
    for alpha_coeff, alpha_power in [(0.5, 1.2), (0.1, 1.5)]:
        on_cpu = _parameters_for(generator, alpha_coeff, alpha_power, dtype=np.float64)
        on_gpu = _parameters_for(generator, alpha_coeff, alpha_power, dtype=np.float32, xp=cp)
        assert _same_parameters(on_gpu, on_cpu)
    cell_above = _parameters_for(generator, 0.5, 1.3, dtype=np.float64)
    off_node = _parameters_for(generator, 0.5, 1.2 + 1e-4, dtype=np.float32, xp=cp)
    assert _same_parameters(off_node, cell_above)


def test_is_calibrated_reports_a_float32_invalid_node_as_invalid(generator, invalid_nodes):
    _, alpha_nodes, power_nodes, rows, columns = invalid_nodes
    alpha32 = alpha_nodes[rows].astype(np.float32)
    power32 = power_nodes[columns].astype(np.float32)
    assert not generator.is_calibrated(alpha32, power32).any()
    assert not generator.is_calibrated(alpha_nodes[rows], power_nodes[columns]).any()


def test_is_calibrated_reads_the_cell_below_an_invalid_column(generator, invalid_nodes):
    invalid, alpha_nodes, power_nodes, rows, columns = invalid_nodes
    has_cell_below = columns > 0
    alpha32 = alpha_nodes[rows[has_cell_below]].astype(np.float32)
    power_below32 = power_nodes[columns[has_cell_below] - 1].astype(np.float32)
    is_valid_below = ~invalid[rows[has_cell_below], columns[has_cell_below] - 1]
    assert np.array_equal(generator.is_calibrated(alpha32, power_below32), is_valid_below)


def test_is_usable_reads_the_same_cell_for_float32_and_float64(generator, invalid_nodes):
    _, alpha_nodes, power_nodes, rows, columns = invalid_nodes
    assert np.array_equal(
        generator.is_usable(
            alpha_nodes[rows].astype(np.float32), power_nodes[columns].astype(np.float32)
        ),
        generator.is_usable(alpha_nodes[rows], power_nodes[columns]),
    )
