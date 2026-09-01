import numpy as np
import pytest

from fullwave.sensor import Sensor


def test_post_init_mask_conversion():
    # Provide a 1D array and verify conversion to at least 2D and mapping of coordinates.
    mask = np.array([1, 0, 1])
    sensor = Sensor(mask)
    assert sensor.mask.ndim >= 2
    expected_outcoords = np.argwhere(sensor.mask)
    np.testing.assert_array_equal(sensor.outcoords, expected_outcoords)


def test_validate_success():
    # Create a valid 2D sensor mask with at least one true value.
    mask = np.array([[0, 1], [1, 0]])
    sensor = Sensor(mask)
    grid_shape = sensor.mask.shape
    # Should pass without raising an error.
    sensor.validate(grid_shape)


def test_validate_fail_wrong_shape():
    # The sensor mask shape doesn't match the provided grid shape.
    mask = np.array([[1, 0], [0, 1]])
    sensor = Sensor(mask)
    wrong_shape = (2, 3)  # Incorrect grid shape.
    with pytest.raises(AssertionError):
        sensor.validate(wrong_shape)


def test_validate_fail_no_true():
    # A sensor mask without any true values should raise an AssertionError upon validation.
    mask = np.array([[0, 0], [0, 0]])
    sensor = Sensor(mask)
    with pytest.raises(AssertionError):
        sensor.validate(sensor.mask.shape)


# --- Coordinate input mode tests ---


def test_coords_input_2d():
    coords = np.array([[0, 1], [1, 0]])
    grid_shape = (2, 2)
    sensor = Sensor(coords=coords, grid_shape=grid_shape)
    assert sensor.n_sensors == 2
    assert sensor.grid_shape == grid_shape
    assert not sensor.is_3d
    np.testing.assert_array_equal(sensor.outcoords, coords)


def test_coords_input_3d():
    coords = np.array([[0, 1, 2], [3, 4, 5]])
    grid_shape = (10, 10, 10)
    sensor = Sensor(coords=coords, grid_shape=grid_shape)
    assert sensor.n_sensors == 2
    assert sensor.is_3d
    np.testing.assert_array_equal(sensor.outcoords, coords)


def test_coords_input_validate():
    coords = np.array([[0, 1], [1, 0]])
    grid_shape = (2, 2)
    sensor = Sensor(coords=coords, grid_shape=grid_shape)
    sensor.validate(grid_shape)


def test_coords_input_mask_property():
    coords = np.array([[0, 1], [1, 0]])
    grid_shape = (2, 2)
    sensor = Sensor(coords=coords, grid_shape=grid_shape)
    expected_mask = np.array([[0, 1], [1, 0]])
    np.testing.assert_array_equal(sensor.mask, expected_mask)


def test_coords_equivalence_with_mask():
    mask = np.array([[0, 1, 0], [1, 0, 1]])
    sensor_from_mask = Sensor(mask)
    sensor_from_coords = Sensor(
        coords=sensor_from_mask.outcoords,
        grid_shape=sensor_from_mask.grid_shape,
    )
    assert sensor_from_mask.n_sensors == sensor_from_coords.n_sensors
    np.testing.assert_array_equal(sensor_from_mask.outcoords, sensor_from_coords.outcoords)
    np.testing.assert_array_equal(sensor_from_mask.mask, sensor_from_coords.mask)


def test_coords_without_grid_shape_raises():
    coords = np.array([[0, 1]])
    with pytest.raises(ValueError, match="grid_shape is required"):
        Sensor(coords=coords)


def test_coords_and_mask_raises():
    mask = np.array([[1, 0]])
    coords = np.array([[0, 0]])
    with pytest.raises(ValueError, match="mutually exclusive"):
        Sensor(mask=mask, coords=coords, grid_shape=(1, 2))


def test_no_input_raises():
    with pytest.raises(ValueError, match="Either mask"):
        Sensor()


# --- Sparse-grid mode tests ---


def test_sparse_grid_2d():
    sensor = Sensor(mod_x=4, mod_y=4)
    assert sensor.is_sparse_grid
    assert sensor.mod_x == 4
    assert sensor.mod_y == 4
    assert sensor.mod_z == 0
    assert not sensor.is_3d
    assert sensor.n_sensors == 0
    assert sensor.outcoords.shape == (0, 2)
    assert sensor.grid_shape is None


def test_sparse_grid_3d():
    sensor = Sensor(mod_x=4, mod_y=4, mod_z=2)
    assert sensor.is_sparse_grid
    assert sensor.mod_z == 2
    assert sensor.is_3d
    assert sensor.outcoords.shape == (0, 3)


def test_sparse_grid_validate_2d_ok():
    sensor = Sensor(mod_x=4, mod_y=4)
    sensor.validate((100, 200))  # 2D grid — should not raise


def test_sparse_grid_validate_3d_missing_mod_z_raises():
    sensor = Sensor(mod_x=4, mod_y=4)  # mod_z defaults to 0
    with pytest.raises(ValueError, match="mod_z"):
        sensor.validate((100, 200, 50))  # 3D grid


def test_sparse_grid_validate_3d_ok():
    sensor = Sensor(mod_x=4, mod_y=4, mod_z=4)
    sensor.validate((100, 200, 50))  # should not raise


def test_sparse_grid_missing_mod_y_raises():
    with pytest.raises(ValueError, match="Both mod_x and mod_y"):
        Sensor(mod_x=4)


def test_sparse_grid_missing_mod_x_raises():
    with pytest.raises(ValueError, match="Both mod_x and mod_y"):
        Sensor(mod_y=4)


def test_sparse_grid_with_mask_raises():
    mask = np.array([[1, 0]])
    with pytest.raises(ValueError, match="mutually exclusive"):
        Sensor(mask=mask, mod_x=4, mod_y=4)


def test_sparse_grid_with_coords_raises():
    coords = np.array([[0, 0]])
    with pytest.raises(ValueError, match="mutually exclusive"):
        Sensor(coords=coords, grid_shape=(1, 2), mod_x=4, mod_y=4)


def test_sparse_grid_str():
    sensor = Sensor(mod_x=4, mod_y=4)
    assert "sparse-grid" in str(sensor)
    assert "mod_x=4" in str(sensor)
