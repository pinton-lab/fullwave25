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
    with pytest.raises(ValueError, match="Either mask or coords"):
        Sensor()
