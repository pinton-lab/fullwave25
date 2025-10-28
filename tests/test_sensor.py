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
