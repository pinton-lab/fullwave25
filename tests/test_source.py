import numpy as np
import pytest

from fullwave.source import Source


def test_post_init():
    # p0 has one element; p_mask has one True so that sum is 1.
    p0 = np.array([1])
    p_mask = np.array([[False, True], [False, False]])
    src = Source(p0, p_mask)
    # p0 and p_mask have been coerced to at least 2d.
    assert src.p0.ndim == 2
    assert src.mask.ndim == 2
    # Verify that the incoords property matches the True position in p_mask.
    expected_coords = np.argwhere(p_mask)
    assert np.array_equal(src.incoords, expected_coords)


def test_validate_success():
    p_mask = np.array([[False, True], [False, False]])
    p0 = np.array([[2, 3]])  # positive pressure source.
    src = Source(p0, p_mask)
    grid_shape = p_mask.shape
    # This should pass the validations.
    src.validate(grid_shape)
    # Check that icmat property returns p0.
    assert np.array_equal(src.icmat, src.p0)


def test_validate_wrong_grid_shape():
    p_mask = np.array([[False, True], [False, False]])
    p0 = np.array([3])
    src = Source(p0, p_mask)
    wrong_shape = (3, 3)
    with pytest.raises(AssertionError):
        src.validate(np.array(wrong_shape))


def test_validate_no_active_source():
    # Create a p_mask with no True values.
    p_mask = np.array([[False, False], [False, False]])
    # To satisfy __post_init__, p0 must have rows equal to the sum of p_mask (0).
    p0 = np.empty((0, 1))
    src = Source(p0, p_mask)
    grid_shape = p_mask.shape
    with pytest.raises(AssertionError):
        src.validate(grid_shape)
