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


# --- Coordinate input mode tests ---


def test_coords_input_2d():
    coords = np.array([[0, 1]])
    p0 = np.array([[1.0, 2.0, 3.0]])
    grid_shape = (2, 2)
    src = Source(p0, coords=coords, grid_shape=grid_shape)
    assert src.n_sources == 1
    assert src.grid_shape == grid_shape
    assert not src.is_3d
    np.testing.assert_array_equal(src.incoords, coords)


def test_coords_input_3d():
    coords = np.array([[0, 1, 2], [3, 4, 5]])
    p0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    grid_shape = (10, 10, 10)
    src = Source(p0, coords=coords, grid_shape=grid_shape)
    assert src.n_sources == 2
    assert src.is_3d


def test_coords_input_validate():
    coords = np.array([[0, 1]])
    p0 = np.array([[1.0]])
    grid_shape = (2, 2)
    src = Source(p0, coords=coords, grid_shape=grid_shape)
    src.validate(grid_shape)


def test_coords_equivalence_with_mask():
    p_mask = np.array([[False, True], [True, False]])
    p0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    src_from_mask = Source(p0, p_mask)
    src_from_coords = Source(
        p0,
        coords=src_from_mask.incoords,
        grid_shape=src_from_mask.grid_shape,
    )
    assert src_from_mask.n_sources == src_from_coords.n_sources
    np.testing.assert_array_equal(src_from_mask.incoords, src_from_coords.incoords)
    np.testing.assert_array_equal(src_from_mask.mask, src_from_coords.mask)


def test_coords_without_grid_shape_raises():
    coords = np.array([[0, 1]])
    p0 = np.array([[1.0]])
    with pytest.raises(ValueError, match="grid_shape is required"):
        Source(p0, coords=coords)


def test_coords_and_mask_raises():
    mask = np.array([[False, True]])
    coords = np.array([[0, 1]])
    p0 = np.array([[1.0]])
    with pytest.raises(ValueError, match="mutually exclusive"):
        Source(p0, mask=mask, coords=coords, grid_shape=(1, 2))


def test_no_mask_or_coords_raises():
    p0 = np.array([[1.0]])
    with pytest.raises(
        ValueError,  # noqa: PT011
    ):
        Source(p0)


# --- Additive source (p0_additive) tests ---


def test_p0_additive_optional():
    """Source without p0_additive has p0_additive None."""
    p_mask = np.array([[False, True], [True, False]])
    p0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    src = Source(p0, p_mask)
    assert src.p0_additive is None


def test_p0_additive_stored_and_same_shape():
    """Source with p0_additive stores it and validates shape."""
    p_mask = np.array([[False, True], [True, False]])
    p0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    p0_add = np.array([[0.1, 0.2], [0.3, 0.4]])
    src = Source(p0, p_mask, p0_additive=p0_add)
    assert src.p0_additive is not None
    np.testing.assert_array_almost_equal(src.p0_additive, p0_add)
    assert "p0_additive" in str(src)


def test_p0_additive_shape_mismatch_raises():
    """Source with p0_additive shape different from incoords/incoords_add raises ValueError."""
    p_mask = np.array([[False, True], [True, False]])
    p0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    p0_add_wrong = np.array([[0.1, 0.2]])  # 1 row instead of 2
    with pytest.raises(ValueError, match="p0_additive shape.*must have.*rows"):
        Source(p0, p_mask, p0_additive=p0_add_wrong)


def test_additive_only_with_mask():
    """Source with p0=None and p0_additive uses zeros for icmat (additive-only / soft IC)."""
    p_mask = np.array([[False, True], [True, False]])
    p0_add = np.array([[0.1, 0.2], [0.3, 0.4]])
    src = Source(None, p_mask, p0_additive=p0_add)
    assert src.p0_additive is not None
    np.testing.assert_array_almost_equal(src.p0_additive, p0_add)
    np.testing.assert_array_almost_equal(src.p0, np.zeros_like(p0_add))
    assert np.array_equal(src.icmat, src.p0)


def test_additive_only_with_coords():
    """Source with p0=None, coords and p0_additive (additive-only)."""
    coords = np.array([[0, 1], [1, 0]])
    grid_shape = (2, 2)
    p0_add = np.array([[0.1, 0.2], [0.3, 0.4]])
    src = Source(None, coords=coords, grid_shape=grid_shape, p0_additive=p0_add)
    assert src.n_sources == 2
    np.testing.assert_array_almost_equal(src.p0, np.zeros_like(p0_add))
    np.testing.assert_array_almost_equal(src.p0_additive, p0_add)


def test_both_p0_and_p0_additive_none_raises():
    """Source with both p0 and p0_additive None raises ValueError."""
    p_mask = np.array([[False, True]])
    with pytest.raises(ValueError, match="At least one of p0 or p0_additive must be provided"):
        Source(None, p_mask)


def test_incoords_add_defaults_to_incoords():
    """When p0_additive is given and incoords_add is None, incoords_add equals incoords."""
    p_mask = np.array([[False, True], [True, False]])
    p0 = np.array([[1.0, 2.0], [3.0, 4.0]])
    p0_add = np.array([[0.1, 0.2], [0.3, 0.4]])
    src = Source(p0, p_mask, p0_additive=p0_add)
    assert src.incoords_add is not None
    np.testing.assert_array_equal(src.incoords_add, src.incoords)
    assert src.n_sources_add == 2


def test_incoords_add_separate_from_incoords():
    """Hard source at one point, soft (additive) at another; incoords_add separate from incoords."""
    # One hard source at (0,1), one soft at (1,0)
    coords_hard = np.array([[0, 1]])
    incoords_add = np.array([[1, 0]])
    grid_shape = (2, 2)
    p0 = np.array([[1.0, 2.0]])  # (1, nt) for hard
    p0_add = np.array([[0.1, 0.2]])  # (1, nt) for soft
    src = Source(
        p0,
        coords=coords_hard,
        grid_shape=grid_shape,
        p0_additive=p0_add,
        coords_additive=incoords_add,
    )
    assert src.n_sources == 1
    assert src.n_sources_add == 1
    np.testing.assert_array_equal(src.incoords, coords_hard)
    np.testing.assert_array_equal(src.incoords_add, incoords_add)
    np.testing.assert_array_almost_equal(src.p0, p0)
    np.testing.assert_array_almost_equal(src.p0_additive, p0_add)


def test_incoords_add_additive_only_with_explicit_coords():
    """Additive-only source with grid_shape and incoords_add (no mask/coords)."""
    grid_shape = (2, 2)
    incoords_add = np.array([[0, 1], [1, 0]])
    p0_add = np.array([[0.1, 0.2], [0.3, 0.4]])
    src = Source(
        None,
        grid_shape=grid_shape,
        coords_additive=incoords_add,
        p0_additive=p0_add,
    )
    assert src.n_sources == 2
    assert src.n_sources_add == 2
    np.testing.assert_array_equal(src.incoords, incoords_add)
    np.testing.assert_array_equal(src.incoords_add, incoords_add)
    np.testing.assert_array_almost_equal(src.p0, np.zeros_like(p0_add))
    np.testing.assert_array_almost_equal(src.p0_additive, p0_add)


def test_p0_additive_incoords_add_length_mismatch_raises():
    """p0_additive and incoords_add must have same number of rows."""
    coords_hard = np.array([[0, 1]])
    incoords_add = np.array([[1, 0], [1, 1]])  # 2 points
    grid_shape = (2, 2)
    p0 = np.array([[1.0, 2.0]])
    p0_add = np.array([[0.1, 0.2]])  # 1 row
    with pytest.raises(ValueError, match="incoords_add"):
        Source(
            p0,
            coords=coords_hard,
            grid_shape=grid_shape,
            p0_additive=p0_add,
            coords_additive=incoords_add,
        )
