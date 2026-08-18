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
        ValueError,
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
    with pytest.raises(ValueError, match="At least one of p0, p0_additive"):
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
    """An additive-only source has no hard nodes.

    The hard node list used to take the additive positions, which made the
    solver assign a zero pressure at them on every step. That is a
    pressure-release wall standing on the source.
    """
    grid_shape = (2, 2)
    incoords_add = np.array([[0, 1], [1, 0]])
    p0_add = np.array([[0.1, 0.2], [0.3, 0.4]])
    src = Source(
        None,
        grid_shape=grid_shape,
        coords_additive=incoords_add,
        p0_additive=p0_add,
    )
    assert src.n_sources == 0
    assert src.n_sources_add == 2
    assert src.incoords.shape == (0, 2)
    assert src.p0.shape == (0, p0_add.shape[1])
    np.testing.assert_array_equal(src.incoords_add, incoords_add)
    np.testing.assert_array_almost_equal(src.p0_additive, p0_add)


def test_hard_and_additive_together_keep_their_own_nodes():
    """A model may drive a clamped source and an added source at once."""
    hard = np.zeros((6, 4), dtype=bool)
    hard[0] = True
    additive = np.zeros((6, 4), dtype=bool)
    additive[3] = True
    src = Source(
        p0=np.ones((4, 5)),
        mask=hard,
        p0_additive=2 * np.ones((4, 5)),
        mask_additive=additive,
    )
    assert src.n_sources == 4
    assert src.n_sources_add == 4
    np.testing.assert_array_equal(np.unique(src.incoords[:, 0]), [0])
    np.testing.assert_array_equal(np.unique(src.incoords_add[:, 0]), [3])
    assert src.p0.max() == 1.0
    assert src.p0_additive.max() == 2.0


def test_additive_only_with_mask_has_no_hard_nodes():
    """The same holds when the additive positions come from a mask."""
    mask_add = np.zeros((3, 4), dtype=bool)
    mask_add[0] = True
    p0_add = np.ones((4, 5))
    src = Source(p0_additive=p0_add, mask_additive=mask_add)
    assert src.n_sources == 0
    assert src.n_sources_add == 4
    assert src.grid_shape == (3, 4)


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


# --- Velocity source (u0/v0/w0) tests ---


class TestVelocitySource:
    """Tests for velocity source components (u0, v0, w0)."""

    _grid_shape = (4, 4)
    _nt = 5

    def _p0(self, n: int = 1) -> np.ndarray:
        return np.ones((n, self._nt), dtype=np.float64)

    def _base_source(self, n: int = 1) -> Source:
        """Minimal hard pressure source at (0,0)..(n-1,0)."""
        coords = np.array([[i, 0] for i in range(n)])
        return Source(self._p0(n), coords=coords, grid_shape=self._grid_shape)

    # ---- construction with coords ----

    def test_u0_with_coords(self):
        """u0 with coords_u stores incoords_u and u0."""
        coords_u = np.array([[1, 2]])
        u0 = self._p0()
        src = Source(
            self._p0(),
            coords=np.array([[0, 0]]),
            grid_shape=self._grid_shape,
            u0=u0,
            coords_u=coords_u,
        )
        assert src.n_sources_u == 1
        np.testing.assert_array_equal(src.incoords_u, coords_u)
        np.testing.assert_array_equal(src.u0, u0)

    def test_u0_with_mask(self):
        """u0 with mask_u is equivalent to coords extracted from mask."""
        mask_u = np.zeros(self._grid_shape, dtype=bool)
        mask_u[1, 2] = True
        u0 = self._p0()
        src = Source(
            self._p0(),
            coords=np.array([[0, 0]]),
            grid_shape=self._grid_shape,
            u0=u0,
            mask_u=mask_u,
        )
        expected_coords = np.argwhere(mask_u)
        assert src.n_sources_u == 1
        np.testing.assert_array_equal(src.incoords_u, expected_coords)

    # ---- all three components ----

    def test_all_three_velocity_components(self):
        """All three u/v/w components can be specified simultaneously."""
        coords_u = np.array([[0, 1]])
        coords_v = np.array([[1, 0], [1, 1]])
        coords_w = np.array([[2, 2], [3, 3], [0, 3]])
        src = Source(
            self._p0(),
            coords=np.array([[0, 0]]),
            grid_shape=self._grid_shape,
            u0=np.ones((1, self._nt)),
            coords_u=coords_u,
            v0=np.ones((2, self._nt)),
            coords_v=coords_v,
            w0=np.ones((3, self._nt)),
            coords_w=coords_w,
        )
        assert src.n_sources_u == 1
        assert src.n_sources_v == 2
        assert src.n_sources_w == 3

    # ---- partial (only u provided) ----

    def test_only_u_provided(self):
        """Only u component; v and w remain None with count 0."""
        coords_u = np.array([[0, 1]])
        src = Source(
            self._p0(),
            coords=np.array([[0, 0]]),
            grid_shape=self._grid_shape,
            u0=self._p0(),
            coords_u=coords_u,
        )
        assert src.n_sources_u == 1
        assert src.n_sources_v == 0
        assert src.n_sources_w == 0
        assert src.v0 is None
        assert src.w0 is None
        assert src.incoords_v is None
        assert src.incoords_w is None

    # ---- error cases ----

    def test_u0_without_coords_or_mask_raises(self):
        """Providing u0 without coords_u or mask_u must raise ValueError."""
        with pytest.raises(ValueError, match="neither coords_u nor mask_u"):
            Source(
                self._p0(),
                coords=np.array([[0, 0]]),
                grid_shape=self._grid_shape,
                u0=self._p0(),
            )

    def test_coords_u_and_mask_u_raises(self):
        """Providing both coords_u and mask_u must raise ValueError."""
        mask_u = np.zeros(self._grid_shape, dtype=bool)
        mask_u[0, 1] = True
        coords_u = np.array([[0, 1]])
        with pytest.raises(ValueError, match="mutually exclusive"):
            Source(
                self._p0(),
                coords=np.array([[0, 0]]),
                grid_shape=self._grid_shape,
                u0=self._p0(),
                coords_u=coords_u,
                mask_u=mask_u,
            )

    def test_coords_u_without_u0_raises(self):
        """Providing coords_u but no u0 must raise ValueError."""
        with pytest.raises(ValueError, match="without u0"):
            Source(
                self._p0(),
                coords=np.array([[0, 0]]),
                grid_shape=self._grid_shape,
                coords_u=np.array([[0, 1]]),
            )

    def test_u0_row_mismatch_raises(self):
        """u0 row count not matching coords_u raises ValueError."""
        coords_u = np.array([[0, 1], [1, 0]])  # 2 points
        u0_wrong = self._p0(1)  # 1 row
        with pytest.raises(ValueError, match="u0 has 1 rows"):
            Source(
                self._p0(),
                coords=np.array([[0, 0]]),
                grid_shape=self._grid_shape,
                u0=u0_wrong,
                coords_u=coords_u,
            )

    # ---- __str__ includes velocity info ----

    def test_str_includes_velocity_component(self):
        """__str__ reports velocity components when present."""
        coords_u = np.array([[0, 1]])
        src = Source(
            self._p0(),
            coords=np.array([[0, 0]]),
            grid_shape=self._grid_shape,
            u0=self._p0(),
            coords_u=coords_u,
        )
        assert "u0" in str(src)

    # ---- combined with hard pressure source ----

    def test_combined_pressure_and_velocity(self):
        """Hard pressure source and velocity source can coexist."""
        coords_p = np.array([[0, 0], [0, 1]])
        coords_u = np.array([[1, 0]])
        src = Source(
            self._p0(2),
            coords=coords_p,
            grid_shape=self._grid_shape,
            u0=self._p0(1),
            coords_u=coords_u,
        )
        assert src.n_sources == 2
        assert src.n_sources_u == 1

    # ---- velocity-only (no hard pressure source) ----

    def test_velocity_only_no_pressure_source(self):
        """Source with only u0 and no p0 has n_sources==0 and empty p0."""
        coords_u = np.array([[0, 1], [1, 0]])
        src = Source(
            grid_shape=self._grid_shape,
            u0=self._p0(2),
            coords_u=coords_u,
        )
        assert src.n_sources == 0
        assert src.n_sources_u == 2
        assert src.p0.shape == (0, self._nt)
        assert src.p0_additive is None
        assert src.incoords.shape == (0, 2)

    def test_velocity_only_missing_grid_shape_raises(self):
        """Velocity-only source without grid_shape raises ValueError."""
        with pytest.raises(ValueError, match="grid_shape is required"):
            Source(u0=self._p0(), coords_u=np.array([[0, 1]]))

    def test_velocity_only_validate_passes(self):
        """validate() accepts a velocity-only source."""
        src = Source(
            grid_shape=self._grid_shape,
            u0=self._p0(),
            coords_u=np.array([[0, 1]]),
        )
        src.validate(self._grid_shape)

    def test_all_none_raises(self):
        """No p0, no p0_additive, no velocity → ValueError."""
        with pytest.raises(ValueError):
            Source(grid_shape=self._grid_shape)
