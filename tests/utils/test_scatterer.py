import numpy as np
import pytest

from fullwave.utils.scatterer import (
    _expand_scatterer_pixels,
    _expand_scatterer_pixels_3d,
    generate_resolution_based_scatterer,
    generate_scatterer_from_ratio_num_scatterer_to_wavelength,
)


class DummyGrid2D:
    def __init__(self, nx, ny, dt, ppw, c0=1500.0, f0=1e6):
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.is_3d = False
        self.ppw = ppw
        self.shape = (nx, ny)
        self.wavelength = c0 / f0
        self.dx = self.wavelength / ppw
        self.dy = self.wavelength / ppw
        self.domain_size = (nx * self.dx, ny * self.dy)


class DummyGrid3D:
    def __init__(self, nx, ny, nz, dt, ppw, c0=1500.0, f0=1e6):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dt = dt
        self.is_3d = True
        self.ppw = ppw
        self.shape = (nx, ny, nz)
        self.wavelength = c0 / f0
        self.dx = self.wavelength / ppw
        self.dy = self.wavelength / ppw
        self.dz = self.wavelength / ppw
        self.domain_size = (nx * self.dx, ny * self.dy, nz * self.dz)


def test_generate_scatterer_with_seed():
    """Test generate_scatterer with a seed."""
    grid = DummyGrid2D(nx=10, ny=10, dt=0.1, ppw=12)

    ratio_scatterer_num_to_wavelength = 0.3
    scatterer, scatterer_info = generate_scatterer_from_ratio_num_scatterer_to_wavelength(
        grid,
        ratio_scatterer_num_to_wavelength=ratio_scatterer_num_to_wavelength,
        scatter_value_std=0.08,
        seed=42,
    )
    num_per_wl = scatterer_info["num_scatterer_per_wavelength"]

    assert scatterer.shape == (10, 10)
    assert isinstance(num_per_wl, float)
    assert num_per_wl == ratio_scatterer_num_to_wavelength * grid.ppw


def test_generate_scatterer_with_rng():
    """Test generate_scatterer with a random number generator."""
    grid = DummyGrid2D(nx=10, ny=10, dt=0.1, ppw=12)
    ratio_scatterer_num_to_wavelength = 0.3

    rng = np.random.default_rng(seed=42)
    scatterer, scatterer_info = generate_scatterer_from_ratio_num_scatterer_to_wavelength(
        grid,
        ratio_scatterer_num_to_wavelength=ratio_scatterer_num_to_wavelength,
        scatter_value_std=0.08,
        rng=rng,
    )
    num_per_wl = scatterer_info["num_scatterer_per_wavelength"]

    assert scatterer.shape == (10, 10)
    assert isinstance(num_per_wl, float)
    assert num_per_wl == ratio_scatterer_num_to_wavelength * grid.ppw


def test_generate_scatterer_raises_when_both_seed_and_rng():
    """Test that providing both seed and rng raises ValueError."""
    grid = DummyGrid2D(nx=100, ny=100, dt=0.1, ppw=12)

    rng = np.random.default_rng(seed=42)

    with pytest.raises(ValueError, match="Provide either seed or rng, not both"):
        generate_scatterer_from_ratio_num_scatterer_to_wavelength(grid, seed=42, rng=rng)


def test_generate_scatterer_raises_when_neither_seed_nor_rng():
    """Test that providing neither seed nor rng raises ValueError."""
    grid = DummyGrid2D(nx=100, ny=100, dt=0.1, ppw=10)

    with pytest.raises(ValueError, match="Provide either seed or rng"):
        generate_scatterer_from_ratio_num_scatterer_to_wavelength(grid)


def test_generate_scatterer_reproducibility():
    """Test that using the same seed produces the same result."""
    grid = DummyGrid2D(nx=50, ny=50, dt=0.1, ppw=10)

    scatterer1, _ = generate_scatterer_from_ratio_num_scatterer_to_wavelength(grid, seed=123)
    scatterer2, _ = generate_scatterer_from_ratio_num_scatterer_to_wavelength(grid, seed=123)

    np.testing.assert_array_equal(scatterer1, scatterer2)


def test_generate_scatterer_different_seeds():
    """Test that different seeds produce different results."""
    grid = DummyGrid2D(nx=50, ny=50, dt=0.1, ppw=10)

    scatterer1, _ = generate_scatterer_from_ratio_num_scatterer_to_wavelength(grid, seed=123)
    scatterer2, _ = generate_scatterer_from_ratio_num_scatterer_to_wavelength(grid, seed=456)

    assert not np.array_equal(scatterer1, scatterer2)


def test_generate_scatterer_rng_reproducibility():
    """Test that using the same rng produces the same result."""
    grid = DummyGrid2D(nx=10, ny=10, dt=0.1, ppw=12)
    ratio_scatterer_num_to_wavelength = 0.3

    rng = np.random.default_rng(seed=42)
    scatterer1, _ = generate_scatterer_from_ratio_num_scatterer_to_wavelength(
        grid,
        ratio_scatterer_num_to_wavelength=ratio_scatterer_num_to_wavelength,
        scatter_value_std=0.08,
        rng=rng,
    )

    rng = np.random.default_rng(seed=42)
    scatterer2, _ = generate_scatterer_from_ratio_num_scatterer_to_wavelength(
        grid,
        ratio_scatterer_num_to_wavelength=ratio_scatterer_num_to_wavelength,
        scatter_value_std=0.08,
        rng=rng,
    )
    np.testing.assert_array_equal(scatterer1, scatterer2)


def test_generate_scatterer_rng_different():
    """Test that different rngs produce different results."""
    grid = DummyGrid2D(nx=10, ny=10, dt=0.1, ppw=12)
    ratio_scatterer_num_to_wavelength = 0.3

    rng1 = np.random.default_rng(seed=42)
    scatterer1, _ = generate_scatterer_from_ratio_num_scatterer_to_wavelength(
        grid,
        ratio_scatterer_num_to_wavelength=ratio_scatterer_num_to_wavelength,
        scatter_value_std=0.08,
        rng=rng1,
    )

    rng2 = np.random.default_rng(seed=43)
    scatterer2, _ = generate_scatterer_from_ratio_num_scatterer_to_wavelength(
        grid,
        ratio_scatterer_num_to_wavelength=ratio_scatterer_num_to_wavelength,
        scatter_value_std=0.08,
        rng=rng2,
    )

    assert not np.array_equal(scatterer1, scatterer2)


def test_generate_scatterer_same_rng_different_result():
    """Test that using the same rng twice produces different results."""
    grid = DummyGrid2D(nx=10, ny=10, dt=0.1, ppw=12)
    ratio_scatterer_num_to_wavelength = 0.3

    rng1 = np.random.default_rng(seed=42)
    scatterer1, _ = generate_scatterer_from_ratio_num_scatterer_to_wavelength(
        grid,
        ratio_scatterer_num_to_wavelength=ratio_scatterer_num_to_wavelength,
        scatter_value_std=0.08,
        rng=rng1,
    )
    scatterer2, _ = generate_scatterer_from_ratio_num_scatterer_to_wavelength(
        grid,
        ratio_scatterer_num_to_wavelength=ratio_scatterer_num_to_wavelength,
        scatter_value_std=0.08,
        rng=rng1,
    )

    assert not np.array_equal(scatterer1, scatterer2)


def test_generate_scatterer_3d():
    """Test generate_scatterer with a 3D grid."""
    grid = DummyGrid3D(nx=50, ny=50, nz=50, dt=0.1, ppw=12)

    ratio_scatterer_num_to_wavelength = 0.3
    scatterer, scatterer_info = generate_scatterer_from_ratio_num_scatterer_to_wavelength(
        grid,
        ratio_scatterer_num_to_wavelength=ratio_scatterer_num_to_wavelength,
        seed=42,
    )

    num_per_wl = scatterer_info["num_scatterer_per_wavelength"]

    assert scatterer.shape == (50, 50, 50)
    assert isinstance(num_per_wl, float)
    assert num_per_wl == 0.3 * grid.ppw


def test_generate_scatterer_values_distribution():
    """Test that scatterer values follow expected distribution."""
    grid = DummyGrid2D(nx=100, ny=100, dt=0.1, ppw=10)

    scatterer, _ = generate_scatterer_from_ratio_num_scatterer_to_wavelength(
        grid,
        ratio_scatterer_num_to_wavelength=0.5,
        scatter_value_std=0.1,
        seed=42,
    )

    # Most values should be 1.0 (non-scatterer locations)
    assert np.sum(scatterer == 1.0) > 0
    # Some values should be different (scatterer locations)
    assert np.sum(scatterer != 1.0) > 0
    # All values should be positive (normal distribution around 1.0)
    assert np.all(scatterer > 0)


def test_generate_resolution_based_scatterer():
    """Test generate_resolution_based_scatterer function."""
    grid = DummyGrid2D(nx=50, ny=50, dt=0.1, ppw=10, c0=1500.0, f0=1e6)
    num_scatterer = 100
    ncycles = 5

    scatter_map, scatter_info = generate_resolution_based_scatterer(
        grid,
        num_scatterer,
        ncycles,
        seed=42,
    )
    scatterer_count = scatter_info["scatterer_count"]
    scatterer_percent = scatter_info["ratio_scatterer_to_total_grid"]
    assert scatter_map.shape == (50, 50)
    assert isinstance(scatterer_count, int)
    assert isinstance(scatterer_percent, float)
    assert 0 <= scatterer_percent <= 100


def test_expand_scatterer_pixels_3d_with_radius_1():
    """Test 3D scatterer expansion with radius 1."""
    # Create a simple scatterer with one point
    scatterer = np.ones((10, 10, 10), dtype=float)
    scatterer[5, 5, 5] = 0.5

    expanded = _expand_scatterer_pixels_3d(scatterer, radius=1, use_rectangle_expansion=True)

    # Check that the center voxel is expanded
    assert expanded[5, 5, 5] == 0.5
    # Check that neighboring voxels are also affected
    assert expanded[4, 5, 5] == 0.5
    assert expanded[6, 5, 5] == 0.5
    assert expanded[5, 4, 5] == 0.5
    assert expanded[5, 6, 5] == 0.5
    assert expanded[5, 5, 4] == 0.5
    assert expanded[5, 5, 6] == 0.5


def test_expand_scatterer_pixels_3d_rectangular_vs_spherical():
    """Test difference between rectangular and spherical expansion in 3D."""
    scatterer = np.ones((10, 10, 10), dtype=float)
    scatterer[5, 5, 5] = 0.5

    expanded_rect = _expand_scatterer_pixels_3d(scatterer, radius=1, use_rectangle_expansion=True)
    expanded_sphere = _expand_scatterer_pixels_3d(
        scatterer,
        radius=1,
        use_rectangle_expansion=False,
    )

    # Both should affect the center
    assert expanded_rect[5, 5, 5] == 0.5
    assert expanded_sphere[5, 5, 5] == 0.5

    # Rectangular should create a cube
    assert expanded_rect[4, 4, 4] == 0.5

    # Spherical should not expand to corners (distance > radius)
    # Distance from (5,5,5) to (4,4,4) is sqrt(3) ≈ 1.73 > 1
    assert expanded_sphere[4, 4, 4] != 0.5


def test_expand_scatterer_pixels_3d_multiple_scatterers():
    """Test 3D expansion with multiple scatterers."""
    scatterer = np.ones((10, 10, 10), dtype=float)
    scatterer[3, 3, 3] = 0.3
    scatterer[7, 7, 7] = 0.7

    expanded = _expand_scatterer_pixels_3d(scatterer, radius=1, use_rectangle_expansion=True)

    # Check both scatterers are expanded
    assert expanded[3, 3, 3] == 0.3
    assert expanded[2, 3, 3] == 0.3
    assert expanded[7, 7, 7] == 0.7
    assert expanded[8, 7, 7] == 0.7


def test_expand_scatterer_pixels_3d_boundary_handling():
    """Test that 3D expansion handles boundaries correctly."""
    scatterer = np.ones((10, 10, 10), dtype=float)
    scatterer[0, 0, 0] = 0.5

    expanded = _expand_scatterer_pixels_3d(scatterer, radius=1, use_rectangle_expansion=True)

    # Should not go out of bounds
    assert expanded.shape == (10, 10, 10)
    assert expanded[0, 0, 0] == 0.5
    assert expanded[1, 0, 0] == 0.5
    assert expanded[0, 1, 0] == 0.5
    assert expanded[0, 0, 1] == 0.5


def test_expand_scatterer_pixels_3d_larger_radius():
    """Test 3D expansion with larger radius."""
    scatterer = np.ones((20, 20, 20), dtype=float)
    scatterer[10, 10, 10] = 0.5

    expanded = _expand_scatterer_pixels_3d(scatterer, radius=2, use_rectangle_expansion=True)

    # Check expanded region
    assert expanded[10, 10, 10] == 0.5
    assert expanded[8, 10, 10] == 0.5
    assert expanded[12, 10, 10] == 0.5
    assert expanded[10, 8, 10] == 0.5
    assert expanded[10, 12, 10] == 0.5
    assert expanded[10, 10, 8] == 0.5
    assert expanded[10, 10, 12] == 0.5


def test_expand_scatterer_pixels_3d_preserves_max_value():
    """Test that 3D expansion preserves maximum values when overlapping."""
    scatterer = np.ones((10, 10, 10), dtype=float)
    scatterer[4, 5, 5] = 0.3
    scatterer[6, 5, 5] = 0.7  # Higher value

    expanded = _expand_scatterer_pixels_3d(scatterer, radius=1, use_rectangle_expansion=True)

    # Overlapping region should have the maximum value
    assert expanded[5, 5, 5] == 0.7


def test_expand_scatterer_pixels_with_radius_1():
    """Test 2D scatterer expansion with radius 1."""
    # Create a simple scatterer with one point
    scatterer = np.ones((10, 10), dtype=float)
    scatterer[5, 5] = 0.5

    expanded = _expand_scatterer_pixels(scatterer, radius=1, use_rectangle_expansion=True)

    # Check that the center pixel is expanded
    assert expanded[5, 5] == 0.5
    # Check that neighboring pixels are also affected
    assert expanded[4, 5] == 0.5
    assert expanded[6, 5] == 0.5
    assert expanded[5, 4] == 0.5
    assert expanded[5, 6] == 0.5

    # other neighbors should remain unaffected
    assert expanded[4, 4] != 1.0
    assert expanded[4, 6] != 1.0
    assert expanded[6, 4] != 1.0
    assert expanded[6, 6] != 1.0


def test_expand_scatterer_pixels_rectangular_vs_circular():
    """Test difference between rectangular and circular expansion in 2D."""
    scatterer = np.ones((10, 10), dtype=float)
    scatterer[5, 5] = 0.5

    expanded_rect = _expand_scatterer_pixels(scatterer, radius=1, use_rectangle_expansion=True)
    expanded_circle = _expand_scatterer_pixels(scatterer, radius=1, use_rectangle_expansion=False)

    # Both should affect the center
    assert expanded_rect[5, 5] == 0.5
    assert expanded_circle[5, 5] == 0.5

    # Rectangular should create a square
    assert expanded_rect[4, 4] == 0.5

    # Circular should not expand to corners (distance > radius)
    # Distance from (5,5) to (4,4) is sqrt(2) ≈ 1.41 > 1
    assert expanded_circle[4, 4] == 1.0


def test_expand_scatterer_pixels_multiple_scatterers():
    """Test 2D expansion with multiple scatterers."""
    scatterer = np.ones((10, 10), dtype=float)
    scatterer[3, 3] = 0.3
    scatterer[7, 7] = 0.7

    expanded = _expand_scatterer_pixels(scatterer, radius=1, use_rectangle_expansion=True)

    # Check both scatterers are expanded
    assert expanded[3, 3] == 0.3
    assert expanded[2, 3] == 0.3
    assert expanded[7, 7] == 0.7
    assert expanded[8, 7] == 0.7


def test_expand_scatterer_pixels_boundary_handling():
    """Test that 2D expansion handles boundaries correctly."""
    scatterer = np.ones((10, 10), dtype=float)
    scatterer[0, 0] = 0.5

    expanded = _expand_scatterer_pixels(scatterer, radius=1, use_rectangle_expansion=True)

    # Should not go out of bounds
    assert expanded.shape == (10, 10)
    assert expanded[0, 0] == 0.5
    assert expanded[1, 0] == 0.5
    assert expanded[0, 1] == 0.5


def test_expand_scatterer_pixels_larger_radius():
    """Test 2D expansion with larger radius."""
    scatterer = np.ones((20, 20), dtype=float)
    scatterer[10, 10] = 0.5

    expanded = _expand_scatterer_pixels(scatterer, radius=2, use_rectangle_expansion=True)

    # Check expanded region
    assert expanded[10, 10] == 0.5
    assert expanded[8, 10] == 0.5
    assert expanded[12, 10] == 0.5
    assert expanded[10, 8] == 0.5
    assert expanded[10, 12] == 0.5

    # other neighbors should remain unaffected
    assert expanded[7, 7] == 1.0
    assert expanded[7, 12] == 1.0
    assert expanded[12, 7] == 1.0


def test_expand_scatterer_pixels_preserves_max_value():
    """Test that 2D expansion preserves maximum values when overlapping."""
    scatterer = np.ones((10, 10), dtype=float)
    scatterer[4, 5] = 0.3
    scatterer[6, 5] = 0.7  # Higher value

    expanded = _expand_scatterer_pixels(scatterer, radius=1, use_rectangle_expansion=True)

    # Overlapping region should have the maximum value
    assert expanded[5, 5] == 0.7


def test_expand_scatterer_pixels_circular_expansion():
    """Test circular expansion pattern in 2D."""
    scatterer = np.ones((10, 10), dtype=float)
    scatterer[5, 5] = 0.5

    expanded = _expand_scatterer_pixels(scatterer, radius=2, use_rectangle_expansion=False)

    # Center should be affected
    assert expanded[5, 5] == 0.5
    # Points within radius should be affected
    assert expanded[5, 3] == 0.5  # distance = 2
    assert expanded[3, 5] == 0.5  # distance = 2
    # Diagonal corners should not be affected (distance > 2)
    # Distance from (5,5) to (3,3) is sqrt(8) ≈ 2.83 > 2
    assert expanded[3, 3] == 1.0
