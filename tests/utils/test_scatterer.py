import numpy as np
import pytest

from fullwave.utils.scatterer import (
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
