import numpy as np
import pytest

from fullwave.grid import Grid


def test_properties_2d():
    domain_size = (0.12, 0.12)  # meters
    f0 = 1e6
    duration = 1e-5
    c0 = 1540
    ppw = 12
    cfl = 0.2
    grid = Grid(domain_size, f0, duration, c0=c0, ppw=ppw, cfl=cfl)

    wavelength = c0 / f0
    dx_expected = wavelength / ppw
    dt_expected = cfl * dx_expected / c0

    # angular frequency
    assert np.isclose(grid.omega, 2 * np.pi * f0)
    # wavelength
    assert np.isclose(grid.wavelength, wavelength)
    # grid points in x and y, nz should be 0 for 2D
    nx_expected = int(domain_size[0] / wavelength * ppw)
    ny_expected = int(domain_size[1] / wavelength * ppw)
    assert grid.nx == nx_expected
    assert grid.ny == ny_expected
    assert grid.nz == 0
    # grid spacing
    assert np.isclose(grid.dx, dx_expected)
    assert np.isclose(grid.dy, dx_expected)
    assert grid.dz == 0
    # time step and number of time steps
    assert np.isclose(grid.dt, dt_expected)
    nt_expected = int(duration / dt_expected)
    assert grid.nt == nt_expected


def test_properties_3d():
    domain_size = (0.12, 0.12, 0.12)  # meters
    f0 = 1e6
    duration = 1e-5
    c0 = 1540
    ppw = 12
    cfl = 0.2
    grid = Grid(domain_size, f0, duration, c0=c0, ppw=ppw, cfl=cfl)

    wavelength = c0 / f0
    dx_expected = wavelength / ppw
    dt_expected = cfl * dx_expected / c0

    # angular frequency
    assert np.isclose(grid.omega, 2 * np.pi * f0)
    # wavelength
    assert np.isclose(grid.wavelength, wavelength)
    # grid points in all dimensions
    nx_expected = int(domain_size[0] / wavelength * ppw)
    ny_expected = int(domain_size[1] / wavelength * ppw)
    nz_expected = int(domain_size[2] / wavelength * ppw)
    assert grid.nx == nx_expected
    assert grid.ny == ny_expected
    assert grid.nz == nz_expected
    # grid spacing
    assert np.isclose(grid.dx, dx_expected)
    assert np.isclose(grid.dy, dx_expected)
    assert np.isclose(grid.dz, dx_expected)
    # time step and number of time steps
    assert np.isclose(grid.dt, dt_expected)
    nt_expected = int(duration / dt_expected)
    assert grid.nt == nt_expected


def test_invalid_domain_size():
    # domain_size must be a 2-tuple or 3-tuple
    with pytest.raises(AssertionError):
        Grid((0.1,), 1e6, 1e-5)
    with pytest.raises(AssertionError):
        Grid((0.1, 0.1, 0.1, 0.1), 1e6, 1e-5)
