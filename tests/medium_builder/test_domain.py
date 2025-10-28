from pathlib import Path

import numpy as np
import pytest

from fullwave.medium_builder.domain import Domain
from fullwave.utils import check_functions

# Override check_instance to bypass instance checking for testing purposes.
check_functions.check_instance = lambda instance, cls: None  # noqa: ARG005


class DummyGrid2D:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.is_3d = False


class DummyDomain(Domain):
    def _setup_base_geometry(self) -> np.ndarray:
        return np.full((self.nx, self.ny), 1.0)

    def _setup_sound_speed(self) -> np.ndarray:
        return np.full((self.nx, self.ny), 2.0)

    def _setup_density(self) -> np.ndarray:
        return np.full((self.nx, self.ny), 3.0)

    def _setup_alpha_coeff(self) -> np.ndarray:
        return np.full((self.nx, self.ny), 4.0)

    def _setup_alpha_power(self) -> np.ndarray:
        return np.full((self.nx, self.ny), 5.0)

    def _setup_beta(self) -> np.ndarray:
        return np.full((self.nx, self.ny), 6.0)

    def _setup_air(self) -> np.ndarray:
        return np.full((self.nx, self.ny), 7.0)


@pytest.fixture
def grid():
    # Create a dummy grid with fixed dimensions.
    return DummyGrid2D(10, 15)


@pytest.fixture
def domain(grid):
    # Instantiate DummyDomain with the dummy grid.
    return DummyDomain(name="dummy", grid=grid)


def test_domain_attributes_shape(domain, grid):
    # Ensure that all attributes have the correct shape.
    expected_shape = (grid.nx, grid.ny)
    attrs = [
        domain.base_geometry,
        domain.sound_speed,
        domain.density,
        domain.alpha_coeff,
        domain.alpha_power,
        domain.beta,
        domain.air,
    ]
    for attr in attrs:
        assert attr.shape == expected_shape


def test_domain_attribute_values(domain, grid):
    # Check that each method returns the expected constant values.
    np.testing.assert_array_equal(domain.base_geometry, np.full((grid.nx, grid.ny), 1.0))
    np.testing.assert_array_equal(domain.sound_speed, np.full((grid.nx, grid.ny), 2.0))
    np.testing.assert_array_equal(domain.density, np.full((grid.nx, grid.ny), 3.0))
    np.testing.assert_array_equal(domain.alpha_coeff, np.full((grid.nx, grid.ny), 4.0))
    np.testing.assert_array_equal(domain.alpha_power, np.full((grid.nx, grid.ny), 5.0))
    np.testing.assert_array_equal(domain.beta, np.full((grid.nx, grid.ny), 6.0))
    np.testing.assert_array_equal(domain.air, np.full((grid.nx, grid.ny), 7.0))


def test_plot_default(monkeypatch, domain):
    calls = []

    def dummy_plot(self, export_path, show):
        calls.append((export_path, show))

    # Patch the Medium.plot method on the class of the medium returned by the property.
    monkeypatch.setattr(domain.medium.__class__, "plot", dummy_plot)
    domain.plot()
    expected_export_path = Path("./temp/temp.png")
    assert len(calls) == 1
    # Check that the default export_path is used.
    assert calls[0][0] == expected_export_path
    # Check that the default show value (False) is used.
    assert calls[0][1] is False


def test_plot_custom(monkeypatch, domain):
    calls = []

    def dummy_plot(self, export_path, show):
        calls.append((export_path, show))

    monkeypatch.setattr(domain.medium.__class__, "plot", dummy_plot)
    custom_export = "custom/path.png"
    domain.plot(export_path=custom_export, show=True)
    assert len(calls) == 1
    assert calls[0][0] == custom_export
    assert calls[0][1] is True
