import numpy as np
import pytest

from fullwave.medium_builder.presets.domain_background import BackgroundDomain
from fullwave.utils import check_functions

# Override check_instance to bypass instance checking for testing purposes.
check_functions.check_instance = lambda instance, cls: None  # noqa: ARG005


# Dummy grid class for testing
class DummyGrid2D:
    def __init__(self, nx, ny):
        self.nx = nx
        self.ny = ny
        self.is_3d = False


# Dummy material properties for testing
class DummyMaterialProperties:
    def __init__(self, sound_speed=1500.0, density=1.0, alpha_coeff=0.5, alpha_power=1.5, beta=0.3):
        self.sound_speed = sound_speed
        self.density = density
        self.alpha_coeff = alpha_coeff
        self.alpha_power = alpha_power
        self.beta = beta


def get_named_properties(value):
    """Build a dictionary of properties for named background."""
    return {
        "sound_speed": value["sound_speed"],
        "density": value["density"],
        "alpha_coeff": value["alpha_coeff"],
        "alpha_power": value["alpha_power"],
        "beta": value["beta"],
    }


@pytest.fixture
def grid():
    return DummyGrid2D(nx=10, ny=20)


@pytest.fixture
def default_material():
    return DummyMaterialProperties(
        sound_speed=1500.0,
        density=1.0,
        alpha_coeff=0.5,
        alpha_power=1.5,
        beta=0.3,
    )


@pytest.fixture
def named_material():
    mat = DummyMaterialProperties(
        sound_speed=1500.0,
        density=1.0,
        alpha_coeff=0.5,
        alpha_power=1.5,
        beta=0.3,
    )
    # Create a custom property dictionary.
    custom_values = {
        "sound_speed": 2000.0,
        "density": 1.5,
        "alpha_coeff": 0.7,
        "alpha_power": 1.8,
        "beta": 0.4,
    }
    mat.custom = custom_values
    return mat


def test_setup_base_geometry(grid, default_material):
    bd = BackgroundDomain(grid=grid, material_properties=default_material)
    geo = bd._setup_base_geometry()  # noqa: SLF001
    assert geo.shape == (grid.nx, grid.ny)
    np.testing.assert_array_equal(geo, np.ones((grid.nx, grid.ny)))


def test_setup_sound_speed_default(grid, default_material):
    bd = BackgroundDomain(grid=grid, material_properties=default_material)
    sound_speed = bd._setup_sound_speed()  # noqa: SLF001
    expected = np.ones((grid.nx, grid.ny)) * default_material.sound_speed
    np.testing.assert_array_almost_equal(sound_speed, expected)


def test_setup_sound_speed_named(grid, named_material):
    bd = BackgroundDomain(
        grid=grid,
        background_property_name="custom",
        material_properties=named_material,
    )
    sound_speed = bd._setup_sound_speed()  # noqa: SLF001
    expected_value = named_material.custom["sound_speed"]
    expected = np.ones((grid.nx, grid.ny)) * expected_value
    np.testing.assert_array_almost_equal(sound_speed, expected)


def test_setup_density_default(grid, default_material):
    bd = BackgroundDomain(grid=grid, material_properties=default_material)
    density = bd._setup_density()  # noqa: SLF001
    expected = np.ones((grid.nx, grid.ny)) * default_material.density
    np.testing.assert_array_almost_equal(density, expected)


def test_setup_density_named(grid, named_material):
    bd = BackgroundDomain(
        grid=grid,
        background_property_name="custom",
        material_properties=named_material,
    )
    density = bd._setup_density()  # noqa: SLF001
    expected_value = named_material.custom["density"]
    expected = np.ones((grid.nx, grid.ny)) * expected_value
    np.testing.assert_array_almost_equal(density, expected)


def test_setup_alpha_coeff_default(grid, default_material):
    bd = BackgroundDomain(grid=grid, material_properties=default_material)
    alpha_coeff = bd._setup_alpha_coeff()  # noqa: SLF001
    expected = np.ones((grid.nx, grid.ny)) * default_material.alpha_coeff
    np.testing.assert_array_almost_equal(alpha_coeff, expected)


def test_setup_alpha_coeff_named(grid, named_material):
    bd = BackgroundDomain(
        grid=grid,
        background_property_name="custom",
        material_properties=named_material,
    )
    alpha_coeff = bd._setup_alpha_coeff()  # noqa: SLF001
    expected_value = named_material.custom["alpha_coeff"]
    expected = np.ones((grid.nx, grid.ny)) * expected_value
    np.testing.assert_array_almost_equal(alpha_coeff, expected)


def test_setup_alpha_power_default(grid, default_material):
    bd = BackgroundDomain(grid=grid, material_properties=default_material)
    alpha_power = bd._setup_alpha_power()  # noqa: SLF001
    expected = np.ones((grid.nx, grid.ny)) * default_material.alpha_power
    np.testing.assert_array_almost_equal(alpha_power, expected)


def test_setup_alpha_power_named(grid, named_material):
    bd = BackgroundDomain(
        grid=grid,
        background_property_name="custom",
        material_properties=named_material,
    )
    alpha_power = bd._setup_alpha_power()  # noqa: SLF001
    expected_value = named_material.custom["alpha_power"]
    expected = np.ones((grid.nx, grid.ny)) * expected_value
    np.testing.assert_array_almost_equal(alpha_power, expected)


def test_setup_beta_default(grid, default_material):
    bd = BackgroundDomain(grid=grid, material_properties=default_material)
    beta = bd._setup_beta()  # noqa: SLF001
    expected = np.ones((grid.nx, grid.ny)) * default_material.beta
    np.testing.assert_array_almost_equal(beta, expected)


def test_setup_beta_named(grid, named_material):
    bd = BackgroundDomain(
        grid=grid,
        background_property_name="custom",
        material_properties=named_material,
    )
    beta = bd._setup_beta()  # noqa: SLF001
    expected_value = named_material.custom["beta"]
    expected = np.ones((grid.nx, grid.ny)) * expected_value
    np.testing.assert_array_almost_equal(beta, expected)
