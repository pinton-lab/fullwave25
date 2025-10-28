import numpy as np

from fullwave.constants import MaterialProperties
from fullwave.medium_builder.domain import Domain
from fullwave.medium_builder.medium_builder import MediumBuilder


class FakeDomain(Domain):
    def __init__(self, name, geometry, maps):
        self.name = name
        self.base_geometry = geometry
        self.sound_speed = maps["sound_speed"]
        self.density = maps["density"]
        self.beta = maps["beta"]
        self.alpha_coeff = maps["alpha_coeff"]
        self.alpha_power = maps["alpha_power"]
        if "air" in maps:
            self.air = maps["air"]

    def _setup_base_geometry(self):
        return self.base_geometry

    def _setup_sound_speed(self):
        return self.sound_speed

    def _setup_density(self):
        return self.density

    def _setup_beta(self):
        return self.beta

    def _setup_alpha_coeff(self):
        return self.alpha_coeff

    def _setup_alpha_power(self):
        return self.alpha_power

    def _setup_air(self):
        return self.air


def create_maps(shape, value):
    # creates a dictionary of maps filled with the given value
    maps = {
        "sound_speed": np.full(shape, value),
        "density": np.full(shape, value + 1),
        "beta": np.full(shape, value + 2),
        "alpha_coeff": np.full(shape, value + 3),
        "alpha_power": np.full(shape, value + 4),
        "air": np.zeros(shape, dtype=int),
    }
    return maps


class DummyGrid2D:
    def __init__(self, nx, ny, dt):
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.is_3d = False


def test_background_only(monkeypatch):
    shape = (5, 5)
    grid = DummyGrid2D(nx=shape[0], ny=shape[1], dt=1e-4)

    # Bypass instance and path checks in Medium.__init__.
    import fullwave.medium as medium_module
    import fullwave.medium_builder.medium_builder as mb_module

    dummy_check = type(
        "dummy",
        (),
        {
            "check_instance": lambda _, instance, cls: None,  # noqa: ARG005
            "check_path_exists": lambda _, path: None,  # noqa: ARG005
            "check_compatible_value": lambda _,
            value,  # noqa: ARG005
            compatible_values,  # noqa: ARG005
            error_message_template: None,  # noqa: ARG005
        },
    )()
    monkeypatch.setattr(mb_module, "check_functions", dummy_check)
    monkeypatch.setattr(medium_module, "check_functions", dummy_check)

    geometry = np.ones(shape, dtype=int)
    maps = create_maps(shape, 1500)
    background = FakeDomain("background", geometry, maps)

    mb = MediumBuilder(grid=grid)
    mb.register_domain(background)
    medium = mb.run()

    # Verify that the medium contains the background maps
    np.testing.assert_array_equal(medium.sound_speed, maps["sound_speed"])
    np.testing.assert_array_equal(medium.density, maps["density"])
    np.testing.assert_array_equal(medium.beta, maps["beta"])
    np.testing.assert_array_equal(medium.alpha_coeff, maps["alpha_coeff"])
    np.testing.assert_array_equal(medium.alpha_power, maps["alpha_power"])
    np.testing.assert_array_equal(medium.air_map, np.zeros(shape, dtype=geometry.dtype))


def test_domain_override(monkeypatch):
    shape = (5, 5)
    grid = DummyGrid2D(nx=shape[0], ny=shape[1], dt=1e-4)

    # Bypass instance and path checks in Medium.__init__.
    import fullwave.medium as medium_module
    import fullwave.medium_builder.medium_builder as mb_module

    dummy_check = type(
        "dummy",
        (),
        {
            "check_instance": lambda _, instance, cls: None,  # noqa: ARG005
            "check_path_exists": lambda _, path: None,  # noqa: ARG005
            "check_compatible_value": lambda _,
            value,  # noqa: ARG005
            compatible_values,  # noqa: ARG005
            error_message_template: None,  # noqa: ARG005
        },
    )()
    monkeypatch.setattr(mb_module, "check_functions", dummy_check)
    monkeypatch.setattr(medium_module, "check_functions", dummy_check)

    # background: uniform maps value=1500
    bg_geometry = np.zeros(shape, dtype=int)
    bg_geometry[:] = 1  # all points active for background
    bg_maps = create_maps(shape, 1500)
    background = FakeDomain("background", bg_geometry, bg_maps)

    # second domain: override only a sub-region
    domain_geometry = np.zeros(shape, dtype=int)
    domain_geometry[2:4, 2:4] = 1  # override region
    domain_maps = create_maps(shape, 1600)
    tissue = FakeDomain("tissue", domain_geometry, domain_maps)

    mb = MediumBuilder(grid=grid)
    mb.register_domain(background)
    mb.register_domain(tissue)
    medium = mb.run()

    # For points in override region, values should come from "tissue"
    override_slice = (slice(2, 4), slice(2, 4))
    np.testing.assert_array_equal(
        medium.sound_speed[override_slice],
        domain_maps["sound_speed"][override_slice],
    )
    np.testing.assert_array_equal(
        medium.density[override_slice],
        domain_maps["density"][override_slice],
    )
    np.testing.assert_array_equal(medium.beta[override_slice], domain_maps["beta"][override_slice])
    np.testing.assert_array_equal(
        medium.alpha_coeff[override_slice],
        domain_maps["alpha_coeff"][override_slice],
    )
    np.testing.assert_array_equal(
        medium.alpha_power[override_slice],
        domain_maps["alpha_power"][override_slice],
    )
    # Other regions remain from background
    not_override = np.ones(shape, dtype=bool)
    not_override[2:4, 2:4] = False
    np.testing.assert_array_equal(
        medium.sound_speed[not_override],
        bg_maps["sound_speed"][not_override],
    )


def test_ignore_non_linearity(monkeypatch):
    shape = (5, 5)
    grid = DummyGrid2D(nx=shape[0], ny=shape[1], dt=1e-4)

    # Bypass instance and path checks in Medium.__init__.
    import fullwave.medium as medium_module
    import fullwave.medium_builder.medium_builder as mb_module

    dummy_check = type(
        "dummy",
        (),
        {
            "check_instance": lambda _, instance, cls: None,  # noqa: ARG005
            "check_path_exists": lambda _, path: None,  # noqa: ARG005
            "check_compatible_value": lambda _,
            value,  # noqa: ARG005
            compatible_values,  # noqa: ARG005
            error_message_template: None,  # noqa: ARG005
        },
    )()
    monkeypatch.setattr(mb_module, "check_functions", dummy_check)
    monkeypatch.setattr(medium_module, "check_functions", dummy_check)

    geometry = np.ones(shape, dtype=int)
    bg_maps = create_maps(shape, 1500)
    background = FakeDomain("background", geometry, bg_maps)

    # Domain with different beta values
    domain_maps = create_maps(shape, 1600)
    tissue = FakeDomain("tissue", geometry, domain_maps)

    mb = MediumBuilder(grid=grid, ignore_non_linearity=True)
    mb.register_domain(background)
    mb.register_domain(tissue)
    medium = mb.run()

    # When non-linearity is ignored, beta_map should be zeros
    np.testing.assert_array_equal(medium.beta, np.zeros(shape, dtype=bg_maps["beta"].dtype))


def test_abdominal_wall_override(
    monkeypatch,
):
    shape = (5, 5)
    grid = DummyGrid2D(nx=shape[0], ny=shape[1], dt=1e-4)

    # Bypass instance and path checks in Medium.__init__.
    import fullwave.medium as medium_module
    import fullwave.medium_builder.medium_builder as mb_module

    dummy_check = type(
        "dummy",
        (),
        {
            "check_instance": lambda _, instance, cls: None,  # noqa: ARG005
            "check_path_exists": lambda _, path: None,  # noqa: ARG005
            "check_compatible_value": lambda _,
            value,  # noqa: ARG005
            compatible_values,  # noqa: ARG005
            error_message_template: None,  # noqa: ARG005
        },
    )()
    monkeypatch.setattr(mb_module, "check_functions", dummy_check)
    monkeypatch.setattr(medium_module, "check_functions", dummy_check)

    # Setup background
    bg_geometry = np.ones(shape, dtype=int)
    bg_maps = create_maps(shape, 1500)
    background = FakeDomain("background", bg_geometry, bg_maps)

    # Domain named "abdominal_wall" with selective override region
    domain_geometry = np.zeros(shape, dtype=int)
    # override only a subset and set values close to background except one region
    domain_geometry[1:3, 1:3] = 1
    domain_maps = create_maps(shape, 1500)  # same as background
    # Change one element to enforce override based on rtol check
    domain_maps["density"][1, 1] = 1700
    abdominal_wall = FakeDomain("abdominal_wall", domain_geometry, domain_maps)

    mb = MediumBuilder(
        grid=grid,
        material_properties=MaterialProperties(),
        background_domain_properties="water",
    )
    # simulate that MaterialProperties.water.density is 1501 (for example)
    # we patch material_properties attribute for testing purposes.
    mb.material_properties.water = {
        "sound_speed": 1500,
        "density": 1501,
        "alpha_coeff": 1502,
        "alpha_power": 1503,
        "beta": 1504,
    }

    mb.register_domain(background)
    mb.register_domain(abdominal_wall)
    medium = mb.run()

    # For the override region, all points except
    # where density nearly equals background remain unchanged,
    # but at (1,1) the density should be taken from abdominal_wall.
    # Because (1,1) in abdominal_wall has a distinct value, it should override.
    expected_density = bg_maps["density"].copy()
    # Determine non-zero index for "abdominal_wall" for density map under rtol condition
    # Here we simply expect the (1,1) point gets updated.
    expected_density[1, 1] = domain_maps["density"][1, 1]
    np.testing.assert_array_equal(medium.density, expected_density)
