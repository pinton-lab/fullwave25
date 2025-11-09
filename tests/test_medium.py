import numpy as np
import pytest

import fullwave.medium as medium_module
from fullwave.medium import Medium, MediumRelaxationMaps
from fullwave.solver.utils import initialize_relaxation_param_dict


class DummyGrid2D:
    def __init__(self, nx, ny, dt):
        self.nx = nx
        self.ny = ny
        self.dt = dt
        self.is_3d = False


def test_post_init_conversion(monkeypatch):
    # Provide 1D arrays as inputs

    grid_shape = (1, 1)
    grid = DummyGrid2D(nx=grid_shape[0], ny=grid_shape[1], dt=1e-4)

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
    monkeypatch.setattr(medium_module, "check_functions", dummy_check)

    sound_speed = np.array([1500])
    density = np.array([1000])
    alpha_coeff = np.array([0.5])
    alpha_power = np.array([1.2])
    beta = np.array([0.8])
    medium = Medium(grid, sound_speed, density, alpha_coeff, alpha_power, beta)
    # Check that all fields are converted to at least 2D arrays.
    assert medium.sound_speed.ndim >= 2
    assert medium.density.ndim >= 2
    assert medium.alpha_coeff.ndim >= 2
    assert medium.alpha_power.ndim >= 2
    assert medium.beta.ndim >= 2


def test_check_fields_valid(monkeypatch):
    # Create arrays with consistent shape.
    grid_shape = (2, 2)

    grid = DummyGrid2D(nx=grid_shape[0], ny=grid_shape[1], dt=1e-4)

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
    monkeypatch.setattr(medium_module, "check_functions", dummy_check)

    sound_speed = np.ones(grid_shape) * 1500
    density = np.ones(grid_shape) * 1000
    alpha_coeff = np.ones(grid_shape) * 0.5
    alpha_power = np.ones(grid_shape) * 1.2
    beta = np.ones(grid_shape) * 0.8
    medium = Medium(grid, sound_speed, density, alpha_coeff, alpha_power, beta)
    # Should pass without errors.
    medium.check_fields()


def test_check_fields_invalid(monkeypatch):
    grid_shape = (2, 2)
    grid = DummyGrid2D(nx=grid_shape[0], ny=grid_shape[1], dt=1e-4)

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
    monkeypatch.setattr(medium_module, "check_functions", dummy_check)
    # Create one field with an inconsistent shape.
    sound_speed = np.ones(grid_shape) * 1500
    density = np.ones(grid_shape) * 1000
    alpha_coeff = np.ones((2, 3)) * 0.5  # Incorrect shape
    alpha_power = np.ones(grid_shape) * 1.2
    beta = np.ones(grid_shape) * 0.8
    with pytest.raises(AssertionError):
        _ = Medium(grid, sound_speed, density, alpha_coeff, alpha_power, beta)


def test_plot_exports_file(tmp_path, monkeypatch):
    # Create a medium with 2D arrays.
    grid_shape = (2, 2)
    grid = DummyGrid2D(nx=grid_shape[0], ny=grid_shape[1], dt=1e-4)

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
    monkeypatch.setattr(medium_module, "check_functions", dummy_check)

    sound_speed = np.ones(grid_shape) * 1500.0
    density = np.ones(grid_shape) * 1000.0
    alpha_coeff = np.ones(grid_shape) * 0.5
    alpha_power = np.ones(grid_shape) * 1.2
    beta = np.ones(grid_shape) * 0.8

    medium = Medium(grid, sound_speed, density, alpha_coeff, alpha_power, beta)
    export_file = tmp_path / "test_plot.png"

    # Call plot with file export.
    medium.plot(export_path=str(export_file), show=False)

    assert export_file.exists()


def test_plot_show(monkeypatch, tmp_path):
    # Create a medium with 2D arrays.

    grid_shape = (2, 2)
    grid = DummyGrid2D(nx=grid_shape[0], ny=grid_shape[1], dt=1e-4)

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
    monkeypatch.setattr(medium_module, "check_functions", dummy_check)

    show_called = False

    def fake_show():
        nonlocal show_called
        show_called = True

    monkeypatch.setattr("matplotlib.pyplot.show", fake_show)

    sound_speed = np.ones(grid_shape) * 1500.0
    density = np.ones(grid_shape) * 1000.0
    alpha_coeff = np.ones(grid_shape) * 0.5
    alpha_power = np.ones(grid_shape) * 1.2
    beta = np.ones(grid_shape) * 0.8

    medium = Medium(grid, sound_speed, density, alpha_coeff, alpha_power, beta)
    export_file = tmp_path / "test_plot.png"

    # Call plot with show flag enabled.
    medium.plot(export_path=str(export_file), show=True)

    assert show_called
    assert export_file.exists()


def test_bulk_modulus(monkeypatch):
    grid_shape = (2, 2)
    grid = DummyGrid2D(nx=grid_shape[0], ny=grid_shape[1], dt=1e-4)

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
    monkeypatch.setattr(medium_module, "check_functions", dummy_check)

    sound_speed = np.ones(grid_shape) * 1500.0
    density = np.ones(grid_shape) * 1000.0
    alpha_coeff = np.ones(grid_shape) * 0.5
    alpha_power = np.ones(grid_shape) * 1.2
    beta = np.ones(grid_shape) * 0.8
    medium = Medium(grid, sound_speed, density, alpha_coeff, alpha_power, beta)
    bulk = medium.bulk_modulus
    expected = (sound_speed**2) * density
    assert bulk.shape == grid_shape
    np.testing.assert_allclose(bulk, expected)


def test_n_air_with_provided_air_map(monkeypatch):
    grid_shape = (2, 2)
    grid = DummyGrid2D(nx=grid_shape[0], ny=grid_shape[1], dt=1e-4)

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
    monkeypatch.setattr(medium_module, "check_functions", dummy_check)

    # Define an air map with two "air" positions (nonzero values)
    air_map_input = np.array([[1, 0], [0, 1]], dtype=np.int64)
    sound_speed = np.ones(grid_shape) * 1500
    density = np.ones(grid_shape) * 1000
    alpha_coeff = np.ones(grid_shape) * 0.5
    alpha_power = np.ones(grid_shape) * 1.2
    beta = np.ones(grid_shape) * 0.8
    medium = Medium(
        grid,
        sound_speed,
        density,
        alpha_coeff,
        alpha_power,
        beta,
        air_map=air_map_input,
    )
    # n_air should equal the number of nonzero elements in the provided air_map.
    expected_n_air = np.count_nonzero(air_map_input)
    assert medium.n_air == expected_n_air


# Tests for MediumRelaxationMaps


def get_dummy_relaxation_dict(shape, n_relaxation_mechanisms=2):
    # Use the same keys as returned by initialize_relaxation_param_dict
    base = initialize_relaxation_param_dict(n_relaxation_mechanisms=n_relaxation_mechanisms)
    # Replace default zeros with ones for testing purposes.
    return {key: np.ones(shape) for key, _ in base.items()}


def test_medium_relaxation_post_init_conversion(monkeypatch):
    grid_shape = (1, 1)
    grid = DummyGrid2D(nx=grid_shape[0], ny=grid_shape[1], dt=1e-4)

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
    monkeypatch.setattr(medium_module, "check_functions", dummy_check)

    sound_speed = np.array([1500])
    density = np.array([1000])
    beta = np.array([0.8])
    relaxation_dict = get_dummy_relaxation_dict(grid_shape, n_relaxation_mechanisms=2)
    medium_relax = MediumRelaxationMaps(grid, sound_speed, density, beta, relaxation_dict)
    # Check that arrays are at least 2D and have the correct shape.
    assert medium_relax.sound_speed.ndim >= 2
    assert medium_relax.density.ndim >= 2
    assert medium_relax.beta.ndim >= 2
    for val in medium_relax.relaxation_param_dict.values():
        assert val.ndim >= 2
        assert val.shape == grid_shape


def test_medium_relaxation_check_fields_valid(monkeypatch):
    grid_shape = (2, 2)

    grid = DummyGrid2D(nx=grid_shape[0], ny=grid_shape[1], dt=1e-4)

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
    monkeypatch.setattr(medium_module, "check_functions", dummy_check)

    sound_speed = np.ones(grid_shape) * 1500
    density = np.ones(grid_shape) * 1000
    beta = np.ones(grid_shape) * 0.8
    relaxation_dict = get_dummy_relaxation_dict(grid_shape, n_relaxation_mechanisms=2)
    medium_relax = MediumRelaxationMaps(grid, sound_speed, density, beta, relaxation_dict)
    # Should pass without assertion errors.
    medium_relax.check_fields()


def test_medium_relaxation_check_fields_invalid():
    grid_shape = (2, 2)
    grid = DummyGrid2D(nx=grid_shape[0], ny=grid_shape[1], dt=1e-4)
    sound_speed = np.ones(grid_shape) * 1500
    density = np.ones(grid_shape) * 1000
    beta = np.ones(grid_shape) * 0.8
    relaxation_dict = get_dummy_relaxation_dict(grid_shape, n_relaxation_mechanisms=2)
    # Introduce invalid shape in one relaxation parameter
    relaxation_dict_key = next(iter(relaxation_dict.keys()))
    relaxation_dict[relaxation_dict_key] = np.ones((2, 3))
    with pytest.raises(ValueError, match=".*Relaxation parameter map shape error.*"):
        _ = MediumRelaxationMaps(grid, sound_speed, density, beta, relaxation_dict)


def test_medium_relaxation_plot_exports_file(tmp_path):
    grid_shape = (2, 2)
    grid = DummyGrid2D(nx=grid_shape[0], ny=grid_shape[1], dt=1e-4)
    sound_speed = np.ones(grid_shape) * 1500.0
    density = np.ones(grid_shape) * 1000.0
    beta = np.ones(grid_shape) * 0.8
    relaxation_dict = get_dummy_relaxation_dict(grid_shape, n_relaxation_mechanisms=2)
    medium_relax = MediumRelaxationMaps(grid, sound_speed, density, beta, relaxation_dict)
    export_file = tmp_path / "test_relax_plot.png"
    medium_relax.plot(export_path=str(export_file), show=False)
    assert export_file.exists()


def test_medium_relaxation_plot_show(monkeypatch, tmp_path):
    show_called = False

    def fake_show():
        nonlocal show_called
        show_called = True

    monkeypatch.setattr("matplotlib.pyplot.show", fake_show)
    grid_shape = (2, 2)
    grid = DummyGrid2D(nx=grid_shape[0], ny=grid_shape[1], dt=1e-4)

    sound_speed = np.ones(grid_shape) * 1500.0
    density = np.ones(grid_shape) * 1000.0
    beta = np.ones(grid_shape) * 0.8
    relaxation_dict = get_dummy_relaxation_dict(grid_shape, n_relaxation_mechanisms=2)
    medium_relax = MediumRelaxationMaps(grid, sound_speed, density, beta, relaxation_dict)
    export_file = tmp_path / "test_relax_plot.png"
    medium_relax.plot(export_path=str(export_file), show=True)
    assert show_called
    assert export_file.exists()


def test_build_creates_medium_relaxation_maps(monkeypatch, tmp_path):
    # Create a dummy grid with required attributes.
    grid_shape = (2, 2)
    grid = DummyGrid2D(nx=grid_shape[0], ny=grid_shape[1], dt=1e-4)

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
    monkeypatch.setattr(medium_module, "check_functions", dummy_check)

    # Create test arrays.
    sound_speed = np.ones(grid_shape) * 1500.0
    density = np.ones(grid_shape) * 1000.0
    alpha_coeff = np.ones(grid_shape) * 0.5
    alpha_power = np.ones(grid_shape) * 1.2
    beta = np.ones(grid_shape) * 0.8
    air_map = np.array([[0, 1], [1, 0]], dtype=np.int64)

    # Create a dummy relaxation parameters database file path.
    dummy_db_path = tmp_path / "dummy_relaxation.mat"
    dummy_db_path.write_text("dummy")

    # Monkeypatch generate_relaxation_params to return a dummy relaxation parameters dict.
    def dummy_generate_relaxation_params(
        n_relaxation_mechanisms,
        alpha_coeff,
        alpha_power,
        path_database,
    ):
        keys = list(initialize_relaxation_param_dict(n_relaxation_mechanisms).keys())
        return {key: np.ones(grid_shape) for key in keys}

    monkeypatch.setattr(
        medium_module,
        "generate_relaxation_params",
        dummy_generate_relaxation_params,
    )

    # Create a Medium instance and call build.
    medium_instance = Medium(
        grid=grid,
        sound_speed=sound_speed,
        density=density,
        alpha_coeff=alpha_coeff,
        alpha_power=alpha_power,
        beta=beta,
        air_map=air_map,
        path_relaxation_parameters_database=dummy_db_path,
        n_relaxation_mechanisms=2,
    )
    medium_relax = medium_instance.build()

    # Check that the returned object is an instance of MediumRelaxationMaps.
    assert isinstance(medium_relax, MediumRelaxationMaps)

    # Validate field shapes.
    assert medium_relax.sound_speed.shape == grid_shape
    assert medium_relax.density.shape == grid_shape
    assert medium_relax.beta.shape == grid_shape

    # Validate relaxation parameter maps.
    relax_keys = initialize_relaxation_param_dict(medium_relax.n_relaxation_mechanisms).keys()
    for key in relax_keys:
        np.testing.assert_allclose(medium_relax.relaxation_param_dict[key], np.ones(grid_shape))


def test_medium_relaxation_maps_build_returns_self(monkeypatch):
    """Test that MediumRelaxationMaps.build() returns self."""
    grid_shape = (2, 2)
    grid = DummyGrid2D(nx=grid_shape[0], ny=grid_shape[1], dt=1e-4)

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
    monkeypatch.setattr(medium_module, "check_functions", dummy_check)

    sound_speed = np.ones(grid_shape) * 1500
    density = np.ones(grid_shape) * 1000
    beta = np.ones(grid_shape) * 0.8
    relaxation_dict = get_dummy_relaxation_dict(grid_shape, n_relaxation_mechanisms=2)
    medium_relax = MediumRelaxationMaps(grid, sound_speed, density, beta, relaxation_dict)

    # Call build and verify it returns the same instance
    result = medium_relax.build()

    assert result is medium_relax
    assert isinstance(result, MediumRelaxationMaps)

    for key in relaxation_dict:
        np.testing.assert_allclose(result.relaxation_param_dict[key], relaxation_dict[key])
