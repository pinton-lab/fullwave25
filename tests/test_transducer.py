import numpy as np
import pytest

import fullwave
from fullwave.transducer import Transducer, TransducerGeometry


# Minimal dummy grid class for testing
class DummyGrid2D:
    def __init__(self, nx, ny, dx, dy):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.is_3d = False


def test_valid_transducer():
    grid = DummyGrid2D(nx=400, ny=400, dx=1e-3, dy=1e-3)
    number_elements = 10
    element_width_m = 1e-3
    element_spacing_m = 1e-3
    element_layer_px = 1

    position_px = (1, 1)
    geom = TransducerGeometry(
        grid,
        number_elements=number_elements,
        element_width_px=1,
        element_spacing_px=1,
        element_layer_px=element_layer_px,
        position_px=position_px,
        validate_input=False,
    )

    # Check computed properties
    assert geom.element_pitch_m == element_width_m + element_spacing_m
    expected_width = number_elements * element_width_m + (number_elements - 1) * element_spacing_m
    np.testing.assert_allclose(geom.transducer_width_m, expected_width)

    # Check that the element mask is a boolean array of the correct shape.
    mask = geom.element_mask_input
    assert mask.shape == (grid.nx, grid.ny)

    # For each element, verify that the corresponding region is marked True.
    # In _create_element_mask, a block of 4 grid points in x is set.
    for i in range(number_elements):
        element_pos_y = i * int(geom.element_pitch_px)
        x_start = geom.position_px[0]
        x_end = x_start + element_layer_px
        y_start = element_pos_y + geom.position_px[1]
        y_end = y_start + int(geom.element_width_px)
        assert np.all(mask[x_start:x_end, y_start:y_end])


def test_invalid_transducer_y():
    # Test error when the transducer extends beyond grid in y-direction.
    grid = DummyGrid2D(nx=200, ny=50, dx=1e-3, dy=1e-3)
    number_elements = 10
    element_width_m = 1e-3
    element_spacing_m = 1e-3
    # For y: round(40e-3/1e-3)=40, and required width = 10+9 = 19 pixels => 40+19=59 > 50.
    position_m = (1e-3, 40e-3, 0)
    with pytest.raises(ValueError, match="positioned outside the grid in the y-direction"):
        TransducerGeometry(
            grid,
            number_elements=number_elements,
            element_width_m=element_width_m,
            element_spacing_m=element_spacing_m,
            position_m=position_m,
            element_layer_px=1,
            validate_input=False,
        )


def test_invalid_transducer_x():
    # Test error when the transducer is positioned outside the grid in the x-direction.
    grid = DummyGrid2D(nx=100, ny=200, dx=1e-3, dy=1e-3)
    # x-position: round(101e-3/1e-3)=101, which is > grid.nx (100)
    position_m = (101e-3, 10e-3, 0)
    with pytest.raises(ValueError, match="positioned outside the grid in the x-direction"):
        TransducerGeometry(
            grid,
            number_elements=10,
            element_width_m=1e-3,
            element_spacing_m=1e-3,
            position_m=position_m,
            element_layer_px=1,
            validate_input=False,
        )


def test_negative_position():
    grid = DummyGrid2D(nx=200, ny=200, dx=1e-3, dy=1e-3)
    position_m = (-10e-3, 10e-3)
    trans = TransducerGeometry(
        grid,
        number_elements=10,
        element_width_m=1e-3,
        element_spacing_m=1e-3,
        position_m=position_m,
        element_layer_px=1,
        validate_input=False,
    )
    np.testing.assert_array_equal(trans.position_px, [0, 10])


# Dummy grid class with nt attribute for GeneralTransducer tests
class DummyGridWithTime:
    def __init__(self, nx, ny, dx, dy, nt):
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.nt = nt
        self.is_3d = False


# Dummy classes for fullwave.Sensor and fullwave.Source to use in tests.
class DummySensor:
    def __init__(self, mask):
        self.mask = mask


class DummySource:
    def __init__(self, signal, mask):
        self.signal = signal
        self.mask = mask


# Override fullwave.Sensor and fullwave.Source for testing.
fullwave.Sensor = DummySensor
fullwave.Source = DummySource


def test_valid_general_transducer():
    # Use a grid where the number of grid points in x equals the number of elements,
    # so that boolean indexing in sensor_mask/source_mask works.
    number_elements = 10
    grid = DummyGridWithTime(
        nx=number_elements,
        ny=50,
        dx=1e-3,
        dy=1e-3,
        nt=50,
    )
    # Position chosen so that position_px = [1, 1, ...]
    position_m = (1e-3, 1e-3, 0)
    geom = TransducerGeometry(
        grid,
        number_elements=number_elements,
        element_width_m=1e-3,
        element_spacing_m=1e-3,
        element_layer_px=1,
        position_m=position_m,
        validate_input=False,
    )
    input_signal = np.ones((number_elements, grid.nt))
    gt = Transducer(
        transducer_geometry=geom,
        grid=grid,
        input_signal=input_signal,
        validate_input=False,
    )
    # Check that the active elements default to all True.
    np.testing.assert_array_equal(
        gt.active_source_elements,
        np.ones(number_elements, dtype=bool),
    )
    # Check that the sensor and source properties return instances of our dummy classes.
    sensor = gt.sensor
    source = gt.source
    assert isinstance(sensor, fullwave.sensor.Sensor)
    assert isinstance(source, fullwave.source.Source)
    # Verify that the sensor and source store the correct mask and signal.
    active_ids = np.where(gt.active_source_elements)[0] + 1
    expected_mask_input = np.isin(gt.transducer_geometry.indexed_element_mask_input, active_ids)
    expected_mask_output = np.isin(gt.transducer_geometry.indexed_element_mask_output, active_ids)

    np.testing.assert_array_equal(sensor.mask, expected_mask_output)
    np.testing.assert_array_equal(source.mask, expected_mask_input)


def test_invalid_signal_shape():
    number_elements = 10
    grid = DummyGridWithTime(
        nx=number_elements,
        ny=50,
        dx=1e-3,
        dy=1e-3,
        nt=60,
    )
    geom = TransducerGeometry(
        grid,
        number_elements=number_elements,
        element_width_px=1,
        element_width_m=None,
        element_spacing_px=1,
        element_spacing_m=None,
        element_layer_px=1,
        element_layer_m=None,
        position_px=(1, 1),
        validate_input=False,
    )
    # Create an input_signal with incorrect number of columns.
    wrong_signal = np.ones((number_elements, grid.nt + 5))
    with pytest.raises(ValueError, match="Input signal has the wrong number of time points"):
        Transducer(
            transducer_geometry=geom,
            grid=grid,
            input_signal=wrong_signal,
            validate_input=False,
        )


def test_custom_active_elements():
    number_elements = 10
    grid = DummyGridWithTime(
        nx=number_elements,
        ny=50,
        dx=1e-3,
        dy=1e-3,
        nt=40,
    )
    position_m = (1e-3, 1e-3, 0)
    geom = TransducerGeometry(
        grid,
        number_elements=number_elements,
        element_width_m=1e-3,
        element_spacing_m=1e-3,
        element_layer_px=1,
        position_m=position_m,
        validate_input=False,
    )
    # Define custom active source and sensor element arrays.
    custom_active = np.array([i % 2 == 0 for i in range(number_elements)], dtype=bool)

    input_signal = np.ones((geom.n_sources_per_element * custom_active.sum(), grid.nt))

    gt = Transducer(
        transducer_geometry=geom,
        grid=grid,
        input_signal=input_signal,
        active_source_elements=custom_active,
        active_sensor_elements=custom_active,
        validate_input=False,
    )
    # Check that the custom active arrays are stored correctly.
    np.testing.assert_array_equal(gt.active_source_elements, custom_active)
    np.testing.assert_array_equal(gt.active_sensor_elements, custom_active)
    # Verify sensor_mask and source_mask

    active_ids = np.where(custom_active)[0] + 1
    expected_mask_input = np.isin(gt.transducer_geometry.indexed_element_mask_input, active_ids)
    expected_mask_output = np.isin(gt.transducer_geometry.indexed_element_mask_output, active_ids)

    np.testing.assert_array_equal(gt.sensor_mask, expected_mask_output)
    np.testing.assert_array_equal(gt.source_mask, expected_mask_input)
