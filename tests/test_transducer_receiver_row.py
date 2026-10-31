"""The receivers of a flat array lie one row in front of the source cells."""

import numpy as np
import pytest

from fullwave.transducer import TransducerGeometry


class DummyGrid2D:
    is_3d = False

    def __init__(self, *, nx: int, ny: int, dx: float, dy: float) -> None:
        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy


class DummyGrid3D:
    is_3d = True

    def __init__(self, *, nx: int, ny: int, nz: int, d: float) -> None:
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = d
        self.dy = d
        self.dz = d


def _as_set(coords):
    return {tuple(row) for row in coords.tolist()}


@pytest.mark.parametrize("average_surface_signals", [True, False])
@pytest.mark.parametrize("layer", [1, 5])
def test_flat_2d_sensor_row_in_front(average_surface_signals, layer):
    grid = DummyGrid2D(nx=100, ny=100, dx=1e-3, dy=1e-3)
    geom = TransducerGeometry(
        grid,
        number_elements=8,
        element_width_px=3,
        element_spacing_px=1,
        element_layer_px=layer,
        position_px=(20, 5),
        validate_input=False,
        average_surface_signals=average_surface_signals,
    )
    assert not (_as_set(geom._sensor_coords) & _as_set(geom._source_coords))
    assert np.all(geom._sensor_coords[:, 0] == geom.position_px[0] + layer)


@pytest.mark.parametrize("layer", [1, 4])
def test_flat_3d_sensor_row_in_front(layer):
    grid = DummyGrid3D(nx=60, ny=60, nz=60, d=1e-3)
    geom = TransducerGeometry(
        grid,
        number_elements=4,
        element_width_px=3,
        element_height_px=3,
        element_spacing_px=1,
        element_layer_px=layer,
        position_px=(10, 5, 5),
        validate_input=False,
    )
    assert not (_as_set(geom._sensor_coords) & _as_set(geom._source_coords))
    assert np.all(geom._sensor_coords[:, 0] == geom.position_px[0] + layer)


def test_flat_sensor_row_outside_grid_raises():
    grid = DummyGrid2D(nx=26, ny=100, dx=1e-3, dy=1e-3)
    with pytest.raises(ValueError, match="sensor row"):
        TransducerGeometry(
            grid,
            number_elements=4,
            element_width_px=3,
            element_spacing_px=1,
            element_layer_px=5,
            position_px=(21, 5),
            validate_input=False,
        )
