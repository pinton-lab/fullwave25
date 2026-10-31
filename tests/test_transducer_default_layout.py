"""The default layout of a flat array and the helpers that go with it."""

import numpy as np
import pytest

import fullwave
from fullwave.transducer import (
    MARGIN_WATER,
    PROBE_FILL,
    LinearTransducer,
    default_flat_element_layer_px,
    make_p4_1c_trasnducer,
)


def _grid(
    *,
    ppw: int = 6,
    f0: float = 5.208e6,
    domain_size: tuple[float, float] = (0.01, 0.0382),
) -> fullwave.Grid:
    return fullwave.Grid(
        domain_size=domain_size,
        f0=f0,
        duration=5.0e-6,
        c0=1540.0,
        ppw=ppw,
        cfl=0.4,
    )


def _rows(array: np.ndarray) -> tuple[int, int]:
    return int(array[:, 0].min()), int(array[:, 0].max())


@pytest.mark.parametrize(("ppw", "expected"), [(6, 8), (8, 8), (12, 12), (20, 20)])
def test_the_default_layer_is_one_wavelength_and_at_least_eight_rows(ppw, expected):
    assert default_flat_element_layer_px(_grid(ppw=ppw)) == expected


@pytest.mark.parametrize("ppw", [6, 12])
@pytest.mark.parametrize("name", ["l7_4", "p4_1c"])
def test_a_preset_starts_at_row_zero_with_the_receiver_row_in_front(name, ppw):
    grid = _grid(ppw=ppw)
    layer = default_flat_element_layer_px(grid)
    transducer = getattr(fullwave.Transducer, name)(grid)
    assert _rows(transducer.source_coords) == (0, layer - 1)
    assert _rows(transducer.sensor_coords) == (layer, layer)
    assert transducer.face_row_px == layer + 1


def test_a_preset_keeps_a_layer_it_is_given():
    transducer = fullwave.Transducer.l7_4(_grid(), element_layer_px=3)
    assert _rows(transducer.source_coords) == (0, 2)
    assert transducer.face_row_px == 4


def test_the_p4_1c_helper_takes_the_default_layer_and_row_zero():
    grid = _grid()
    transducer = make_p4_1c_trasnducer(grid)
    assert _rows(transducer.source_coords) == (0, 7)
    assert _rows(transducer.sensor_coords) == (8, 8)


def test_the_linear_transducer_builds_with_the_default_layer_and_row_zero():
    grid = _grid()
    transducer = LinearTransducer(grid)
    assert _rows(transducer.source_coords) == (0, 7)
    assert _rows(transducer.sensor_coords) == (8, 8)
    assert int(transducer.source_coords[:, 1].min()) == 0


def test_a_flat_geometry_without_position_or_layer_takes_the_defaults():
    grid = _grid()
    geometry = fullwave.TransducerGeometry(
        grid,
        number_elements=16,
        element_width_m=0.25e-3,
        element_spacing_m=0.05e-3,
    )
    assert tuple(int(v) for v in geometry.position_px) == (0, 0)
    assert int(geometry.element_layer_px) == 8
    assert _rows(geometry._source_coords) == (0, 7)


def test_a_curved_geometry_keeps_its_position_and_demands_a_layer():
    grid = _grid()
    with pytest.raises(ValueError, match="element_layer"):
        fullwave.TransducerGeometry(
            grid,
            number_elements=16,
            element_width_m=0.0,
            element_spacing_m=0.5e-3,
            radius=0.05,
        )


def test_the_c5_2v_keeps_its_own_layer():
    grid = _grid(f0=3.7e6, domain_size=(0.03, 0.05))
    transducer = fullwave.Transducer.c5_2v(grid)
    assert int(transducer.transducer_geometry.element_layer_px) == round(grid.ppw * 3)


def test_the_air_goes_directly_behind_the_first_source_row():
    grid = _grid()
    transducer = fullwave.Transducer.l7_4(grid, element_layer_px=3, face_depth_m=5 * grid.dx)
    air = transducer.make_surface_reflective_with_air()
    source = transducer.source_mask
    first_row = int(transducer.source_coords[:, 0].min())
    assert first_row == 5
    assert not (air & source).any()
    columns = np.unique(transducer.source_coords[:, 1])
    assert air[2:5, columns].all()
    assert not air[:2].any()
    assert not air[5:].any()
    assert int(air.sum()) == 3 * len(columns)


def test_the_air_is_clipped_at_the_top_of_the_grid():
    grid = _grid()
    transducer = fullwave.Transducer.l7_4(grid, element_layer_px=3, face_depth_m=2 * grid.dx)
    air = transducer.make_surface_reflective_with_air()
    columns = np.unique(transducer.source_coords[:, 1])
    assert air[:2, columns].all()
    assert int(air.sum()) == 2 * len(columns)


def test_the_air_of_a_convex_array_lies_behind_its_source_cells():
    grid = _grid(f0=3.7e6, domain_size=(0.03, 0.05))
    transducer = fullwave.Transducer.c5_2v(grid)
    air = transducer.make_surface_reflective_with_air()
    assert air.any()
    assert not (air & transducer.source_mask).any()


def test_the_old_misspelled_name_is_gone():
    assert not hasattr(fullwave.Transducer, "make_suraface_reflective_with_air")


def _maps(grid: fullwave.Grid) -> dict[str, np.ndarray]:
    shape = (grid.nx, grid.ny)
    return {
        "sound_speed": np.full(shape, 1500.0),
        "density": np.full(shape, 1100.0),
        "alpha_coeff": np.full(shape, 0.5),
        "alpha_power": np.ones(shape),
    }


def _wide_transducer():
    grid = _grid(domain_size=(0.02, 0.045))
    return grid, fullwave.Transducer.l7_4(grid)


def test_the_kerf_and_the_margin_fill_the_rows_above_the_face():
    grid, transducer = _wide_transducer()
    maps = _maps(grid)
    filled = transducer.fill_probe_above_face(**maps)
    face = transducer.face_row_px
    source = transducer.source_mask
    columns = np.flatnonzero(source.any(axis=0))
    first, last = columns[0], columns[-1]
    kerf = np.zeros(grid.shape, dtype=bool)
    kerf[:face, first : last + 1] = True
    margin = np.zeros(grid.shape, dtype=bool)
    margin[:face] = True
    margin[:, first : last + 1] = False
    assert kerf.any()
    assert margin[:, :first].any()
    assert margin[:, last + 1 :].any()
    for name in PROBE_FILL:
        assert np.unique(filled[name][kerf]).tolist() == [PROBE_FILL[name]]
        assert np.unique(filled[name][margin]).tolist() == [MARGIN_WATER[name]]
        assert np.unique(filled[name][source]).tolist() == [PROBE_FILL[name]]
        np.testing.assert_array_equal(filled[name][face:], maps[name][face:])
    assert np.unique(filled["alpha_power"][kerf]).tolist() == [1.0]
    assert np.unique(filled["alpha_power"][margin]).tolist() == [1.999]


def test_the_margin_starts_beside_the_first_and_the_last_element_edge():
    grid, transducer = _wide_transducer()
    filled = transducer.fill_probe_above_face(**_maps(grid))
    columns = np.flatnonzero(transducer.source_mask.any(axis=0))
    row = transducer.face_row_px - 1
    power = filled["alpha_power"][row]
    assert power[columns[0] - 1] == MARGIN_WATER["alpha_power"]
    assert power[columns[0]] == PROBE_FILL["alpha_power"]
    assert power[columns[-1]] == PROBE_FILL["alpha_power"]
    assert power[columns[-1] + 1] == MARGIN_WATER["alpha_power"]


def test_the_fill_values_are_the_calibrated_ones():
    assert PROBE_FILL == {
        "sound_speed": 1540.0,
        "density": 1000.0,
        "alpha_coeff": 0.005,
        "alpha_power": 1.0,
    }
    assert MARGIN_WATER == {**PROBE_FILL, "alpha_power": 1.999}


def test_the_probe_fill_leaves_the_input_maps_alone():
    grid = _grid()
    transducer = fullwave.Transducer.l7_4(grid)
    maps = _maps(grid)
    transducer.fill_probe_above_face(**maps)
    assert np.unique(maps["sound_speed"]).tolist() == [1500.0]


def test_the_probe_fill_domain_holds_the_kerf_and_the_margin():
    grid, transducer = _wide_transducer()
    domain = transducer.probe_fill_domain()
    face = transducer.face_row_px
    source = transducer.source_mask
    geometry = domain.base_geometry.astype(bool)
    assert not geometry[face:].any()
    assert geometry[:face].all()
    assert np.unique(domain.alpha_power[source]).tolist() == [PROBE_FILL["alpha_power"]]
    columns = np.flatnonzero(source.any(axis=0))
    row = face - 1
    assert domain.alpha_power[row, columns[0] - 1] == MARGIN_WATER["alpha_power"]
    assert domain.alpha_power[row, columns[0]] == PROBE_FILL["alpha_power"]
    assert domain.sound_speed[row, columns[0] - 1] == PROBE_FILL["sound_speed"]


def test_the_face_depth_is_the_face_row_times_the_grid_step():
    grid = _grid()
    transducer = fullwave.Transducer.l7_4(grid)
    assert transducer.face_depth_m == transducer.face_row_px * grid.dx


@pytest.mark.parametrize("name", ["l7_4", "p4_1c"])
def test_a_wider_grid_centers_the_array_and_fills_the_columns_beside_it(name):
    grid = _grid(domain_size=(0.02, 0.045))
    transducer = getattr(fullwave.Transducer, name)(grid)
    columns = np.flatnonzero(transducer.source_mask.any(axis=0))
    left, right = columns[0], grid.ny - 1 - columns[-1]
    assert left > 0
    assert abs(left - right) <= 1
    filled = transducer.fill_probe_above_face(**_maps(grid))
    assert np.unique(filled["sound_speed"][: transducer.face_row_px]).tolist() == [1540.0]


def test_the_probe_fill_refuses_a_curved_array():
    grid = _grid(f0=3.7e6, domain_size=(0.03, 0.05))
    transducer = fullwave.Transducer.c5_2v(grid)
    with pytest.raises(ValueError, match="curved"):
        transducer.fill_probe_above_face(**_maps(grid))
