"""The kerf of the L7-4 preset: a gap of exactly ``kerf_px`` pixels, held by default."""

import numpy as np
import pytest

import fullwave

PITCH_M = 0.298e-3
PHYSICAL_KERF_M = 0.048e-3
N_ELEMENTS = 128


def l74_grid(*, ppw):
    f0, c0 = 5.208e6, 1540.0
    dx = c0 / f0 / ppw
    return fullwave.Grid(
        domain_size=(20 * dx, 40e-3),
        f0=f0,
        duration=2e-6,
        c0=c0,
        ppw=ppw,
        cfl=0.3,
    )


def first_row_of(coords, ids):
    """Return the lateral pixels and element ids of the first source row."""
    in_first_row = coords[:, 0] == coords[0, 0]
    return coords[in_first_row, 1], ids[in_first_row]


def kerf_gap_widths(kerf_coords):
    """Return the widths of the runs of kerf pixels in the first row that has kerf."""
    row = kerf_coords[kerf_coords[:, 0] == kerf_coords[0, 0], 1]
    breaks = np.flatnonzero(np.diff(row) > 1)
    return np.diff(np.concatenate([[0], breaks + 1, [len(row)]])).tolist()


@pytest.mark.parametrize("ppw", [6, 12])
def test_the_default_l74_holds_a_kerf_of_exactly_one_pixel(ppw):
    transducer = fullwave.Transducer.l7_4(l74_grid(ppw=ppw))
    assert transducer.hold_kerf
    widths = kerf_gap_widths(transducer.kerf_coords)
    assert len(widths) == N_ELEMENTS - 1
    assert set(widths) == {1}


@pytest.mark.parametrize("ppw", [6, 12])
def test_l74_element_widths_differ_by_at_most_one_pixel(ppw):
    geometry = fullwave.Transducer.l7_4(l74_grid(ppw=ppw)).transducer_geometry
    _, element_ids = first_row_of(geometry._source_coords, geometry._source_ids)
    _, widths = np.unique(element_ids, return_counts=True)
    assert len(widths) == N_ELEMENTS
    assert widths.max() - widths.min() <= 1


def test_l74_with_kerf_px_zero_has_touching_elements_and_no_kerf():
    transducer = fullwave.Transducer.l7_4(l74_grid(ppw=6), kerf_px=0)
    lateral, _ = first_row_of(
        transducer.transducer_geometry._source_coords, transducer.transducer_geometry._source_ids
    )
    assert transducer.kerf_coords.shape == (0, 2)
    assert np.diff(lateral).tolist() == [1] * (len(lateral) - 1)


def test_l74_with_the_physical_kerf_has_gaps_of_different_widths_at_6_points_per_wavelength():
    transducer = fullwave.Transducer.l7_4(l74_grid(ppw=6), kerf_px=None, hold_kerf=False)
    assert set(kerf_gap_widths(transducer.kerf_coords)) == {1, 2}


def test_l74_without_kerf_px_and_hold_has_the_source_of_the_geometry_in_meters():
    grid = l74_grid(ppw=6)
    legacy = fullwave.Transducer.l7_4(grid, kerf_px=None, hold_kerf=False)
    geometry = legacy.transducer_geometry
    in_meters = fullwave.transducer.TransducerGeometry(
        grid,
        number_elements=N_ELEMENTS,
        element_width_m=PITCH_M - PHYSICAL_KERF_M,
        element_spacing_m=PHYSICAL_KERF_M,
        element_layer_px=geometry.element_layer_px,
        position_px=tuple(geometry.position_px),
    )
    assert geometry.kerf_px is None
    assert not legacy.hold_kerf
    assert np.array_equal(geometry._source_coords, in_meters._source_coords)
