"""Tests for the transmit delays and the probe presets."""

import math

import numpy as np
import pytest

import fullwave
from fullwave import transmit


def _grid():
    return fullwave.Grid(
        domain_size=(2.0e-2, 5.0e-2),
        f0=5.208e6,
        duration=2.0e-6,
        c0=1540.0,
        ppw=12,
        cfl=0.4,
    )


def _centers(elements=8, pitch_m=1.0e-3, axial_m=0.0):
    lateral = (np.arange(elements) - (elements - 1) / 2.0) * pitch_m
    return np.stack([np.full(elements, axial_m), lateral], axis=1)


def test_a_straight_plane_wave_delays_nothing():
    delays = transmit.plane_wave_delays(_centers(), 0.0, 1540.0)
    assert delays == pytest.approx(np.zeros(8))


def test_the_plane_wave_delay_ramp_is_the_steering_sine():
    pitch = 1.0e-3
    for angle in (-15.0, -5.0, 5.0, 15.0):
        delays = transmit.plane_wave_delays(_centers(pitch_m=pitch), angle, 1540.0)
        step = np.diff(delays)
        assert step == pytest.approx(np.full(7, pitch * math.sin(math.radians(angle)) / 1540.0))


def test_the_plane_wave_delay_starts_at_zero():
    for angle in (-15.0, 0.0, 15.0):
        assert transmit.plane_wave_delays(_centers(), angle, 1540.0).min() == pytest.approx(0.0)


def test_a_focus_fires_the_farthest_element_first():
    centers = _centers(elements=9, pitch_m=1.0e-3)
    delays = transmit.focused_delays(centers, np.array([2.0e-2, 0.0]), 1540.0)
    assert delays.argmax() == 4
    assert delays[0] == pytest.approx(0.0)
    assert delays[-1] == pytest.approx(0.0)


def test_a_focus_delay_matches_the_path_difference():
    centers = _centers(elements=3, pitch_m=4.0e-3)
    focus = np.array([1.0e-2, 0.0])
    delays = transmit.focused_delays(centers, focus, 1540.0)
    edge = math.hypot(1.0e-2, 4.0e-3)
    assert delays[1] - delays[0] == pytest.approx((edge - 1.0e-2) / 1540.0)


def test_a_diverging_wave_fires_the_nearest_element_first():
    centers = _centers(elements=9, pitch_m=1.0e-3)
    delays = transmit.diverging_delays(centers, np.array([-1.0e-2, 0.0]), 1540.0)
    assert delays.argmin() == 4
    assert delays.min() == pytest.approx(0.0)


def test_no_taper_weights_every_element_the_same():
    assert transmit.tukey_weights(16, 0.0) == pytest.approx(np.ones(16))


def test_a_taper_falls_at_the_aperture_edge():
    weights = transmit.tukey_weights(32, 0.5)
    assert weights[0] < weights[16]
    assert weights[0] == pytest.approx(weights[-1])


def test_the_layer_is_read_from_the_row_inside_each_element():
    coords = np.array([[10, 0], [10, 1], [11, 0], [11, 1], [10, 5], [11, 5]])
    identifiers = np.array([1, 1, 1, 1, 2, 2])
    layers = transmit.layer_of_each_pixel(coords, identifiers)
    assert layers.tolist() == [0, 0, 1, 1, 0, 1]


def test_the_layer_is_measured_inside_the_element_on_an_arc():
    coords = np.array([[10, 0], [11, 0], [14, 9], [15, 9]])
    identifiers = np.array([1, 1, 2, 2])
    layers = transmit.layer_of_each_pixel(coords, identifiers)
    assert layers.tolist() == [0, 1, 0, 1]


def test_a_silent_element_carries_no_signal():
    grid = _grid()
    coords = np.array([[10, 0], [10, 1]])
    identifiers = np.array([1, 2])
    signal = transmit.signal_of(
        grid,
        coords,
        identifiers,
        transmit.Pulse(),
        np.zeros(2),
        np.array([1.0, 0.0]),
    )
    assert np.abs(signal[0]).max() > 0
    assert np.abs(signal[1]).max() == 0


def test_the_signal_carries_the_element_weight():
    grid = _grid()
    coords = np.array([[10, 0], [10, 1]])
    identifiers = np.array([1, 2])
    signal = transmit.signal_of(
        grid,
        coords,
        identifiers,
        transmit.Pulse(pressure=1.0e5),
        np.zeros(2),
        np.array([1.0, 0.5]),
    )
    assert np.abs(signal[1]).max() == pytest.approx(0.5 * np.abs(signal[0]).max())


def test_a_named_array_carries_its_own_geometry():
    grid = _grid()
    transducer = fullwave.Transducer.l7_4(grid)
    geometry = transducer.transducer_geometry
    assert geometry.number_elements == 128
    assert geometry.element_width_m == pytest.approx(0.298e-3 - 0.048e-3, rel=0.05)


def test_a_named_array_carries_its_own_excitation():
    grid = _grid()
    assert fullwave.Transducer.l7_4(grid).pulse.pressure == pytest.approx(3.162e5)
    assert fullwave.Transducer.p4_1c(grid).pulse.pressure == pytest.approx(1.0e5)


def test_a_curved_array_sits_on_an_arc():
    grid = fullwave.Grid(
        domain_size=(9.0e-2, 9.0e-2), f0=3.7e6, duration=2.0e-6, c0=1540.0, ppw=8, cfl=0.4
    )
    transducer = fullwave.Transducer.c5_2v(grid)
    rows = np.unique(np.asarray(transducer.transducer_geometry._source_coords)[:, 0])
    assert len(rows) > 1


def test_a_transducer_refuses_an_element_outside_the_aperture():
    grid = _grid()
    transducer = fullwave.Transducer.p4_1c(grid)
    with pytest.raises(ValueError, match="outside the aperture"):
        transducer.synthetic_aperture(65)


def test_one_element_fires_in_a_synthetic_aperture_transmit():
    grid = _grid()
    transducer = fullwave.Transducer.p4_1c(grid)
    transducer.synthetic_aperture(3)
    identifiers = np.asarray(transducer.transducer_geometry._source_ids)
    loud = np.abs(transducer.signal).max(axis=1) > 0
    assert set(identifiers[loud].tolist()) == {3}


def test_a_sub_aperture_drives_only_its_own_elements():
    grid = _grid()
    transducer = fullwave.Transducer.p4_1c(grid)
    transducer.active_source_elements[:] = False
    transducer.active_source_elements[10:20] = True
    transducer.plane_wave()
    assert transducer.signal.shape[0] == len(transducer.source_coords)
    identifiers = np.asarray(transducer.transducer_geometry._source_ids)
    active = np.isin(identifiers, np.arange(11, 21))
    assert transducer.signal.shape[0] == int(active.sum())


def test_the_transducer_refuses_an_unknown_source_type():
    grid = _grid()
    with pytest.raises(ValueError, match="source_type"):
        fullwave.Transducer.p4_1c(grid, source_type="assign")


def test_the_additive_signal_is_scaled_so_the_aperture_radiates_its_pressure():
    grid = _grid()
    hard = fullwave.Transducer.p4_1c(grid, source_type="clamped")
    additive = fullwave.Transducer.p4_1c(grid)
    hard.plane_wave()
    additive.plane_wave()
    scale = 2.0 * grid.c0 * grid.dt / grid.dx
    assert np.abs(additive.source.p0_additive).max() == pytest.approx(
        scale * np.abs(hard.source.p0).max()
    )


def test_the_additive_signal_carries_the_scale_at_every_courant_number():
    for courant in (0.45, 0.4, 0.3, 0.2):
        grid = fullwave.Grid(
            domain_size=(2.0e-2, 5.0e-2),
            f0=5.208e6,
            duration=2.0e-6,
            c0=1540.0,
            ppw=12,
            cfl=courant,
        )
        hard = fullwave.Transducer.p4_1c(grid, source_type="clamped")
        additive = fullwave.Transducer.p4_1c(grid)
        hard.plane_wave()
        additive.plane_wave()
        ratio = np.abs(additive.source.p0_additive).max() / np.abs(hard.source.p0).max()
        assert ratio == pytest.approx(2.0 * courant)


def test_an_additive_transducer_builds_an_additive_source():
    grid = _grid()
    transducer = fullwave.Transducer.p4_1c(grid)
    transducer.plane_wave()
    source = transducer.source
    assert source.p0_additive is not None
    assert source.incoords.shape[0] == 0
    assert source.incoords_add.shape[0] == transducer.n_sources


def _stack_grid():
    return fullwave.Grid(
        domain_size=(3.0e-2, 5.0e-2), f0=5.208e6, duration=2.0e-6, c0=1540.0, ppw=12, cfl=0.4
    )


def _stack_maps(grid):
    shape = (grid.nx, grid.ny)
    return {
        "sound_speed": np.full(shape, 1540.0),
        "density": np.full(shape, 1000.0),
        "alpha_coeff": np.full(shape, 0.5),
        "alpha_power": np.full(shape, 1.0),
        "beta": np.zeros(shape),
    }


def _placed(grid):
    face_m = fullwave.TransducerStack.backing_thickness_m + 3 * grid.wavelength
    return fullwave.Transducer.l7_4(grid, position_m=(face_m, 1.0e-3))


def test_the_transducer_stack_refuses_a_face_too_shallow_for_the_backing():
    grid = _stack_grid()
    transducer = fullwave.Transducer.l7_4(grid, position_m=(0.0, 1.0e-3))
    with pytest.raises(ValueError, match="backing needs"):
        transducer.apply_transducer_stack(**_stack_maps(grid))


def test_the_backing_carries_the_calibrated_values():
    grid = _stack_grid()
    maps = _stack_maps(grid)
    transducer = _placed(grid)
    transducer.apply_transducer_stack(**maps, rng=np.random.default_rng(0))
    face = int(np.asarray(transducer.transducer_geometry._source_coords)[:, 0].min())
    rough = round(fullwave.TransducerStack.backing_face_roughness_m / grid.dx)
    deep = slice(0, face - rough - 1)
    assert np.unique(maps["sound_speed"][deep]) == pytest.approx([1450.0])
    assert np.unique(maps["density"][deep]) == pytest.approx([1700.0])
    assert np.unique(maps["alpha_coeff"][deep]) == pytest.approx([20.0])


def test_the_backing_face_is_rough():
    grid = _stack_grid()
    maps = _stack_maps(grid)
    transducer = _placed(grid)
    transducer.apply_transducer_stack(**maps, rng=np.random.default_rng(0))
    face = int(np.asarray(transducer.transducer_geometry._source_coords)[:, 0].min())
    rough = round(fullwave.TransducerStack.backing_face_roughness_m / grid.dx)
    band = maps["sound_speed"][face - rough : face]
    filled = float((band == 1450.0).mean())
    assert 0.0 < filled < 1.0


def test_the_layers_below_the_face_are_in_order():
    grid = _stack_grid()
    maps = _stack_maps(grid)
    transducer = _placed(grid)
    transducer.apply_transducer_stack(**maps, rng=np.random.default_rng(0))
    face = int(np.asarray(transducer.transducer_geometry._source_coords)[:, 0].min())
    below = face + transducer.transducer_geometry.element_layer_px
    column = maps["sound_speed"][:, grid.ny // 2]
    matching = below + round(fullwave.TransducerStack.matching_thickness_m / grid.dx) - 1
    lens = matching + round(fullwave.TransducerStack.lens_thickness_m / grid.dx)
    standoff = lens + round(fullwave.TransducerStack.standoff_thickness_m / grid.dx)
    assert column[matching] == pytest.approx(fullwave.TransducerStack.matching_sound_speed_m_s)
    assert column[lens] == pytest.approx(fullwave.TransducerStack.lens_sound_speed_m_s)
    assert column[standoff] == pytest.approx(fullwave.TransducerStack.standoff_sound_speed_m_s)


def test_the_element_rows_keep_the_medium_they_sit_in():
    grid = _stack_grid()
    maps = _stack_maps(grid)
    transducer = _placed(grid)
    face = int(np.asarray(transducer.transducer_geometry._source_coords)[:, 0].min())
    rows = transducer.transducer_geometry.element_layer_px
    transducer.apply_transducer_stack(**maps, rng=np.random.default_rng(0))
    assert np.unique(maps["sound_speed"][face : face + rows]) == pytest.approx([1540.0])


def test_the_backing_keeps_the_speckle_when_a_scatterer_is_given():
    grid = _stack_grid()
    maps = _stack_maps(grid)
    transducer = _placed(grid)
    speckle = np.random.default_rng(1).normal(1.0, 0.02, size=maps["density"].shape)
    transducer.apply_transducer_stack(**maps, scatterer=speckle, rng=np.random.default_rng(0))
    face = int(np.asarray(transducer.transducer_geometry._source_coords)[:, 0].min())
    rough = round(fullwave.TransducerStack.backing_face_roughness_m / grid.dx)
    deep = maps["density"][: face - rough - 1]
    assert deep.std() > 0
    assert deep.mean() == pytest.approx(1700.0, rel=0.01)


def test_a_curved_array_refuses_the_stack_because_it_is_its_own_backing():
    grid = fullwave.Grid(
        domain_size=(9.0e-2, 9.0e-2), f0=3.7e6, duration=2.0e-6, c0=1540.0, ppw=8, cfl=0.4
    )
    transducer = fullwave.Transducer.c5_2v(grid)
    shape = (grid.nx, grid.ny)
    maps = {
        "sound_speed": np.full(shape, 1540.0),
        "density": np.full(shape, 1000.0),
        "alpha_coeff": np.full(shape, 0.5),
        "alpha_power": np.full(shape, 1.0),
        "beta": np.zeros(shape),
    }
    with pytest.raises(ValueError, match="own backing"):
        transducer.apply_transducer_stack(**maps)


def test_a_named_array_centers_on_the_width_the_grid_gives_it():
    grid = fullwave.Grid(
        domain_size=(2.0e-2, 42.5e-3),
        f0=2.0e6,
        duration=2.0e-6,
        c0=1540.0,
        ppw=12,
        cfl=0.4,
    )
    geometry = fullwave.Transducer.l7_4(grid).transducer_geometry
    left = int(geometry.position_px[1])
    right = grid.ny - left - geometry.transducer_width_px
    assert left >= 0
    assert abs(left - right) <= 1


def test_a_named_array_sits_at_the_face_depth_it_is_given():
    grid = _stack_grid()
    depth_m = fullwave.TransducerStack.backing_thickness_m
    geometry = fullwave.Transducer.l7_4(grid, face_depth_m=depth_m).transducer_geometry
    assert int(geometry._source_coords[:, 0].min()) == round(depth_m / grid.dx)


def test_the_face_depth_leaves_the_lateral_centering_alone():
    grid = _stack_grid()
    depth_m = fullwave.TransducerStack.backing_thickness_m
    shallow = fullwave.Transducer.l7_4(grid).transducer_geometry
    deep = fullwave.Transducer.l7_4(grid, face_depth_m=depth_m).transducer_geometry
    assert int(deep.position_px[1]) == int(shallow.position_px[1])
    assert deep.transducer_width_px == shallow.transducer_width_px


def test_the_stack_accepts_a_face_placed_by_the_face_depth():
    grid = _stack_grid()
    maps = _stack_maps(grid)
    transducer = fullwave.Transducer.l7_4(
        grid,
        face_depth_m=fullwave.TransducerStack.backing_thickness_m,
    )
    transducer.apply_transducer_stack(**maps, rng=np.random.default_rng(0))
    face = int(np.asarray(transducer.transducer_geometry._source_coords)[:, 0].min())
    rough = round(fullwave.TransducerStack.backing_face_roughness_m / grid.dx)
    assert np.unique(maps["density"][: face - rough - 1]) == pytest.approx([1700.0])


def test_the_aperture_width_matches_the_pixels_the_geometry_built():
    for hertz in (1.0e6, 2.0e6, 3.7e6, 5.208e6):
        grid = fullwave.Grid(
            domain_size=(2.0e-2, 42.5e-3),
            f0=hertz,
            duration=2.0e-6,
            c0=1540.0,
            ppw=12,
            cfl=0.4,
        )
        geometry = fullwave.Transducer.l7_4(grid).transducer_geometry
        lateral = geometry._source_coords[:, 1]
        assert geometry.transducer_width_px == int(lateral.max() - lateral.min()) + 1


def test_a_curved_array_that_misses_the_grid_says_so():
    grid = fullwave.Grid(
        domain_size=(3.0e-2, 1.8e-2, 0.8e-2),
        f0=2.0e6,
        duration=2.0e-6,
        c0=1540.0,
        ppw=6,
        cfl=0.4,
    )
    with pytest.raises(ValueError, match="no pixel on the grid"):
        fullwave.TransducerGeometry(
            grid,
            number_elements=32,
            element_width_m=0.0,
            element_spacing_m=0.5e-3,
            element_height_m=4.0e-3,
            element_layer_px=18,
            position_m=(0.0, 0.0, 2.0e-3),
            radius=4.0e-2,
            zero_offset=2.0e-3,
        )
