"""Unit tests for MachBeamformer; only the last one needs a GPU and skips without it."""

from types import SimpleNamespace

import numpy as np
import pytest

from fullwave.beamformer import mach_beamformer
from fullwave.beamformer.mach_beamformer import (
    MachBeamformer,
    face_crossing_time_s,
    focused_arrival_s,
    plane_wave_arrival_s,
    synthetic_aperture_arrival_s,
    user_arrival_s,
)

SOUND_SPEED = 1540.0


def _line_elements(count: int = 9, pitch_m: float = 0.3e-3) -> np.ndarray:
    positions = np.zeros((count, 3))
    positions[:, 0] = (np.arange(count) - (count - 1) / 2) * pitch_m
    return positions


def test_plane_wave_straight_ahead_arrives_with_depth_over_sound_speed():
    points = np.array([[0.0, 0.0, 10e-3], [3e-3, 0.0, 20e-3]])
    arrival = plane_wave_arrival_s(points, _line_elements(), angle_deg=0.0, sound_speed=SOUND_SPEED)
    np.testing.assert_allclose(arrival, points[:, 2] / SOUND_SPEED, rtol=1e-12)


def test_plane_wave_steered_matches_projection_and_earliest_element_is_zero():
    elements = _line_elements()
    angle = 15.0
    points = np.array([[2e-3, 0.0, 12e-3]])
    arrival = plane_wave_arrival_s(points, elements, angle_deg=angle, sound_speed=SOUND_SPEED)
    sin, cos = np.sin(np.radians(angle)), np.cos(np.radians(angle))
    earliest_projection = elements[:, 0].min() * sin
    expected = (2e-3 * sin + 12e-3 * cos - earliest_projection) / SOUND_SPEED
    np.testing.assert_allclose(arrival, expected, rtol=1e-12)
    at_elements = plane_wave_arrival_s(elements, elements, angle_deg=angle, sound_speed=SOUND_SPEED)
    assert at_elements.min() == pytest.approx(0.0, abs=1e-15)


def test_focused_on_axis_converges_then_diverges_around_the_focus():
    elements = _line_elements()
    focus = np.array([0.0, 0.0, 15e-3])
    farthest = np.linalg.norm(elements - focus, axis=1).max()
    points = np.array([[0.0, 0.0, 10e-3], [0.0, 0.0, 15e-3], [0.0, 0.0, 20e-3]])
    arrival = focused_arrival_s(points, elements, focus_m=focus, sound_speed=SOUND_SPEED)
    expected = np.array([farthest - 5e-3, farthest, farthest + 5e-3]) / SOUND_SPEED
    np.testing.assert_allclose(arrival, expected, rtol=1e-12)


def test_focused_rejects_a_focus_behind_the_face():
    with pytest.raises(ValueError, match="in front of the probe face"):
        focused_arrival_s(
            np.zeros((1, 3)),
            _line_elements(),
            focus_m=np.array([0.0, 0.0, -1e-3]),
            sound_speed=SOUND_SPEED,
        )


def test_synthetic_aperture_is_distance_from_the_element():
    elements = _line_elements()
    points = np.array([[1e-3, 0.0, 8e-3]])
    arrival = synthetic_aperture_arrival_s(points, elements, element=3, sound_speed=SOUND_SPEED)
    distance = np.hypot(1e-3 - elements[2, 0], 8e-3)
    np.testing.assert_allclose(arrival, distance / SOUND_SPEED, rtol=1e-12)


def test_synthetic_aperture_rejects_an_element_outside_the_aperture():
    with pytest.raises(ValueError, match="outside the aperture"):
        synthetic_aperture_arrival_s(
            np.zeros((1, 3)), _line_elements(), element=0, sound_speed=SOUND_SPEED
        )


def test_user_arrival_checks_the_point_count():
    assert user_arrival_s(np.zeros((2, 5)), n_points=5).shape == (2, 5)
    with pytest.raises(ValueError, match="arrival_s must have shape"):
        user_arrival_s(np.zeros(4), n_points=5)


def test_face_crossing_time_adds_the_travel_from_the_back_source_row():
    dt, dx, c0 = 1e-8, 5e-5, 1500.0
    time = np.arange(400) * dt

    def pulse(peak_s):
        return np.sin(2 * np.pi * 2e6 * (time - peak_s)) * np.exp(-(((time - peak_s) / 5e-7) ** 2))

    signal = np.zeros((3, 400))
    signal[1] = pulse(2e-6)
    signal[2] = pulse(2.5e-6)
    transducer = SimpleNamespace(
        signal=signal,
        grid=SimpleNamespace(dt=dt, dx=dx, c0=c0),
        source_coords=np.array([[10, 0], [11, 0], [12, 0]]),
        face_row_px=14,
    )
    expected = 2e-6 + 4 * dx / c0
    assert face_crossing_time_s(transducer) == pytest.approx(expected, abs=3e-8)


def test_beamform_without_mach_raises_import_error_that_names_the_extra(monkeypatch):
    def _missing(name, *args, **kwargs):
        raise ImportError(f"No module named {name}")

    monkeypatch.setattr(mach_beamformer.importlib, "import_module", _missing)
    mach_beamformer._load_mach_modules.cache_clear()
    beamformer = MachBeamformer(
        lateral_position_m=np.zeros(1),
        axial_position_m=np.ones(1) * 1e-2,
        receiver_positions_m=_line_elements(),
        sound_speed=SOUND_SPEED,
        center_frequency_hz=2e6,
        sampling_frequency_hz=4e7,
    )
    try:
        with pytest.raises(ImportError, match=r"fullwave25\[mach\]"):
            beamformer.beamform(np.zeros((9, 100)), tx_arrival_s=np.zeros(1))
    finally:
        mach_beamformer._load_mach_modules.cache_clear()


def _mach_modules_or_skip():
    """Return the loaded mach modules after a GPU round trip, or skip the test.

    Any exception counts, because a missing CUDA library raises ImportError (not
    ModuleNotFoundError) at import or at the first call.
    """
    try:
        cupy, _, _ = mach_beamformer._load_mach_modules()
        cupy.asnumpy(cupy.asarray(np.ones(4)) * 2.0)
    except Exception as error:
        pytest.skip(f"mach beamforming is unusable on this machine: {error}")


def test_point_target_peaks_at_its_position_for_a_plane_wave():
    _mach_modules_or_skip()
    elements = _line_elements(count=64)
    sampling_hz, center_hz = 40e6, 2e6
    target = np.array([0.0, 0.0, 20e-3])
    n_time = 1600
    time = np.arange(n_time) / sampling_hz
    delay = (target[2] + np.linalg.norm(elements - target, axis=1)) / SOUND_SPEED
    lag = time[None, :] - delay[:, None]
    rf = np.cos(2 * np.pi * center_hz * lag) * np.exp(-((lag / 0.5e-6) ** 2))

    beamformer = MachBeamformer(
        lateral_position_m=np.arange(-2e-3, 2e-3 + 1e-9, 0.05e-3),
        axial_position_m=np.arange(17e-3, 23e-3, 0.05e-3),
        receiver_positions_m=elements,
        sound_speed=SOUND_SPEED,
        center_frequency_hz=center_hz,
        sampling_frequency_hz=sampling_hz,
        f_number=1.5,
    )
    image = beamformer.beamform(
        rf.astype(np.float32),
        tx_arrival_s=beamformer.plane_wave_arrival_s(angle_deg=0.0),
    )
    i_axial, i_lateral = np.unravel_index(np.argmax(np.abs(image)), image.shape)
    quarter_wavelength_m = SOUND_SPEED / center_hz / 4
    assert abs(beamformer.axial_position_m[i_axial] - target[2]) < quarter_wavelength_m
    assert abs(beamformer.lateral_position_m[i_lateral] - target[0]) < quarter_wavelength_m


def _build_beamformer(**kwargs):
    return MachBeamformer(
        lateral_position_m=np.zeros(1),
        axial_position_m=np.ones(1) * 1e-2,
        receiver_positions_m=_line_elements(),
        sound_speed=SOUND_SPEED,
        center_frequency_hz=2e6,
        sampling_frequency_hz=4e7,
        **kwargs,
    )


def test_defaults_are_f_number_one_and_tukey_half():
    beamformer = _build_beamformer()
    assert beamformer.f_number == 1.0
    assert beamformer.tukey_alpha == 0.5


def test_a_larger_f_number_gives_a_wider_lateral_beam_on_the_gpu():
    _mach_modules_or_skip()
    elements = _line_elements(count=64)
    sampling_hz, center_hz = 40e6, 2e6
    target = np.array([0.0, 0.0, 20e-3])
    time = np.arange(1600) / sampling_hz
    delay = (target[2] + np.linalg.norm(elements - target, axis=1)) / SOUND_SPEED
    lag = time[None, :] - delay[:, None]
    rf = (np.cos(2 * np.pi * center_hz * lag) * np.exp(-((lag / 0.5e-6) ** 2))).astype(np.float32)
    lateral = np.arange(-4e-3, 4e-3 + 1e-9, 0.05e-3)
    widths = []
    for f_number in (1.0, 2.0):
        beamformer = MachBeamformer(
            lateral_position_m=lateral,
            axial_position_m=np.array([20e-3]),
            receiver_positions_m=elements,
            sound_speed=SOUND_SPEED,
            center_frequency_hz=center_hz,
            sampling_frequency_hz=sampling_hz,
            f_number=f_number,
        )
        envelope = np.abs(
            beamformer.beamform(rf, tx_arrival_s=beamformer.plane_wave_arrival_s(angle_deg=0.0))
        )[0]
        widths.append(np.count_nonzero(envelope > envelope.max() / 2))
    assert widths[1] > widths[0]
