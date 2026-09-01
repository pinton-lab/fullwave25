"""Tests for the additive source conversion."""

import numpy as np
import pytest

import fullwave
from fullwave.solver import source_type


def _row_coords(shape, row=0):
    return np.array([[row, column] for column in range(shape[1])], dtype=np.int64)


def _source(shape, n_time=5, amplitude=1.0, row=0):
    coords = _row_coords(shape, row)
    p0 = np.full((len(coords), n_time), amplitude)
    return fullwave.Source(p0=p0, coords=coords, grid_shape=shape)


class _Medium:
    def __init__(self, sound_speed):
        self.sound_speed = sound_speed


class _Grid:
    def __init__(self, dt, dx, f0=1e6):
        self.dt = dt
        self.dx = dx
        self.f0 = f0


class _OnTheDevice:
    """A stand-in for a device array, which gives a host copy through `get`.

    It records the size of every array that leaves the device, so a test can
    show that the whole map never moves.
    """

    def __init__(self, array, moved=None):
        self._array = array
        self.moved = [] if moved is None else moved

    def __getitem__(self, index):
        return _OnTheDevice(self._array[index], self.moved)

    def get(self):
        self.moved.append(self._array.size)
        return self._array


def test_scale_is_twice_the_courant_number():
    dx = 9.625e-5
    for courant in (0.2, 0.1, 0.05):
        dt = courant * dx / 1540.0
        assert source_type.additive_signal_scale(1540.0, dt, dx) == pytest.approx(2 * courant)


def test_scale_follows_the_sound_speed():
    dx = 1e-4
    dt = 1e-8
    slow = source_type.additive_signal_scale(1000.0, dt, dx)
    fast = source_type.additive_signal_scale(2000.0, dt, dx)
    assert fast == pytest.approx(2 * slow)


def test_source_row_finds_the_row():
    assert source_type.source_row(_row_coords((6, 4), row=3), (6, 4)) == 3


def test_source_row_rejects_a_partial_row():
    with pytest.raises(ValueError, match="one whole row of the grid"):
        source_type.source_row(_row_coords((6, 4))[:2], (6, 4))


def test_source_row_rejects_two_rows():
    coords = np.concatenate([_row_coords((6, 4), row=0), _row_coords((6, 4), row=2)])
    with pytest.raises(ValueError, match="one whole row of the grid"):
        source_type.source_row(coords, (6, 4))


def test_source_row_never_allocates_the_grid():
    coords = np.array([[7, 0]], dtype=np.int64)
    with pytest.raises(ValueError, match="one whole row of the grid"):
        source_type.source_row(coords, (10**6, 10**6))


def test_source_row_accepts_a_three_dimensional_row():
    shape = (4, 3, 2)
    coords = np.array(
        [[1, y, z] for y in range(shape[1]) for z in range(shape[2])],
        dtype=np.int64,
    )
    assert source_type.source_row(coords, shape) == 1


def test_node_sound_speeds_from_a_scalar():
    coords = np.array([[0, 0], [0, 1], [0, 2]])
    assert source_type.node_sound_speeds(1540.0, coords).tolist() == [1540.0] * 3


def test_node_sound_speeds_follow_the_map():
    speeds = np.full((6, 4), 1540.0)
    speeds[0, 1] = 1600.0
    coords = np.array([[0, 0], [0, 1], [0, 2]])
    assert source_type.node_sound_speeds(speeds, coords).tolist() == [1540.0, 1600.0, 1540.0]


def test_node_sound_speeds_pass_a_per_position_array_through():
    coords = np.array([[0, 0], [0, 1], [0, 2]])
    per_position = np.array([1500.0, 1540.0, 1580.0])
    assert source_type.node_sound_speeds(per_position, coords).tolist() == per_position.tolist()


def _relaxation_at(positions, kappa=1.0):
    return {
        "kappa_x1": np.full(positions, kappa),
        "kappa_x2": np.full(positions, kappa),
        "d_x1_nu1": np.zeros(positions),
        "alpha_x1_nu1": np.zeros(positions),
        "d_x2_nu1": np.zeros(positions),
        "alpha_x2_nu1": np.zeros(positions),
    }


def test_relaxation_phase_speed_matches_the_stored_speed_without_relaxation():
    speeds = source_type.relaxation_phase_speed(_relaxation_at(3), 1540.0, 1e6)
    assert speeds == pytest.approx(np.full(3, 1540.0))


def test_relaxation_phase_speed_follows_kappa():
    speeds = source_type.relaxation_phase_speed(_relaxation_at(3, kappa=1.01), 1540.0, 1e6)
    assert speeds == pytest.approx(np.full(3, 1540.0 / 1.01))


def test_conversion_empties_the_assigned_positions():
    shape = (6, 4)
    converted = source_type.as_additive_source(
        _source(shape),
        _Grid(dt=0.2 * 1e-4 / 1540.0, dx=1e-4),
        _Medium(np.full(shape, 1540.0)),
    )
    assert converted.incoords.shape[0] == 0
    assert converted.incoords_add.shape[0] == shape[1]


def test_conversion_scales_the_drive():
    shape = (6, 4)
    dx = 1e-4
    courant = 0.2
    converted = source_type.as_additive_source(
        _source(shape, amplitude=1.0e5),
        _Grid(dt=courant * dx / 1540.0, dx=dx),
        _Medium(np.full(shape, 1540.0)),
    )
    assert converted.p0_additive.max() == pytest.approx(1.0e5 * 2 * courant)


def test_conversion_keeps_the_source_position():
    shape = (6, 4)
    converted = source_type.as_additive_source(
        _source(shape, row=2),
        _Grid(dt=0.2 * 1e-4 / 1540.0, dx=1e-4),
        _Medium(np.full(shape, 1540.0)),
    )
    assert np.unique(converted.incoords_add[:, 0]).tolist() == [2]


def test_a_varying_row_scales_node_by_node():
    shape = (6, 4)
    dx = 1e-4
    courant = 0.2
    speeds = np.full(shape, 1540.0)
    speeds[0, 1] = 3080.0
    converted = source_type.as_additive_source(
        _source(shape, amplitude=1.0e5),
        _Grid(dt=courant * dx / 1540.0, dx=dx),
        _Medium(speeds),
    )
    drives = converted.p0_additive[:, 0]
    column = converted.incoords_add[:, 1].tolist().index(1)
    assert drives[column] == pytest.approx(1.0e5 * 2 * courant * 2)
    assert drives.max() == pytest.approx(1.0e5 * 2 * courant * 2)


def test_a_device_sound_speed_map_reaches_the_host():
    shape = (6, 4)
    dx = 1e-4
    courant = 0.2
    grid = _Grid(dt=courant * dx / 1540.0, dx=dx)
    speeds = np.full(shape, 1540.0)
    on_the_host = source_type.as_additive_source(
        _source(shape, amplitude=1.0e5), grid, _Medium(speeds)
    )
    on_the_device = source_type.as_additive_source(
        _source(shape, amplitude=1.0e5), grid, _Medium(_OnTheDevice(speeds))
    )
    assert on_the_device.p0_additive == pytest.approx(on_the_host.p0_additive)


def test_device_relaxation_parameters_reach_the_host():
    shape = (6, 4)
    dx = 1e-4
    courant = 0.2
    grid = _Grid(dt=courant * dx / 1540.0, dx=dx)
    relaxation = _relaxation_at(shape, kappa=1.01)
    medium = _Medium(np.full(shape, 1540.0))
    medium.relaxation_param_dict = relaxation
    on_the_host = source_type.as_additive_source(_source(shape, amplitude=1.0e5), grid, medium)

    on_the_device_medium = _Medium(_OnTheDevice(np.full(shape, 1540.0)))
    on_the_device_medium.relaxation_param_dict = {
        name: _OnTheDevice(value) for name, value in relaxation.items()
    }
    on_the_device = source_type.as_additive_source(
        _source(shape, amplitude=1.0e5), grid, on_the_device_medium
    )
    assert on_the_device.p0_additive == pytest.approx(on_the_host.p0_additive)
    assert on_the_host.p0_additive.max() == pytest.approx(1.0e5 * 2 * courant / 1.01)


def test_the_whole_map_never_leaves_the_device():
    shape = (6, 4)
    dx = 1e-4
    courant = 0.2
    moved = []
    relaxation = {
        name: _OnTheDevice(value, moved)
        for name, value in _relaxation_at(shape, kappa=1.01).items()
    }
    medium = _Medium(_OnTheDevice(np.full(shape, 1540.0), moved))
    medium.relaxation_param_dict = relaxation
    source_type.as_additive_source(
        _source(shape, amplitude=1.0e5), _Grid(dt=courant * dx / 1540.0, dx=dx), medium
    )
    assert moved
    assert max(moved) == shape[1]


class _MediumThatLooksUp:
    """A stand-in for a Medium that builds its relaxation parameters on demand."""

    def __init__(self, sound_speed, kappa):
        self.sound_speed = sound_speed
        self._kappa = kappa
        self.asked_for = None

    def relaxation_parameters_at(self, coords):
        self.asked_for = len(coords)
        return _relaxation_at(len(coords), self._kappa), np.full(len(coords), 1540.0)


def test_a_medium_that_looks_up_is_asked_for_the_source_positions_alone():
    shape = (6, 4)
    dx = 1e-4
    courant = 0.2
    medium = _MediumThatLooksUp(np.full(shape, 1540.0), kappa=1.01)
    converted = source_type.as_additive_source(
        _source(shape, amplitude=1.0e5), _Grid(dt=courant * dx / 1540.0, dx=dx), medium
    )
    assert medium.asked_for == shape[1]
    assert converted.p0_additive.max() == pytest.approx(1.0e5 * 2 * courant / 1.01)


def test_the_exponential_model_keeps_the_stored_sound_speed():
    shape = (6, 4)
    dx = 1e-4
    courant = 0.2
    medium = _MediumThatLooksUp(np.full(shape, 1540.0), kappa=1.01)
    converted = source_type.as_additive_source(
        _source(shape, amplitude=1.0e5),
        _Grid(dt=courant * dx / 1540.0, dx=dx),
        medium,
        use_exponential_attenuation=True,
    )
    assert medium.asked_for is None
    assert converted.p0_additive.max() == pytest.approx(1.0e5 * 2 * courant)


def test_solver_rejects_an_unknown_source_type(tmp_path):
    grid = fullwave.Grid((1e-3, 1e-3), 1e6, 1e-6, c0=1540, ppw=8, cfl=0.2)
    shape = (grid.nx, grid.ny)
    medium = fullwave.Medium(
        grid=grid,
        sound_speed=np.full(shape, 1540.0),
        density=np.full(shape, 1000.0),
        alpha_coeff=np.zeros(shape),
        alpha_power=np.full(shape, 1.0001),
        beta=np.zeros(shape),
    )
    sensor = fullwave.Sensor(mask=np.ones(shape, dtype=bool))
    stand_in = tmp_path / "solver"
    stand_in.write_bytes(b"")
    with pytest.raises(ValueError, match="source_type"):
        fullwave.Solver(
            work_dir=tmp_path,
            grid=grid,
            medium=medium,
            source=_source(shape, n_time=grid.nt),
            sensor=sensor,
            source_type="assign",
            path_fullwave_simulation_bin=stand_in,
        )


def test_an_attenuating_position_still_takes_the_relaxation_phase_speed():
    shape = (6, 4)
    dx = 1e-4
    courant = 0.2
    medium = _MediumThatLooksUp(np.full(shape, 1540.0), kappa=1.01)
    converted = source_type.as_additive_source(
        _source(shape, amplitude=1.0e5), _Grid(dt=courant * dx / 1540.0, dx=dx), medium
    )
    assert converted.p0_additive.max() == pytest.approx(1.0e5 * 2 * courant / 1.01)
