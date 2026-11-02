"""Tests of Solver.run_batch: several transmits on one medium in one simulation."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import fullwave
from fullwave.solver import launcher as launcher_module
from fullwave.solver import solver as solver_module
from fullwave.utils.gpu_memory import GPU_MEMORY_FRACTION, GPU_MEMORY_RESERVE_BYTES

N_FRAMES = 3


def free_bytes_for_members(n_members: int, *, shared_bytes: int, per_member_bytes: int) -> int:
    """Return the smallest free memory that fits ``n_members`` members."""
    needed_bytes = shared_bytes + n_members * per_member_bytes + GPU_MEMORY_RESERVE_BYTES
    return int(needed_bytes / GPU_MEMORY_FRACTION) + 1


def _grid() -> fullwave.Grid:
    domain_size = (1e-3, 1e-3)
    c0 = 1540
    return fullwave.Grid(domain_size=domain_size, f0=1e6, duration=domain_size[0] / c0 * 2, c0=c0)


def _source_mask(grid: fullwave.Grid) -> np.ndarray:
    mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    mask[grid.nx // 2, :] = True
    return mask


def _transmit(grid: fullwave.Grid, scale: float) -> fullwave.Source:
    """Return one transmit: the same source points, with signals scaled by ``scale``."""
    mask = _source_mask(grid)
    signals = np.outer(np.arange(1, mask.sum() + 1), np.linspace(0.0, 1.0, grid.nt))
    return fullwave.Source(scale * signals, mask)


def _build_solver(tmp_path: Path, binary_name: str | None) -> fullwave.Solver:
    """Return a Solver on a fake binary, or on the released binaries for None."""
    grid = _grid()
    shape = (grid.nx, grid.ny)
    medium = fullwave.Medium(
        grid=grid,
        sound_speed=1540 * np.ones(shape),
        density=1000 * np.ones(shape),
        alpha_coeff=0.5 * np.ones(shape),
        alpha_power=1.0 * np.ones(shape),
        beta=np.zeros(shape),
        use_isotropic_relaxation=True,
    )
    sensor = fullwave.Sensor(mask=np.ones(shape, dtype=bool), sampling_modulus_time=2)
    binary = None
    if binary_name is not None:
        binary = tmp_path / binary_name
        binary.touch()
    return fullwave.Solver(
        work_dir=tmp_path / "work",
        grid=grid,
        medium=medium,
        source=_transmit(grid, 1.0),
        sensor=sensor,
        path_fullwave_simulation_bin=binary,
        use_isotropic_relaxation=True,
        verify_gpu=False,
    )


@pytest.fixture(autouse=True)
def _no_gpu(monkeypatch):
    monkeypatch.setattr(launcher_module.Launcher, "_verify_cuda_devices_exist", lambda _: "0")
    monkeypatch.setattr(solver_module, "free_gpu_memory_bytes", lambda _: None)


@pytest.fixture
def batch_solver(tmp_path):
    return _build_solver(tmp_path, "fullwave2_2d_n_relax_multi_gpu_batch")


def _read_int(path: Path) -> int:
    return int(np.fromfile(path, dtype=np.int32)[0])


def test_run_batch_writes_the_signals_of_every_member_in_order(batch_solver):
    transmits = [_transmit(batch_solver.grid, scale) for scale in (1.0, -0.5, 2.0)]
    (simulation_dir,) = batch_solver.run_batch(transmits, generate_input_only=True)

    assert _read_int(simulation_dir / "nsims.dat") == 3
    expected = np.concatenate([t.icmat for t in transmits], axis=0).astype(np.float32)
    written = np.fromfile(simulation_dir / "icmat.dat", dtype=np.float32)
    np.testing.assert_array_equal(written, expected.ravel())


def _short_transmit(grid: fullwave.Grid, *, drive_samples: int, delay: int) -> fullwave.Source:
    """Return a transmit whose drive is ``drive_samples`` long, shifted by ``delay`` samples."""
    mask = _source_mask(grid)
    signal = np.zeros((int(mask.sum()), drive_samples))
    signal[:, delay:] = np.linspace(0.5, 1.0, drive_samples - delay)
    return fullwave.Source(signal, mask)


def test_run_batch_pads_transmits_of_different_drive_lengths_with_zeros(batch_solver):
    short = _short_transmit(batch_solver.grid, drive_samples=40, delay=5)
    long = _short_transmit(batch_solver.grid, drive_samples=90, delay=30)
    (simulation_dir,) = batch_solver.run_batch([short, long], generate_input_only=True)

    length = 90
    written = np.fromfile(simulation_dir / "icmat.dat", dtype=np.float32)
    rows = short.icmat.shape[0]
    written = written.reshape(2 * rows, length)
    np.testing.assert_array_equal(written[:rows, :40], short.icmat.astype(np.float32))
    assert not written[:rows, 40:].any()
    np.testing.assert_array_equal(written[rows:, :90], long.icmat.astype(np.float32))
    assert short.icmat.shape[1] == 40
    assert _read_int(simulation_dir / "nTic.dat") == length


def test_run_batch_pads_to_the_longest_drive_when_the_solver_source_is_shorter(batch_solver):
    batch_solver.source = _short_transmit(batch_solver.grid, drive_samples=20, delay=0)
    short = _short_transmit(batch_solver.grid, drive_samples=40, delay=5)
    long = _short_transmit(batch_solver.grid, drive_samples=90, delay=30)
    (simulation_dir,) = batch_solver.run_batch([short, long], generate_input_only=True)

    rows = short.icmat.shape[0]
    written = np.fromfile(simulation_dir / "icmat.dat", dtype=np.float32)
    assert written.size == 2 * rows * 90


def test_run_batch_still_refuses_transmits_with_other_source_points(batch_solver):
    grid = batch_solver.grid
    mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    mask[grid.nx // 2 + 1, :] = True
    other = fullwave.Source(np.ones((int(mask.sum()), 40)), mask)
    with pytest.raises(ValueError, match="does not match"):
        batch_solver.run_batch([_transmit(grid, 1.0), other], generate_input_only=True)


def test_run_batch_writes_the_same_coordinates_as_a_single_run(batch_solver):
    transmits = [_transmit(batch_solver.grid, scale) for scale in (1.0, 2.0)]
    (batch_dir,) = batch_solver.run_batch(transmits, generate_input_only=True)
    single_dir = batch_solver.run("single", generate_input_only=True)

    for name in ("icc.dat", "outc.dat", "ncoords.dat", "nTic.dat"):
        assert (batch_dir / name).read_bytes() == (single_dir / name).read_bytes(), name


def test_a_single_run_removes_a_member_count_left_by_a_batch(batch_solver):
    transmits = [_transmit(batch_solver.grid, scale) for scale in (1.0, 2.0)]
    (simulation_dir,) = batch_solver.run_batch(
        transmits,
        simulation_dir_name="shared",
        generate_input_only=True,
    )
    assert (simulation_dir / "nsims.dat").exists()

    batch_solver.run("shared", generate_input_only=True)

    assert not (simulation_dir / "nsims.dat").exists()


def test_run_batch_splits_the_transmits_into_runs_of_at_most_max_members(batch_solver):
    transmits = [_transmit(batch_solver.grid, scale) for scale in range(1, 6)]
    directories = batch_solver.run_batch(
        transmits,
        max_members_per_run=2,
        generate_input_only=True,
    )

    assert [d.name for d in directories] == ["txrx_batch_0", "txrx_batch_1", "txrx_batch_2"]
    assert [_read_int(d / "nsims.dat") for d in directories] == [2, 2, 1]
    last = np.fromfile(directories[2] / "icmat.dat", dtype=np.float32)
    np.testing.assert_array_equal(last, transmits[4].icmat.astype(np.float32).ravel())


def test_run_batch_sizes_the_runs_from_the_free_gpu_memory(batch_solver, monkeypatch):
    sensor = batch_solver.pml_builder.extended_sensor
    batch_solver.pml_builder.run(use_pml=batch_solver.use_pml)
    ((_, one, *_),) = batch_solver._gpu_memory_bytes(sensor, n_members=1)
    ((_, two, *_),) = batch_solver._gpu_memory_bytes(sensor, n_members=2)
    per_member = two - one
    shared = one - per_member
    free = free_bytes_for_members(3, shared_bytes=shared, per_member_bytes=per_member)
    monkeypatch.setattr(solver_module, "free_gpu_memory_bytes", lambda _: {"0": free})
    transmits = [_transmit(batch_solver.grid, scale) for scale in range(1, 8)]

    directories = batch_solver.run_batch(transmits, generate_input_only=True)

    assert [_read_int(d / "nsims.dat") for d in directories] == [3, 2, 2]


def test_run_batch_runs_every_member_in_one_run_when_the_free_memory_is_unknown(batch_solver):
    transmits = [_transmit(batch_solver.grid, scale) for scale in range(1, 5)]
    (simulation_dir,) = batch_solver.run_batch(transmits, generate_input_only=True)
    assert _read_int(simulation_dir / "nsims.dat") == 4


def test_run_batch_caps_a_run_at_the_default_and_splits_evenly(batch_solver, monkeypatch):
    monkeypatch.setattr(solver_module, "DEFAULT_MAX_MEMBERS_PER_RUN", 3)
    transmits = [_transmit(batch_solver.grid, scale) for scale in range(1, 8)]

    directories = batch_solver.run_batch(transmits, generate_input_only=True)

    assert [_read_int(d / "nsims.dat") for d in directories] == [3, 2, 2]


def test_split_evenly_gives_runs_of_nearly_equal_size():
    assert [len(run) for run in solver_module._split_evenly(list(range(128)), 32)] == [32] * 4
    assert [len(run) for run in solver_module._split_evenly(list(range(70)), 32)] == [24, 23, 23]
    assert solver_module._split_evenly(list(range(5)), 2) == [[0, 1], [2, 3], [4]]


def test_the_member_bytes_of_the_memory_estimate(batch_solver):
    sensor = batch_solver.pml_builder.extended_sensor
    batch_solver.pml_builder.run(use_pml=batch_solver.use_pml)
    totals = [batch_solver._gpu_memory_bytes(sensor, n_members=n)[0][1] for n in (1, 2, 3)]
    assert totals[0] < totals[1]
    # Each member adds the same bytes; the medium is held once.
    assert totals[2] - totals[1] == totals[1] - totals[0]
    assert totals[1] < 2 * totals[0]


def test_run_batch_returns_the_output_of_every_member_in_order(batch_solver):
    transmits = [_transmit(batch_solver.grid, scale) for scale in (1.0, 2.0, 3.0)]
    n_sensors = batch_solver.pml_builder.extended_sensor.n_sensors

    def fake_run(_self, simulation_dir, *, load_results=True):
        n_members = _read_int(simulation_dir / "nsims.dat")
        for member in range(n_members):
            name = "genout.dat" if member == 0 else f"genout_s{member}.dat"
            values = member * 1000 + np.arange(N_FRAMES * n_sensors, dtype=np.float32)
            values.tofile(simulation_dir / name)
        return simulation_dir.absolute() / "genout.dat"

    with patch.object(launcher_module.Launcher, "run", fake_run):
        result = batch_solver.run_batch(transmits)

    assert result.shape == (3, n_sensors, N_FRAMES)
    for member in range(3):
        expected = member * 1000 + np.arange(N_FRAMES * n_sensors).reshape(-1, n_sensors).T
        np.testing.assert_array_equal(result[member], expected)


def test_run_batch_returns_the_output_paths_in_member_order(batch_solver):
    transmits = [_transmit(batch_solver.grid, scale) for scale in (1.0, 2.0)]

    def fake_run(_self, simulation_dir, *, load_results=True):
        return simulation_dir.absolute() / "genout.dat"

    with patch.object(launcher_module.Launcher, "run", fake_run):
        paths = batch_solver.run_batch(transmits, max_members_per_run=1, load_results=False)

    assert [(p.parent.name, p.name) for p in paths] == [
        ("txrx_batch_0", "genout.dat"),
        ("txrx_batch_1", "genout.dat"),
    ]


def test_run_batch_refuses_a_binary_without_batch_support(tmp_path):
    solver = _build_solver(tmp_path, "fullwave2_2d_n_relax_multi_gpu_cuda129")
    transmits = [_transmit(solver.grid, scale) for scale in (1.0, 2.0)]

    with pytest.raises(ValueError, match="cannot run several transmits"):
        solver.run_batch(transmits, generate_input_only=True)

    # One transmit per run needs no batch support.
    directories = solver.run_batch(transmits, max_members_per_run=1, generate_input_only=True)
    assert len(directories) == 2


def test_run_batch_refuses_velocity_output_of_several_transmits(batch_solver, monkeypatch):
    monkeypatch.setenv("OUTPUT_VELOCITY", "1")
    transmits = [_transmit(batch_solver.grid, scale) for scale in (1.0, 2.0)]
    with pytest.raises(NotImplementedError, match="velocity output"):
        batch_solver.run_batch(transmits, generate_input_only=True)


def test_run_batch_refuses_a_transmit_with_other_coordinates(batch_solver):
    grid = batch_solver.grid
    mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    mask[grid.nx // 3, :] = True
    other = fullwave.Source(np.ones((mask.sum(), grid.nt)), mask)

    with pytest.raises(ValueError, match="source 1 of the batch"):
        batch_solver.run_batch([_transmit(grid, 1.0), other], generate_input_only=True)


def test_run_batch_refuses_an_empty_list_and_a_run_size_below_one(batch_solver):
    with pytest.raises(ValueError, match="at least one source"):
        batch_solver.run_batch([], generate_input_only=True)
    with pytest.raises(ValueError, match="max_members_per_run"):
        batch_solver.run_batch(
            [_transmit(batch_solver.grid, 1.0)],
            max_members_per_run=0,
            generate_input_only=True,
        )


@pytest.mark.parametrize(
    ("use_exponential_attenuation", "expected"),
    [
        (False, "fullwave2_2d_n_relax_multi_gpu_batch_cuda124"),
        (True, "fullwave2_2d_exponential_attenuation_multi_gpu_batch_cuda124"),
    ],
)
def test_the_batch_binary_name_carries_batch_before_the_cuda_version(
    use_exponential_attenuation,
    expected,
):
    with (
        patch.object(solver_module, "_make_cuda_arch_option", return_value="sm_89"),
        patch.object(solver_module, "retrieve_cuda_version", return_value=12.4),
    ):
        path = solver_module._retrieve_fullwave_simulation_path(
            use_exponential_attenuation=use_exponential_attenuation,
            batch=True,
        )
    assert path.name == expected


def test_there_is_no_3d_batch_binary():
    with (
        patch.object(solver_module, "_make_cuda_arch_option", return_value="sm_89"),
        patch.object(solver_module, "retrieve_cuda_version", return_value=12.4),
        pytest.raises(NotImplementedError, match="2D GPU simulations only"),
    ):
        solver_module._retrieve_fullwave_simulation_path(is_3d=True, batch=True)


@pytest.fixture
def default_binary_solver(tmp_path, monkeypatch):
    """Return a Solver built without a binary path, with a fake binary lookup."""
    requested = []

    def fake_path(*, batch=False, **_):
        name = f"fullwave2_2d_n_relax_multi_gpu{'_batch' if batch else ''}_cuda124"
        binary = tmp_path / "bins" / name
        binary.parent.mkdir(exist_ok=True)
        binary.touch()
        requested.append(name)
        return binary

    monkeypatch.setattr(solver_module, "_retrieve_fullwave_simulation_path", fake_path)
    monkeypatch.setattr(solver_module, "ensure_binary", lambda path: path)
    return _build_solver(tmp_path, None), requested


def test_run_batch_uses_the_batch_binary_when_the_user_passed_no_binary(default_binary_solver):
    solver, requested = default_binary_solver
    transmits = [_transmit(solver.grid, scale) for scale in (1.0, 2.0)]

    (simulation_dir,) = solver.run_batch(transmits, generate_input_only=True)

    assert (simulation_dir / "fullwave2_2d_n_relax_multi_gpu_batch_cuda124").is_symlink()
    assert requested[-1] == "fullwave2_2d_n_relax_multi_gpu_batch_cuda124"
    # A later single run uses the normal binary again.
    assert solver.path_fullwave_simulation_bin.name == "fullwave2_2d_n_relax_multi_gpu_cuda124"
    single_dir = solver.run("single", generate_input_only=True)
    assert (single_dir / "fullwave2_2d_n_relax_multi_gpu_cuda124").is_symlink()


def test_run_batch_needs_no_batch_binary_for_one_transmit_per_run(default_binary_solver):
    solver, requested = default_binary_solver
    transmits = [_transmit(solver.grid, scale) for scale in (1.0, 2.0)]

    solver.run_batch(transmits, max_members_per_run=1, generate_input_only=True)

    assert "fullwave2_2d_n_relax_multi_gpu_batch_cuda124" not in requested
