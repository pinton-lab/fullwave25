from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import fullwave
from fullwave.solver import solver as solver_module

_FAKE_BINARY = (
    Path(__file__).parent.parent.parent
    / "fullwave"
    / "solver"
    / "bins"
    / "gpu"
    / "2d"
    / "num_relax=2"
    / "fullwave2_2d_2_relax_multi_gpu_cuda129"
)


def _build_solver(
    tmp_path,
    *,
    save_gpu_memory=False,
    cuda_device_id=None,
):
    domain_size = (1e-3, 1e-3)
    f0 = 1e6
    c0 = 1540
    duration = domain_size[0] / c0 * 2

    grid = fullwave.Grid(
        domain_size=domain_size,
        f0=f0,
        duration=duration,
        c0=c0,
    )

    shape = (grid.nx, grid.ny)
    medium = fullwave.Medium(
        grid=grid,
        sound_speed=c0 * np.ones(shape),
        density=1000 * np.ones(shape),
        alpha_coeff=0.5 * np.ones(shape),
        alpha_power=1.0 * np.ones(shape),
        beta=np.zeros(shape),
        use_isotropic_relaxation=True,
    )

    p_mask = np.zeros(shape, dtype=bool)
    p_mask[grid.nx // 2, :] = True
    source = fullwave.Source(np.ones((p_mask.sum(), grid.nt)), p_mask)

    sensor_mask = np.ones(shape, dtype=bool)
    sensor = fullwave.Sensor(mask=sensor_mask)

    return fullwave.Solver(
        work_dir=tmp_path / "work",
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        use_isotropic_relaxation=True,
        save_gpu_memory=save_gpu_memory,
        cuda_device_id=cuda_device_id,
        verify_gpu=False,
    )


@pytest.fixture(autouse=True)
def _patch_binary(monkeypatch):
    monkeypatch.setattr(
        solver_module,
        "_retrieve_fullwave_simulation_path",
        lambda **kwargs: _FAKE_BINARY,  # noqa: ARG005
    )


def test_single_gpu_relaxation_logs_info(tmp_path):
    solver = _build_solver(tmp_path)
    sensor = solver.pml_builder.extended_sensor

    with patch("fullwave.solver.solver.logger") as mock_logger:
        solver._estimate_gpu_memory(sensor)

    mock_logger.info.assert_called_once()
    fmt = mock_logger.info.call_args[0][0]
    mode_arg = mock_logger.info.call_args[0][2]
    assert "GPU memory estimate" in fmt
    assert mode_arg == "relaxation"


def test_single_gpu_no_halo(tmp_path):
    solver = _build_solver(tmp_path)
    sensor = solver.pml_builder.extended_sensor

    with patch("fullwave.solver.solver.logger") as mock_logger:
        solver._estimate_gpu_memory(sensor)

    # halo argument: dev_id, mode, saving, GB, depth, halo, lateral
    halo_value = mock_logger.info.call_args[0][6]
    assert halo_value == 0


def test_estimate_is_positive(tmp_path):
    solver = _build_solver(tmp_path)
    sensor = solver.pml_builder.extended_sensor

    with patch("fullwave.solver.solver.logger") as mock_logger:
        solver._estimate_gpu_memory(sensor)

    gb_value = mock_logger.info.call_args[0][4]
    assert gb_value > 0


def test_save_gpu_memory_reduces_estimate(tmp_path):
    solver_no_save = _build_solver(tmp_path, save_gpu_memory=False)
    solver_save = _build_solver(tmp_path, save_gpu_memory=True)

    with patch("fullwave.solver.solver.logger") as mock_no_save:
        solver_no_save._estimate_gpu_memory(solver_no_save.pml_builder.extended_sensor)
    gb_no_save = mock_no_save.info.call_args[0][4]

    with patch("fullwave.solver.solver.logger") as mock_save:
        solver_save._estimate_gpu_memory(solver_save.pml_builder.extended_sensor)
    gb_save = mock_save.info.call_args[0][4]

    assert gb_save <= gb_no_save


def test_save_gpu_memory_label_in_log(tmp_path):
    solver = _build_solver(tmp_path, save_gpu_memory=True)
    sensor = solver.pml_builder.extended_sensor

    with patch("fullwave.solver.solver.logger") as mock_logger:
        solver._estimate_gpu_memory(sensor)

    saving_arg = mock_logger.info.call_args[0][3]
    assert "save_gpu_memory" in saving_arg


def test_multi_gpu_two_devices_logs_per_device(tmp_path):
    solver = _build_solver(tmp_path, cuda_device_id=[0, 1])
    sensor = solver.pml_builder.extended_sensor

    with patch("fullwave.solver.solver.logger") as mock_logger:
        solver._estimate_gpu_memory(sensor)

    assert mock_logger.info.call_count == 2


def test_multi_gpu_two_devices_each_has_one_halo_side(tmp_path):
    solver = _build_solver(tmp_path, cuda_device_id=[0, 1])
    sensor = solver.pml_builder.extended_sensor

    with patch("fullwave.solver.solver.logger") as mock_logger:
        solver._estimate_gpu_memory(sensor)

    for call in mock_logger.info.call_args_list:
        halo_value = call[0][6]
        assert halo_value == 8  # 1 side * 8 ghost cells


def test_multi_gpu_three_devices_middle_has_two_halo_sides(tmp_path):
    solver = _build_solver(tmp_path, cuda_device_id=[0, 1, 2])
    sensor = solver.pml_builder.extended_sensor

    with patch("fullwave.solver.solver.logger") as mock_logger:
        solver._estimate_gpu_memory(sensor)

    assert mock_logger.info.call_count == 3
    calls = mock_logger.info.call_args_list
    assert calls[0][0][6] == 8  # first GPU: 1 halo side
    assert calls[1][0][6] == 16  # middle GPU: 2 halo sides
    assert calls[2][0][6] == 8  # last GPU: 1 halo side


def test_multi_gpu_splits_depth(tmp_path):
    solver = _build_solver(tmp_path, cuda_device_id=[0, 1])
    sensor = solver.pml_builder.extended_sensor
    total_depth = solver.pml_builder.extended_grid.nx

    with patch("fullwave.solver.solver.logger") as mock_logger:
        solver._estimate_gpu_memory(sensor)

    calls = mock_logger.info.call_args_list
    depth_gpu0 = calls[0][0][5]
    depth_gpu1 = calls[1][0][5]
    assert depth_gpu0 + depth_gpu1 == total_depth
