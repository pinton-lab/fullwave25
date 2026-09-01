from unittest.mock import patch

import numpy as np
import pytest

import fullwave
from fullwave.solver import solver as solver_module


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


def _gpu_log_calls(mock_logger):
    """Return only per-GPU log calls (skip the experimental feature message)."""
    return [
        call for call in mock_logger.info.call_args_list if "GPU memory estimate" in str(call[0][0])
    ]


@pytest.fixture(autouse=True)
def _patch_binary(monkeypatch, tmp_path_factory):
    binary = tmp_path_factory.mktemp("bins") / "fullwave2_2d_n_relax_multi_gpu"
    binary.touch()
    monkeypatch.setattr(
        solver_module,
        "_retrieve_fullwave_simulation_path",
        lambda **kwargs: binary,  # noqa: ARG005
    )


def test_single_gpu_relaxation_logs_info(tmp_path):
    solver = _build_solver(tmp_path)
    sensor = solver.pml_builder.extended_sensor

    with patch("fullwave.solver.solver.logger") as mock_logger:
        solver._estimate_gpu_memory(sensor)

    gpu_calls = _gpu_log_calls(mock_logger)
    assert len(gpu_calls) == 1
    fmt = gpu_calls[0][0][0]
    mode_arg = gpu_calls[0][0][2]
    assert "GPU memory estimate" in fmt
    assert mode_arg == "relaxation"


def test_single_gpu_no_halo(tmp_path):
    solver = _build_solver(tmp_path)
    sensor = solver.pml_builder.extended_sensor

    with patch("fullwave.solver.solver.logger") as mock_logger:
        solver._estimate_gpu_memory(sensor)

    gpu_calls = _gpu_log_calls(mock_logger)
    # halo argument: dev_id, mode, saving, GB, depth, halo, lateral
    halo_value = gpu_calls[0][0][6]
    assert halo_value == 0


def test_estimate_is_positive(tmp_path):
    solver = _build_solver(tmp_path)
    sensor = solver.pml_builder.extended_sensor

    with patch("fullwave.solver.solver.logger") as mock_logger:
        solver._estimate_gpu_memory(sensor)

    gpu_calls = _gpu_log_calls(mock_logger)
    gb_value = gpu_calls[0][0][4]
    assert gb_value > 0


def test_save_gpu_memory_reduces_estimate(tmp_path):
    solver_no_save = _build_solver(tmp_path, save_gpu_memory=False)
    solver_save = _build_solver(tmp_path, save_gpu_memory=True)

    with patch("fullwave.solver.solver.logger") as mock_no_save:
        solver_no_save._estimate_gpu_memory(solver_no_save.pml_builder.extended_sensor)
    gb_no_save = _gpu_log_calls(mock_no_save)[0][0][4]

    with patch("fullwave.solver.solver.logger") as mock_save:
        solver_save._estimate_gpu_memory(solver_save.pml_builder.extended_sensor)
    gb_save = _gpu_log_calls(mock_save)[0][0][4]

    assert gb_save <= gb_no_save


def test_save_gpu_memory_label_in_log(tmp_path):
    solver = _build_solver(tmp_path, save_gpu_memory=True)
    sensor = solver.pml_builder.extended_sensor

    with patch("fullwave.solver.solver.logger") as mock_logger:
        solver._estimate_gpu_memory(sensor)

    gpu_calls = _gpu_log_calls(mock_logger)
    saving_arg = gpu_calls[0][0][3]
    assert "save_gpu_memory" in saving_arg


def test_multi_gpu_two_devices_logs_per_device(tmp_path):
    solver = _build_solver(tmp_path, cuda_device_id=[0, 1])
    sensor = solver.pml_builder.extended_sensor

    with patch("fullwave.solver.solver.logger") as mock_logger:
        solver._estimate_gpu_memory(sensor)

    gpu_calls = _gpu_log_calls(mock_logger)
    assert len(gpu_calls) == 2


def test_multi_gpu_two_devices_each_has_one_halo_side(tmp_path):
    solver = _build_solver(tmp_path, cuda_device_id=[0, 1])
    sensor = solver.pml_builder.extended_sensor

    with patch("fullwave.solver.solver.logger") as mock_logger:
        solver._estimate_gpu_memory(sensor)

    gpu_calls = _gpu_log_calls(mock_logger)
    for call in gpu_calls:
        halo_value = call[0][6]
        assert halo_value == 8  # 1 side * 8 ghost cells


def test_multi_gpu_three_devices_middle_has_two_halo_sides(tmp_path):
    solver = _build_solver(tmp_path, cuda_device_id=[0, 1, 2])
    sensor = solver.pml_builder.extended_sensor

    with patch("fullwave.solver.solver.logger") as mock_logger:
        solver._estimate_gpu_memory(sensor)

    gpu_calls = _gpu_log_calls(mock_logger)
    assert len(gpu_calls) == 3
    assert gpu_calls[0][0][6] == 8  # first GPU: 1 halo side
    assert gpu_calls[1][0][6] == 16  # middle GPU: 2 halo sides
    assert gpu_calls[2][0][6] == 8  # last GPU: 1 halo side


def test_multi_gpu_splits_depth(tmp_path):
    solver = _build_solver(tmp_path, cuda_device_id=[0, 1])
    sensor = solver.pml_builder.extended_sensor
    total_depth = solver.pml_builder.extended_grid.nx

    with patch("fullwave.solver.solver.logger") as mock_logger:
        solver._estimate_gpu_memory(sensor)

    gpu_calls = _gpu_log_calls(mock_logger)
    depth_gpu0 = gpu_calls[0][0][5]
    depth_gpu1 = gpu_calls[1][0][5]
    assert depth_gpu0 + depth_gpu1 == total_depth


# ---------------------------------------------------------------------------
# Direct formula-verification tests for _mem_exponential / _mem_relaxation
# ---------------------------------------------------------------------------

# Shared test parameters
_SLAB = 1000
_NDLEV = 5
_NSRC = 100
_NTIC = 200
_NSEN = 50
_NRELAX = 2
_NAIR = 30
_FB = 4  # float_bytes
_IB = 4  # int_bytes


class TestMemExponentialFormula:
    """Verify _mem_exponential returns byte counts matching the C formulas."""

    def test_3d_matches_c_formula(self):
        expected = (
            4 * 2 * _SLAB * _FB  # fields (p, u, v, w) x 2 time levels
            + 4 * _SLAB * _FB  # material: rho, K, beta, a_exp
            + 9 * 2 * _NDLEV * _FB
            + _SLAB * _IB  # dmap + dcmap
            + _NSRC * _NTIC * _FB  # source icmat (no save)
            + 3 * _NSRC * _IB  # source coords (3D)
            + _NSEN * _FB  # genoutframe
            + (3 + 1) * _NSEN * _IB  # coordsout_local
            + _NSEN * _IB  # p_idx_array
        )
        result = fullwave.Solver._mem_exponential(
            slab=_SLAB,
            n_deriv_levels=_NDLEV,
            n_sources=_NSRC,
            n_source_timesteps=_NTIC,
            save_gpu_memory=False,
            n_sensors=_NSEN,
            float_bytes=_FB,
            int_bytes=_IB,
            is_3d=True,
        )
        assert result == expected

    def test_2d_matches_c_formula(self):
        expected = (
            3 * 2 * _SLAB * _FB  # fields (p, u, w) x 2 time levels
            + 4 * _SLAB * _FB  # material
            + 9 * 2 * _NDLEV * _FB
            + _SLAB * _IB  # dmap + dcmap
            + _NSRC * _NTIC * _FB  # source icmat
            + 2 * _NSRC * _IB  # source coords (2D)
            + _NSEN * _FB  # genoutframe
            + (2 + 1) * _NSEN * _IB  # coordsout_local
            + _NSEN * _IB  # p_idx_array
        )
        result = fullwave.Solver._mem_exponential(
            slab=_SLAB,
            n_deriv_levels=_NDLEV,
            n_sources=_NSRC,
            n_source_timesteps=_NTIC,
            save_gpu_memory=False,
            n_sensors=_NSEN,
            float_bytes=_FB,
            int_bytes=_IB,
            is_3d=False,
        )
        assert result == expected

    def test_save_gpu_memory(self):
        """save_gpu_memory uses single-slice icmat (n_src x fb) instead of full."""
        result = fullwave.Solver._mem_exponential(
            slab=_SLAB,
            n_deriv_levels=_NDLEV,
            n_sources=_NSRC,
            n_source_timesteps=_NTIC,
            save_gpu_memory=True,
            n_sensors=_NSEN,
            float_bytes=_FB,
            int_bytes=_IB,
            is_3d=True,
        )
        result_no_save = fullwave.Solver._mem_exponential(
            slab=_SLAB,
            n_deriv_levels=_NDLEV,
            n_sources=_NSRC,
            n_source_timesteps=_NTIC,
            save_gpu_memory=False,
            n_sensors=_NSEN,
            float_bytes=_FB,
            int_bytes=_IB,
            is_3d=True,
        )
        # Difference should be exactly the saved icmat bytes
        saved = _NSRC * _NTIC * _FB - _NSRC * _FB
        assert result_no_save - result == saved

    def test_no_sources_no_sensors(self):
        """Source and sensor terms are zero when counts are 0."""
        expected = (
            4 * 2 * _SLAB * _FB  # fields
            + 4 * _SLAB * _FB  # material
            + 9 * 2 * _NDLEV * _FB
            + _SLAB * _IB  # dmap + dcmap
        )
        result = fullwave.Solver._mem_exponential(
            slab=_SLAB,
            n_deriv_levels=_NDLEV,
            n_sources=0,
            n_source_timesteps=_NTIC,
            save_gpu_memory=False,
            n_sensors=0,
            float_bytes=_FB,
            int_bytes=_IB,
            is_3d=True,
        )
        assert result == expected


class TestMemRelaxationFormula:
    """Verify _mem_relaxation returns byte counts matching the C formulas."""

    def test_3d_matches_c_formula(self):
        expected = (
            4 * 2 * _SLAB * _FB  # fields (p, u, v, w) x 2 time levels
            + 2 * (3 * _NRELAX * 2 * _SLAB * _FB)  # psi arrays
            + 3 * _SLAB * _FB  # material: rho, K, beta
            + 2 * _SLAB * _FB  # kappa
            + 2 * (2 * _NRELAX) * _SLAB * _FB  # PML
            + 9 * 2 * _NDLEV * _FB
            + _SLAB * _IB  # dmap + dcmap
            + _NSRC * _NTIC * _FB  # source icmat (no save)
            + 3 * _NSRC * _IB  # source coords (3D)
            + 3 * _NAIR * _IB  # air coords (3D)
            + _NSEN * _FB  # genoutframe
            + (3 + 1) * _NSEN * _IB  # coordsout_local
            + _NSEN * _IB  # p_idx_array
        )
        result = fullwave.Solver._mem_relaxation(
            slab=_SLAB,
            n_deriv_levels=_NDLEV,
            n_sources=_NSRC,
            n_source_timesteps=_NTIC,
            save_gpu_memory=False,
            n_air=_NAIR,
            n_sensors=_NSEN,
            n_relax=_NRELAX,
            float_bytes=_FB,
            int_bytes=_IB,
            is_3d=True,
        )
        assert result == expected

    def test_2d_matches_c_formula(self):
        expected = (
            3 * 2 * _SLAB * _FB  # fields (p, u, w) x 2 time levels
            + 2 * (2 * _NRELAX * 2 * _SLAB * _FB)  # psi arrays (ndim=2)
            + 3 * _SLAB * _FB  # material
            + 2 * _SLAB * _FB  # kappa
            + 2 * (2 * _NRELAX) * _SLAB * _FB  # PML
            + 9 * 2 * _NDLEV * _FB
            + _SLAB * _IB  # dmap + dcmap
            + _NSRC * _NTIC * _FB  # source icmat
            + 2 * _NSRC * _IB  # source coords (2D)
            + 2 * _NAIR * _IB  # air coords (2D)
            + _NSEN * _FB  # genoutframe
            + (2 + 1) * _NSEN * _IB  # coordsout_local
            + _NSEN * _IB  # p_idx_array
        )
        result = fullwave.Solver._mem_relaxation(
            slab=_SLAB,
            n_deriv_levels=_NDLEV,
            n_sources=_NSRC,
            n_source_timesteps=_NTIC,
            save_gpu_memory=False,
            n_air=_NAIR,
            n_sensors=_NSEN,
            n_relax=_NRELAX,
            float_bytes=_FB,
            int_bytes=_IB,
            is_3d=False,
        )
        assert result == expected

    def test_no_air_no_sources_no_sensors(self):
        """Conditional terms are zero when counts are 0."""
        expected = (
            4 * 2 * _SLAB * _FB  # fields
            + 2 * (3 * _NRELAX * 2 * _SLAB * _FB)  # psi
            + 3 * _SLAB * _FB  # material
            + 2 * _SLAB * _FB  # kappa
            + 2 * (2 * _NRELAX) * _SLAB * _FB  # PML
            + 9 * 2 * _NDLEV * _FB
            + _SLAB * _IB  # dmap + dcmap
        )
        result = fullwave.Solver._mem_relaxation(
            slab=_SLAB,
            n_deriv_levels=_NDLEV,
            n_sources=0,
            n_source_timesteps=_NTIC,
            save_gpu_memory=False,
            n_air=0,
            n_sensors=0,
            n_relax=_NRELAX,
            float_bytes=_FB,
            int_bytes=_IB,
            is_3d=True,
        )
        assert result == expected

    def test_save_gpu_memory(self):
        """save_gpu_memory uses single-slice icmat (n_src x fb) instead of full."""
        result = fullwave.Solver._mem_relaxation(
            slab=_SLAB,
            n_deriv_levels=_NDLEV,
            n_sources=_NSRC,
            n_source_timesteps=_NTIC,
            save_gpu_memory=True,
            n_air=_NAIR,
            n_sensors=_NSEN,
            n_relax=_NRELAX,
            float_bytes=_FB,
            int_bytes=_IB,
            is_3d=True,
        )
        result_no_save = fullwave.Solver._mem_relaxation(
            slab=_SLAB,
            n_deriv_levels=_NDLEV,
            n_sources=_NSRC,
            n_source_timesteps=_NTIC,
            save_gpu_memory=False,
            n_air=_NAIR,
            n_sensors=_NSEN,
            n_relax=_NRELAX,
            float_bytes=_FB,
            int_bytes=_IB,
            is_3d=True,
        )
        saved = _NSRC * _NTIC * _FB - _NSRC * _FB
        assert result_no_save - result == saved
