from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import fullwave
from fullwave.solver.solver import (
    _check_compatible_set,
    _make_cuda_arch_option,
    _make_cuda_version_option,
    _retrieve_fullwave_simulation_path,
)


def test_make_cuda_version_option_gpu_mode():
    """Test _make_cuda_version_option in GPU mode with compatible versions."""
    # Test with compatible version
    with patch("fullwave.solver.solver.retrieve_cuda_version", return_value=11.8):
        result = _make_cuda_version_option(use_gpu=True)
        assert result == ("cuda118", 11.8)

    with patch("fullwave.solver.solver.retrieve_cuda_version", return_value=12.4):
        result = _make_cuda_version_option(use_gpu=True)
        assert result == ("cuda124", 12.4)


def test_make_cuda_version_option_cpu_mode():
    """Test _make_cuda_version_option in CPU mode."""
    with patch("fullwave.solver.solver.retrieve_cuda_version", return_value=11.8):
        result = _make_cuda_version_option(use_gpu=False)
        assert result == ("cuda118", 11.8)


def test_make_cuda_version_option_no_cuda():
    """Test _make_cuda_version_option when CUDA is not available."""
    with (
        patch("fullwave.solver.solver.retrieve_cuda_version", return_value=-1),
        pytest.raises(ValueError, match="Could not retrieve CUDA version"),
    ):
        _make_cuda_version_option(use_gpu=True)


def test_make_cuda_version_option_unverified_version_warning():
    """Test _make_cuda_version_option with unverified but compatible CUDA version."""
    with (
        patch("fullwave.solver.solver.retrieve_cuda_version", return_value=11.8),
        patch("fullwave.solver.solver.logger") as mock_logger,
    ):
        result = _make_cuda_version_option(use_gpu=True)
        assert result == ("cuda118", 11.8)
        mock_logger.warning.assert_called_once()
        assert "is not in the verified versions" in str(mock_logger.warning.call_args)


def test_make_cuda_version_option_fallback_to_compatible():
    """Test _make_cuda_version_option falls back to compatible version."""
    with (
        patch("fullwave.solver.solver.retrieve_cuda_version", return_value=12.8),
        patch(
            "fullwave.solver.solver.logger",
        ) as mock_logger,
    ):
        result = _make_cuda_version_option(use_gpu=True)
        assert result == ("cuda126", 12.6)  # Should fall back to 12.6
        mock_logger.warning.assert_called()
        warning_call = str(mock_logger.warning.call_args)
        assert "Using the closest compatible version 12.6 instead" in warning_call


def test_make_cuda_version_option_out_of_range():
    """Test _make_cuda_version_option with CUDA version outside compatible ranges."""
    with (
        patch("fullwave.solver.solver.retrieve_cuda_version", return_value=13.0),
        pytest.raises(ValueError, match=r"CUDA version 13.0 is not in the compatible ranges"),
    ):
        _make_cuda_version_option(use_gpu=True)


def test_make_cuda_arch_option_gpu_mode_compatible():
    """Test _make_cuda_arch_option in GPU mode with compatible architecture."""
    with patch(
        "fullwave.solver.solver.get_cuda_architecture",
        return_value=[{"compute_capability": (8, 9)}],
    ):
        result = _make_cuda_arch_option(use_gpu=True)
        assert result == "sm_89"


def test_make_cuda_arch_option_cpu_mode():
    """Test _make_cuda_arch_option in CPU mode."""
    with patch(
        "fullwave.solver.solver.get_cuda_architecture",
        return_value=[{"compute_capability": (8, 9)}],
    ):
        result = _make_cuda_arch_option(use_gpu=False)
        assert result == "sm_89"


def test_make_cuda_arch_option_incompatible_architecture():
    """Test _make_cuda_arch_option with incompatible CUDA architecture."""
    with (
        patch(
            "fullwave.solver.solver.get_cuda_architecture",
            return_value=[{"compute_capability": (5, 0)}],
        ),
        pytest.raises(ValueError, match=r"CUDA architecture sm_50 is not compatible"),
    ):
        _make_cuda_arch_option(use_gpu=True)


def test_make_cuda_arch_option_unverified_architecture_warning():
    """Test _make_cuda_arch_option with compatible but unverified architecture."""
    with (
        patch(
            "fullwave.solver.solver.get_cuda_architecture",
            return_value=[{"compute_capability": (7, 0)}],
        ),
        patch("fullwave.solver.solver.logger") as mock_logger,
    ):
        result = _make_cuda_arch_option(use_gpu=True)
        assert result == "sm_70"
        mock_logger.warning.assert_called_once()
        assert "is not verified" in str(mock_logger.warning.call_args)


def test_make_cuda_arch_option_verified_architecture():
    """Test _make_cuda_arch_option with verified architecture (no warning)."""
    with (
        patch(
            "fullwave.solver.solver.get_cuda_architecture",
            return_value=[{"compute_capability": (8, 9)}],
        ),
        patch("fullwave.solver.solver.logger") as mock_logger,
    ):
        result = _make_cuda_arch_option(use_gpu=True)
        assert result == "sm_89"
        mock_logger.warning.assert_not_called()


def test_make_cuda_arch_option_multiple_devices():
    """Test _make_cuda_arch_option uses first device when multiple devices available."""
    mock_devices = [
        {"compute_capability": (8, 9)},
        {"compute_capability": (7, 5)},
    ]
    with patch("fullwave.solver.solver.get_cuda_architecture", return_value=mock_devices):
        result = _make_cuda_arch_option(use_gpu=True)
        assert result == "sm_89"


def test_check_compatible_set_valid_combinations():
    """Test _check_compatible_set with valid CUDA version and architecture combinations."""
    # Test some known valid combinations
    assert _check_compatible_set(11.8, "sm_89") is True
    assert _check_compatible_set(12.6, "sm_80") is True
    assert _check_compatible_set(12.9, "sm_100") is True
    assert _check_compatible_set(12.4, "sm_75") is True


def test_check_compatible_set_invalid_combinations():
    """Test _check_compatible_set with invalid CUDA version and architecture combinations."""
    # Test combinations that are not in the set
    assert _check_compatible_set(10.0, "sm_89") is False
    assert _check_compatible_set(11.8, "sm_100") is False  # sm_100 not available in 11.8
    assert _check_compatible_set(13.0, "sm_89") is False
    assert _check_compatible_set(12.6, "sm_999") is False


def test_check_compatible_set_edge_cases():
    """Test _check_compatible_set with edge cases and boundary values."""
    # Test with architectures that exist in some versions but not others
    assert _check_compatible_set(12.9, "sm_120") is True  # Only available in 12.9
    assert _check_compatible_set(12.6, "sm_120") is False  # Not available in 12.6

    # Test with older architectures
    assert _check_compatible_set(11.8, "sm_61") is True
    assert _check_compatible_set(12.9, "sm_61") is True


def test_retrieve_fullwave_simulation_path_2d_gpu():
    """Test _retrieve_fullwave_simulation_path for 2D GPU mode."""
    with (
        patch("fullwave.solver.solver._make_cuda_arch_option", return_value="sm_89"),
        patch("fullwave.solver.solver._make_cuda_version_option", return_value=("cuda126", 12.6)),
        patch("fullwave.solver.solver._check_compatible_set", return_value=True),
    ):
        result = _retrieve_fullwave_simulation_path(
            use_gpu=True,
            is_3d=False,
            use_isotropic_relaxation=False,
        )
        expected_path = (
            Path(__file__).parent.parent.parent
            / "fullwave"
            / "solver"
            / "bins"
            / "gpu"
            / "2d"
            / "num_relax=2"
            / "fullwave2_2d_2_relax_multi_gpu_sm_89_cuda126"
        )
        assert result == expected_path


def test_retrieve_fullwave_simulation_path_3d_gpu():
    """Test _retrieve_fullwave_simulation_path for 3D GPU mode."""
    with (
        patch("fullwave.solver.solver._make_cuda_arch_option", return_value="sm_80"),
        patch("fullwave.solver.solver._make_cuda_version_option", return_value=("cuda118", 11.8)),
        patch("fullwave.solver.solver._check_compatible_set", return_value=True),
    ):
        result = _retrieve_fullwave_simulation_path(
            use_gpu=True,
            is_3d=True,
            use_isotropic_relaxation=False,
        )
        expected_path = (
            Path(__file__).parent.parent.parent
            / "fullwave"
            / "solver"
            / "bins"
            / "gpu"
            / "3d"
            / "num_relax=2"
            / "fullwave2_3d_2_relax_multi_gpu_sm_80_cuda118"
        )
        assert result == expected_path


def test_retrieve_fullwave_simulation_path_2d_cpu_not_implemented():
    """Test _retrieve_fullwave_simulation_path for 2D CPU mode raises NotImplementedError."""
    with (
        patch("fullwave.solver.solver._make_cuda_arch_option", return_value="sm_89"),
        patch("fullwave.solver.solver._make_cuda_version_option", return_value=("cuda126", 12.6)),
        patch("fullwave.solver.solver._check_compatible_set", return_value=True),
        pytest.raises(
            NotImplementedError,
            match="Currently, 2D simulation is not supported in CPU mode",
        ),
    ):
        _retrieve_fullwave_simulation_path(use_gpu=False, is_3d=False)


def test_retrieve_fullwave_simulation_path_3d_cpu_not_implemented():
    """Test _retrieve_fullwave_simulation_path for 3D CPU mode raises NotImplementedError."""
    with (
        patch("fullwave.solver.solver._make_cuda_arch_option", return_value="sm_89"),
        patch("fullwave.solver.solver._make_cuda_version_option", return_value=("cuda126", 12.6)),
        patch("fullwave.solver.solver._check_compatible_set", return_value=True),
        pytest.raises(
            NotImplementedError,
            match="Currently, 3D simulation is not supported in CPU mode",
        ),
    ):
        _retrieve_fullwave_simulation_path(use_gpu=False, is_3d=True)


def test_retrieve_fullwave_simulation_path_calls_helper_functions():
    """Test _retrieve_fullwave_simulation_path calls required helper functions."""
    with (
        patch("fullwave.solver.solver._make_cuda_arch_option") as mock_arch,
        patch("fullwave.solver.solver._make_cuda_version_option") as mock_version,
        patch("fullwave.solver.solver._check_compatible_set") as mock_check,
    ):
        mock_arch.return_value = "sm_89"
        mock_version.return_value = ("cuda126", 12.6)
        mock_check.return_value = True

        _retrieve_fullwave_simulation_path(
            use_gpu=True,
            is_3d=False,
            use_isotropic_relaxation=False,
        )

        mock_arch.assert_called_once_with(use_gpu=True)
        mock_version.assert_called_once_with(use_gpu=True)
        mock_check.assert_called_once_with(cuda_version=12.6, cuda_arch="sm_89")


def test_retrieve_fullwave_simulation_path_different_arch_versions():
    """Test _retrieve_fullwave_simulation_path.

    with different architecture and version combinations
    """
    test_cases = [
        ("sm_75", "cuda124", 12.4),
        ("sm_100", "cuda129", 12.9),
        ("sm_86", "cuda118", 11.8),
    ]

    for arch, version_str, version_float in test_cases:
        with (
            patch("fullwave.solver.solver._make_cuda_arch_option", return_value=arch),
            patch(
                "fullwave.solver.solver._make_cuda_version_option",
                return_value=(version_str, version_float),
            ),
            patch("fullwave.solver.solver._check_compatible_set", return_value=True),
        ):
            result = _retrieve_fullwave_simulation_path(
                use_gpu=True,
                is_3d=True,
                use_isotropic_relaxation=False,
            )
            expected_path = (
                Path(__file__).parent.parent.parent
                / "fullwave"
                / "solver"
                / "bins"
                / "gpu"
                / "3d"
                / "num_relax=2"
                / f"fullwave2_3d_2_relax_multi_gpu_{arch}_{version_str}"
            )
            assert result == expected_path


def _get_solver(
    tmp_path,
    use_isotropic_relaxation_medium,
    use_isotropic_relaxation_solver,
):
    work_dir = tmp_path / "test_solver_use_isotropic_relaxation"
    work_dir.mkdir(parents=True, exist_ok=True)

    domain_size = (1e-2, 1e-2)  # meters
    f0 = 1e6 / 10
    c0 = 1540
    duration = domain_size[0] / c0 * 2

    grid = fullwave.Grid(
        domain_size=domain_size,
        f0=f0,
        duration=duration,
        c0=c0,
    )

    sound_speed_map = 1540 * np.ones((grid.nx, grid.ny))  # m/s
    density_map = 1000 * np.ones((grid.nx, grid.ny))  # kg/m^3
    alpha_coeff_map = 0.5 * np.ones((grid.nx, grid.ny))  # dB/(MHz^y cm)
    alpha_power_map = 1.0 * np.ones((grid.nx, grid.ny))  # power law exponent
    beta_map = 0.0 * np.ones((grid.nx, grid.ny))  # nonlinearity parameter

    medium = fullwave.Medium(
        grid=grid,
        sound_speed=sound_speed_map,
        density=density_map,
        alpha_coeff=alpha_coeff_map,
        alpha_power=alpha_power_map,
        beta=beta_map,
        use_isotropic_relaxation=use_isotropic_relaxation_medium,
    )
    p_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    p_mask[grid.nx // 2, :] = True
    p0 = np.ones((p_mask.sum(), grid.nt))  # [n_sources, nt]

    source = fullwave.Source(p0, p_mask)

    sensor_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    sensor_mask[:, :] = True

    sensor = fullwave.Sensor(mask=sensor_mask, sampling_modulus_time=7)

    fw_solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        use_isotropic_relaxation=use_isotropic_relaxation_solver,
    )
    return fw_solver


@pytest.mark.parametrize(
    ("use_isotropic_relaxation_medium", "use_isotropic_relaxation_solver", "warning_expected"),
    [
        (True, True, False),
        (False, False, False),
        (True, False, True),
        (False, True, True),
    ],
)
def test_use_isotropic_relaxation(
    tmp_path,
    use_isotropic_relaxation_medium,
    use_isotropic_relaxation_solver,
    warning_expected,
    caplog,
):
    """Test Solver with and without isotropic relaxation medium and solver."""
    with (
        patch("fullwave.solver.solver.logger") as mock_logger,
    ):
        if warning_expected:
            _ = _get_solver(
                tmp_path,
                use_isotropic_relaxation_medium=use_isotropic_relaxation_medium,
                use_isotropic_relaxation_solver=use_isotropic_relaxation_solver,
            )
            mock_logger.warning.assert_called_once()
        else:
            _ = _get_solver(
                tmp_path,
                use_isotropic_relaxation_medium=use_isotropic_relaxation_medium,
                use_isotropic_relaxation_solver=use_isotropic_relaxation_solver,
            )
            mock_logger.warning.assert_not_called()
