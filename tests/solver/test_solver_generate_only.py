from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import fullwave


def _build_simple_solver(tmp_path: Path) -> fullwave.Solver:
    """Construct a minimal 2D Solver instance for testing."""
    work_dir = tmp_path / "work_dir_generate_only"
    work_dir.mkdir(parents=True, exist_ok=True)

    # Small, fast grid
    domain_size = (1e-3, 1e-3)  # meters
    f0 = 1e6
    c0 = 1540
    duration = domain_size[0] / c0 * 2

    grid = fullwave.Grid(
        domain_size=domain_size,
        f0=f0,
        duration=duration,
        c0=c0,
    )

    sound_speed_map = c0 * np.ones((grid.nx, grid.ny))
    density_map = 1000 * np.ones((grid.nx, grid.ny))
    alpha_coeff_map = 0.5 * np.ones((grid.nx, grid.ny))
    alpha_power_map = 1.0 * np.ones((grid.nx, grid.ny))
    beta_map = np.zeros((grid.nx, grid.ny))

    medium = fullwave.Medium(
        grid=grid,
        sound_speed=sound_speed_map,
        density=density_map,
        alpha_coeff=alpha_coeff_map,
        alpha_power=alpha_power_map,
        beta=beta_map,
        use_isotropic_relaxation=True,
    )

    # Simple line source at mid-depth
    p_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    p_mask[grid.nx // 2, :] = True
    p0 = np.ones((p_mask.sum(), grid.nt))
    source = fullwave.Source(p0, p_mask)

    # Full-domain sensor
    sensor_mask = np.zeros((grid.nx, grid.ny), dtype=bool)
    sensor_mask[:, :] = True
    sensor = fullwave.Sensor(mask=sensor_mask, sampling_modulus_time=2)

    solver = fullwave.Solver(
        work_dir=work_dir,
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        use_isotropic_relaxation=True,
    )
    return solver


@pytest.fixture
def patched_solver(tmp_path, monkeypatch):
    """Return a Solver with patched binary discovery and CUDA checks."""
    import fullwave.solver.solver as solver_module
    from fullwave.solver import launcher as launcher_module

    # Create a fake binary path to satisfy _retrieve_fullwave_simulation_path.
    fullwave_binary_path = (
        Path(__file__).parent.parent.parent
        / "fullwave"
        / "solver"
        / "bins"
        / "gpu"
        / "2d"
        / "num_relax=2"
        / "fullwave2_2d_2_relax_multi_gpu_cuda129"
    )

    monkeypatch.setattr(
        solver_module,
        "_retrieve_fullwave_simulation_path",
        lambda use_gpu,  # noqa: ARG005
        is_3d,  # noqa: ARG005
        use_exponential_attenuation,  # noqa: ARG005
        use_isotropic_relaxation,  # noqa: ARG005
        n_relax_mechanisms:  # noqa: ARG005
        fullwave_binary_path,
    )
    monkeypatch.setattr(
        launcher_module.Launcher,
        "_verify_cuda_devices_exist",
        lambda x: "0",  # noqa: ARG005
    )

    solver = _build_simple_solver(tmp_path)
    return solver


def test_generate_only_returns_simulation_dir_and_skips_launcher(patched_solver, monkeypatch):
    """Solver.run(generate_input_only=True) should generate inputs and skip launching sim."""
    from fullwave.solver import launcher as launcher_module

    # Ensure Launcher.run is not called.
    with patch.object(launcher_module.Launcher, "run") as mock_launcher_run:
        sim_dir = patched_solver.run(generate_input_only=True)

    assert isinstance(sim_dir, Path)
    assert sim_dir.exists()

    # Check a minimal set of key input files.
    expected_files = [
        "icmat.dat",
        "icc.dat",
        "outc.dat",
        "c.dat",
        "K.dat",
        "rho.dat",
        "beta.dat",
    ]
    for fname in expected_files:
        assert (sim_dir / fname).exists(), f"Expected file {fname} does not exist in {sim_dir}."

    mock_launcher_run.assert_not_called()
