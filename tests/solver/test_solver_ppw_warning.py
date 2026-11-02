import warnings
from pathlib import Path

import numpy as np
import pytest

import fullwave
from fullwave.solver import launcher as launcher_module
from fullwave.solver import solver as solver_module


@pytest.fixture(autouse=True)
def _patched_binary(tmp_path, monkeypatch):
    binary_path = tmp_path / "fullwave_binary"
    binary_path.touch()
    monkeypatch.setattr(
        solver_module,
        "_retrieve_fullwave_simulation_path",
        lambda **_: binary_path,
    )
    monkeypatch.setattr(launcher_module.Launcher, "_verify_cuda_devices_exist", lambda _: "0")


def _build_solver(tmp_path: Path, *, ppw: int, use_exponential_attenuation: bool):
    c0 = 1540
    domain_size = (1e-3, 1e-3)
    grid = fullwave.Grid(
        domain_size=domain_size,
        f0=1e6,
        duration=domain_size[0] / c0 * 2,
        c0=c0,
        ppw=ppw,
    )
    shape = (grid.nx, grid.ny)
    medium = fullwave.Medium(
        grid=grid,
        sound_speed=c0 * np.ones(shape),
        density=1000 * np.ones(shape),
        alpha_coeff=0.5 * np.ones(shape),
        alpha_power=np.ones(shape),
        beta=np.zeros(shape),
        use_isotropic_relaxation=True,
    )
    p_mask = np.zeros(shape, dtype=bool)
    p_mask[grid.nx // 2, :] = True
    source = fullwave.Source(np.ones((p_mask.sum(), grid.nt)), p_mask)
    sensor = fullwave.Sensor(mask=np.ones(shape, dtype=bool), sampling_modulus_time=2)
    return fullwave.Solver(
        work_dir=tmp_path / "work_dir",
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        use_isotropic_relaxation=True,
        use_exponential_attenuation=use_exponential_attenuation,
    )


def test_relaxation_below_validated_ppw_warns_and_still_runs(tmp_path):
    with pytest.warns(UserWarning, match=r"ppw=6 is below 8") as record:
        solver = _build_solver(tmp_path, ppw=6, use_exponential_attenuation=False)

    assert record[0].filename == __file__
    assert solver.run(generate_input_only=True) is not None


def test_relaxation_at_validated_ppw_does_not_warn(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _build_solver(tmp_path, ppw=8, use_exponential_attenuation=False)


def test_exponential_attenuation_below_validated_ppw_does_not_warn(tmp_path):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _build_solver(tmp_path, ppw=6, use_exponential_attenuation=True)
