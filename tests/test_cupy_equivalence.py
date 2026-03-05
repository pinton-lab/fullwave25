"""Tests verifying that CuPy (GPU) and NumPy (CPU) paths produce identical results.

Every test in this module is automatically skipped when CuPy is not installed
or when no CUDA device is available.
"""

import numpy as np
import pytest

import fullwave.medium as medium_module
from fullwave.medium import Medium, MediumExponentialAttenuation, MediumRelaxationMaps
from fullwave.solver.input_file_writer import InputFileWriter
from fullwave.solver.pml_builder import PMLBuilder, PMLBuilderExponentialAttenuation
from fullwave.solver.utils import initialize_relaxation_param_dict
from fullwave.utils.relaxation_parameters import RelaxationParametersGenerator

# ---------------------------------------------------------------------------
# Skip the entire module when CuPy / CUDA is unavailable
# ---------------------------------------------------------------------------
try:
    import cupy as cp

    cp.cuda.runtime.getDeviceCount()  # raises if no device
    _CUPY_AVAILABLE = True
except Exception:
    _CUPY_AVAILABLE = False

pytestmark = pytest.mark.skipif(not _CUPY_AVAILABLE, reason="CuPy or CUDA device not available")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class DummyGrid2D:
    def __init__(self, nx, ny, dt=1e-8, f0=1e6, c0=1540.0, ppw=12, cfl=0.4):
        self.nx = nx
        self.ny = ny
        self.nz = 1
        self.dx = c0 / (f0 * ppw)
        self.dy = self.dx
        self.dz = self.dx
        self.dt = dt
        self.f0 = f0
        self.c0 = c0
        self.ppw = ppw
        self.cfl = cfl
        self.duration = dt * 100
        self.omega = 2.0 * np.pi * f0
        self.is_3d = False


class DummyGrid3D:
    def __init__(self, nx, ny, nz, dt=1e-8, f0=1e6, c0=1540.0, ppw=12, cfl=0.4):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = c0 / (f0 * ppw)
        self.dy = self.dx
        self.dz = self.dx
        self.dt = dt
        self.f0 = f0
        self.c0 = c0
        self.ppw = ppw
        self.cfl = cfl
        self.duration = dt * 100
        self.omega = 2.0 * np.pi * f0
        self.is_3d = True


def _to_np(arr):
    """Convert CuPy array to numpy; no-op for numpy arrays."""
    if isinstance(arr, np.ndarray):
        return arr
    return arr.get()


def _dummy_check_functions():
    return type(
        "dummy",
        (),
        {
            "check_instance": lambda *_args: None,
            "check_path_exists": lambda *_args: None,
            "check_compatible_value": lambda *_args: None,
        },
    )()


def _get_relaxation_dict(shape, n_relaxation_mechanisms=2):
    base = initialize_relaxation_param_dict(n_relaxation_mechanisms=n_relaxation_mechanisms)
    rng = np.random.default_rng(42)
    return {key: rng.uniform(0.5, 2.0, size=shape) for key in base}


# ---------------------------------------------------------------------------
# Medium tests
# ---------------------------------------------------------------------------
class TestMediumCupyEquivalence:
    """Compare CPU vs GPU for Medium methods."""

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        monkeypatch.setattr(medium_module, "check_functions", _dummy_check_functions())

    def _make_medium_pair(self, grid_shape, grid):
        rng = np.random.default_rng(123)
        sound_speed = rng.uniform(1400, 1600, grid_shape)
        density = rng.uniform(900, 1100, grid_shape)
        alpha_coeff = rng.uniform(0.1, 1.0, grid_shape)
        alpha_power = rng.uniform(1.0, 2.0, grid_shape)
        beta = rng.uniform(0.5, 1.5, grid_shape)

        cpu = Medium(
            grid,
            sound_speed.copy(),
            density.copy(),
            alpha_coeff.copy(),
            alpha_power.copy(),
            beta.copy(),
            use_gpu=False,
        )
        gpu = Medium(
            grid,
            sound_speed.copy(),
            density.copy(),
            alpha_coeff.copy(),
            alpha_power.copy(),
            beta.copy(),
            use_gpu=True,
        )
        return cpu, gpu

    def test_db_mhz_cm_to_a_exp_2d(self):
        shape = (32, 32)
        grid = DummyGrid2D(nx=shape[0], ny=shape[1])
        cpu, gpu = self._make_medium_pair(shape, grid)

        cpu_result = cpu._db_mhz_cm_to_a_exp(cpu.alpha_coeff)
        gpu_result = gpu._db_mhz_cm_to_a_exp(gpu.alpha_coeff)

        np.testing.assert_allclose(_to_np(gpu_result), cpu_result, rtol=1e-12)

    def test_db_mhz_cm_to_a_exp_3d(self):
        shape = (16, 16, 16)
        grid = DummyGrid3D(nx=shape[0], ny=shape[1], nz=shape[2])
        cpu, gpu = self._make_medium_pair(shape, grid)

        cpu_result = cpu._db_mhz_cm_to_a_exp(cpu.alpha_coeff)
        gpu_result = gpu._db_mhz_cm_to_a_exp(gpu.alpha_coeff)

        np.testing.assert_allclose(_to_np(gpu_result), cpu_result, rtol=1e-12)


class TestMediumRelaxationMapsCupyEquivalence:
    """Compare CPU vs GPU for MediumRelaxationMaps methods."""

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        monkeypatch.setattr(medium_module, "check_functions", _dummy_check_functions())

    def _make_pair(self, grid_shape, grid):
        rng = np.random.default_rng(456)
        sound_speed = rng.uniform(1400, 1600, grid_shape)
        density = rng.uniform(900, 1100, grid_shape)
        beta = rng.uniform(0.5, 1.5, grid_shape)
        relax = _get_relaxation_dict(grid_shape)

        cpu = MediumRelaxationMaps(
            grid,
            sound_speed.copy(),
            density.copy(),
            beta.copy(),
            {k: v.copy() for k, v in relax.items()},
            use_gpu=False,
        )
        gpu = MediumRelaxationMaps(
            grid,
            sound_speed.copy(),
            density.copy(),
            beta.copy(),
            {k: v.copy() for k, v in relax.items()},
            use_gpu=True,
        )
        return cpu, gpu

    def test_relaxation_param_dict_2d(self):
        shape = (20, 20)
        grid = DummyGrid2D(nx=shape[0], ny=shape[1])
        cpu, gpu = self._make_pair(shape, grid)

        for key in cpu.relaxation_param_dict:
            np.testing.assert_allclose(
                _to_np(gpu.relaxation_param_dict[key]),
                cpu.relaxation_param_dict[key],
                rtol=1e-10,
                err_msg=f"relaxation_param_dict[{key}] mismatch",
            )

    def test_relaxation_param_dict_for_fw2_2d(self):
        shape = (20, 20)
        grid = DummyGrid2D(nx=shape[0], ny=shape[1])
        cpu, gpu = self._make_pair(shape, grid)

        for key in cpu.relaxation_param_dict_for_fw2:
            np.testing.assert_allclose(
                _to_np(gpu.relaxation_param_dict_for_fw2[key]),
                cpu.relaxation_param_dict_for_fw2[key],
                rtol=1e-10,
                err_msg=f"relaxation_param_dict_for_fw2[{key}] mismatch",
            )

    def test_calc_a_and_b(self):
        shape = (20, 20)
        grid = DummyGrid2D(nx=shape[0], ny=shape[1])
        cpu, gpu = self._make_pair(shape, grid)

        rng = np.random.default_rng(789)
        dx = rng.uniform(0.01, 0.5, shape)
        kappa = rng.uniform(0.5, 3.0, shape)
        alpha = rng.uniform(0.01, 0.5, shape)
        dt = 1e-8

        a_cpu, b_cpu = cpu._calc_a_and_b(dx, kappa, alpha, dt)
        a_gpu, b_gpu = gpu._calc_a_and_b(dx, kappa, alpha, dt)

        np.testing.assert_allclose(_to_np(a_gpu), a_cpu, rtol=1e-12)
        np.testing.assert_allclose(_to_np(b_gpu), b_cpu, rtol=1e-12)


# ---------------------------------------------------------------------------
# PMLBuilder tests
# ---------------------------------------------------------------------------
class TestPMLBuilderCupyEquivalence:
    """Compare CPU vs GPU for PMLBuilder._extend_map_for_pml and _apply_transition_and_pml."""

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        monkeypatch.setattr(medium_module, "check_functions", _dummy_check_functions())

    def _make_pml_pair(self, grid, medium, source, sensor, **kwargs):
        cpu = PMLBuilder(
            grid,
            medium,
            source,
            sensor,
            use_gpu=False,
            **kwargs,
        )
        gpu = PMLBuilder(
            grid,
            medium,
            source,
            sensor,
            use_gpu=True,
            **kwargs,
        )
        return cpu, gpu

    @pytest.fixture()
    def setup_2d(self):
        import fullwave

        grid = fullwave.Grid(
            domain_size=(0.01, 0.01),
            f0=1e6,
            duration=1e-6,
            c0=1540.0,
            ppw=12,
            cfl=0.4,
        )
        rng = np.random.default_rng(111)
        shape = (grid.nx, grid.ny)
        sound_speed = rng.uniform(1400, 1600, shape)
        density = rng.uniform(900, 1100, shape)
        alpha_coeff = rng.uniform(0.1, 1.0, shape)
        alpha_power = rng.uniform(1.0, 2.0, shape)
        beta = rng.uniform(0.5, 1.5, shape)

        medium = fullwave.Medium(
            grid=grid,
            sound_speed=sound_speed,
            density=density,
            alpha_coeff=alpha_coeff,
            alpha_power=alpha_power,
            beta=beta,
        )
        src_coords = np.array([[grid.nx // 2, y] for y in range(grid.ny)])
        source = fullwave.Source(
            p0=np.ones((src_coords.shape[0], grid.nt)),
            coords=src_coords,
            grid_shape=shape,
        )
        sen_coords = np.array([[grid.nx // 2 + 5, y] for y in range(grid.ny)])
        sensor = fullwave.Sensor(
            coords=sen_coords,
            grid_shape=shape,
        )
        return grid, medium, source, sensor

    def test_extend_map_for_pml_2d_fill_edge(self, setup_2d):
        grid, medium, source, sensor = setup_2d
        cpu, gpu = self._make_pml_pair(grid, medium, source, sensor)

        rng = np.random.default_rng(222)
        arr = rng.uniform(0.0, 100.0, (grid.nx, grid.ny))

        cpu_result = cpu._extend_map_for_pml(arr.copy(), fill_edge=True)
        gpu_result = gpu._extend_map_for_pml(arr.copy(), fill_edge=True)

        np.testing.assert_allclose(_to_np(gpu_result), cpu_result, rtol=1e-14)

    def test_extend_map_for_pml_2d_zero_fill(self, setup_2d):
        grid, medium, source, sensor = setup_2d
        cpu, gpu = self._make_pml_pair(grid, medium, source, sensor)

        rng = np.random.default_rng(333)
        arr = rng.uniform(0.0, 100.0, (grid.nx, grid.ny))

        cpu_result = cpu._extend_map_for_pml(arr.copy(), fill_edge=False)
        gpu_result = gpu._extend_map_for_pml(arr.copy(), fill_edge=False)

        np.testing.assert_allclose(_to_np(gpu_result), cpu_result, rtol=1e-14)

    def test_apply_transition_and_pml_2d(self, setup_2d):
        grid, medium, source, sensor = setup_2d
        cpu, gpu = self._make_pml_pair(grid, medium, source, sensor)

        ext_shape = cpu.extended_medium.sound_speed.shape
        rng = np.random.default_rng(444)
        arr = rng.uniform(0.1, 5.0, ext_shape)

        for axis in [0, 1]:
            for transition_type in ["smooth", "linear", "cosine", "polynomial"]:
                cpu_result = cpu._apply_transition_and_pml(
                    arr.copy(),
                    value_target=0.0,
                    array_shape=ext_shape,
                    axis=axis,
                    transition_type=transition_type,
                    is_3d=False,
                )
                gpu_result = gpu._apply_transition_and_pml(
                    arr.copy(),
                    value_target=0.0,
                    array_shape=ext_shape,
                    axis=axis,
                    transition_type=transition_type,
                    is_3d=False,
                )
                np.testing.assert_allclose(
                    _to_np(gpu_result),
                    cpu_result,
                    rtol=1e-12,
                    err_msg=f"axis={axis}, transition_type={transition_type}",
                )

    def test_calc_a_and_b(self, setup_2d):
        grid, medium, source, sensor = setup_2d
        cpu, gpu = self._make_pml_pair(grid, medium, source, sensor)

        rng = np.random.default_rng(555)
        shape = cpu.extended_medium.sound_speed.shape
        dx = rng.uniform(0.01, 0.5, shape)
        kappa = rng.uniform(0.5, 3.0, shape)
        alpha = rng.uniform(0.01, 0.5, shape)
        dt = grid.dt

        a_cpu, b_cpu = cpu._calc_a_and_b(dx, kappa, alpha, dt)
        a_gpu, b_gpu = gpu._calc_a_and_b(dx, kappa, alpha, dt)

        np.testing.assert_allclose(_to_np(a_gpu), a_cpu, rtol=1e-12)
        np.testing.assert_allclose(_to_np(b_gpu), b_cpu, rtol=1e-12)

    def test_extended_medium_identical(self, setup_2d):
        """The extended medium (after __init__) should be identical CPU vs GPU."""
        grid, medium, source, sensor = setup_2d
        cpu, gpu = self._make_pml_pair(grid, medium, source, sensor)

        np.testing.assert_allclose(
            _to_np(gpu.extended_medium.sound_speed),
            cpu.extended_medium.sound_speed,
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            _to_np(gpu.extended_medium.density),
            cpu.extended_medium.density,
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            _to_np(gpu.extended_medium.alpha_coeff),
            cpu.extended_medium.alpha_coeff,
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            _to_np(gpu.extended_medium.alpha_power),
            cpu.extended_medium.alpha_power,
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            _to_np(gpu.extended_medium.beta),
            cpu.extended_medium.beta,
            rtol=1e-14,
        )


class TestPMLBuilderExponentialAttenuationCupyEquivalence:
    """Compare CPU vs GPU for PMLBuilderExponentialAttenuation."""

    @pytest.fixture()
    def setup_2d(self):
        import fullwave

        grid = fullwave.Grid(
            domain_size=(0.01, 0.01),
            f0=1e6,
            duration=1e-6,
            c0=1540.0,
            ppw=12,
            cfl=0.4,
        )
        rng = np.random.default_rng(666)
        shape = (grid.nx, grid.ny)
        sound_speed = rng.uniform(1400, 1600, shape)
        density = rng.uniform(900, 1100, shape)
        alpha_coeff = rng.uniform(0.1, 1.0, shape)
        alpha_power = rng.uniform(1.0, 2.0, shape)
        beta = rng.uniform(0.5, 1.5, shape)

        medium = fullwave.Medium(
            grid=grid,
            sound_speed=sound_speed,
            density=density,
            alpha_coeff=alpha_coeff,
            alpha_power=alpha_power,
            beta=beta,
        )
        src_coords = np.array([[grid.nx // 2, y] for y in range(grid.ny)])
        source = fullwave.Source(
            p0=np.ones((src_coords.shape[0], grid.nt)),
            coords=src_coords,
            grid_shape=shape,
        )
        sen_coords = np.array([[grid.nx // 2 + 5, y] for y in range(grid.ny)])
        sensor = fullwave.Sensor(
            coords=sen_coords,
            grid_shape=shape,
        )
        return grid, medium, source, sensor

    def test_mask_body_2d(self, setup_2d):
        grid, medium, source, sensor = setup_2d
        cpu = PMLBuilderExponentialAttenuation(
            grid,
            medium,
            source,
            sensor,
            use_gpu=False,
        )
        gpu = PMLBuilderExponentialAttenuation(
            grid,
            medium,
            source,
            sensor,
            use_gpu=True,
        )

        nx, ny = cpu.extended_medium.sound_speed.shape[:2]
        cpu_mask = cpu._mask_body_2d(nx, ny, cpu.num_boundary_points)
        gpu_mask = gpu._mask_body_2d(nx, ny, gpu.num_boundary_points)

        np.testing.assert_allclose(_to_np(gpu_mask), cpu_mask, rtol=1e-5, atol=1e-7)

    def test_extended_medium_identical(self, setup_2d):
        grid, medium, source, sensor = setup_2d
        cpu = PMLBuilderExponentialAttenuation(
            grid,
            medium,
            source,
            sensor,
            use_gpu=False,
        )
        gpu = PMLBuilderExponentialAttenuation(
            grid,
            medium,
            source,
            sensor,
            use_gpu=True,
        )

        np.testing.assert_allclose(
            _to_np(gpu.extended_medium.sound_speed),
            cpu.extended_medium.sound_speed,
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            _to_np(gpu.extended_medium.density),
            cpu.extended_medium.density,
            rtol=1e-14,
        )

    def test_run_identical(self, setup_2d):
        grid, medium, source, sensor = setup_2d
        cpu_builder = PMLBuilderExponentialAttenuation(
            grid,
            medium,
            source,
            sensor,
            use_gpu=False,
        )
        gpu_builder = PMLBuilderExponentialAttenuation(
            grid,
            medium,
            source,
            sensor,
            use_gpu=True,
        )

        cpu_result = cpu_builder.run(use_pml=True)
        gpu_result = gpu_builder.run(use_pml=True)

        np.testing.assert_allclose(
            _to_np(gpu_result.sound_speed),
            cpu_result.sound_speed,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            _to_np(gpu_result.density),
            cpu_result.density,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            _to_np(gpu_result.alpha_exp),
            cpu_result.alpha_exp,
            rtol=1e-5,
            atol=1e-7,
        )
        np.testing.assert_allclose(
            _to_np(gpu_result.beta),
            cpu_result.beta,
            rtol=1e-12,
        )


class TestMediumExponentialAttenuationCupyEquivalence:
    """Compare CPU vs GPU for MediumExponentialAttenuation."""

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        monkeypatch.setattr(medium_module, "check_functions", _dummy_check_functions())

    def _make_pair(self, grid_shape, grid):
        rng = np.random.default_rng(999)
        sound_speed = rng.uniform(1400, 1600, grid_shape)
        density = rng.uniform(900, 1100, grid_shape)
        alpha_exp = rng.uniform(0.9, 1.0, grid_shape)
        beta = rng.uniform(0.5, 1.5, grid_shape)

        cpu = MediumExponentialAttenuation(
            grid,
            sound_speed.copy(),
            density.copy(),
            alpha_exp.copy(),
            beta.copy(),
            use_gpu=False,
        )
        gpu = MediumExponentialAttenuation(
            grid,
            sound_speed.copy(),
            density.copy(),
            alpha_exp.copy(),
            beta.copy(),
            use_gpu=True,
        )
        return cpu, gpu

    def test_bulk_modulus_2d(self):
        shape = (32, 32)
        grid = DummyGrid2D(nx=shape[0], ny=shape[1])
        cpu, gpu = self._make_pair(shape, grid)

        np.testing.assert_allclose(_to_np(gpu.bulk_modulus), cpu.bulk_modulus, rtol=1e-12)

    def test_bulk_modulus_3d(self):
        shape = (16, 16, 16)
        grid = DummyGrid3D(nx=shape[0], ny=shape[1], nz=shape[2])
        cpu, gpu = self._make_pair(shape, grid)

        np.testing.assert_allclose(_to_np(gpu.bulk_modulus), cpu.bulk_modulus, rtol=1e-12)


class TestMediumRelaxationMapsBulkModulusCupyEquivalence:
    """Compare CPU vs GPU for MediumRelaxationMaps.bulk_modulus."""

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        monkeypatch.setattr(medium_module, "check_functions", _dummy_check_functions())

    def test_bulk_modulus_2d(self):
        shape = (20, 20)
        grid = DummyGrid2D(nx=shape[0], ny=shape[1])
        rng = np.random.default_rng(456)
        sound_speed = rng.uniform(1400, 1600, shape)
        density = rng.uniform(900, 1100, shape)
        beta = rng.uniform(0.5, 1.5, shape)
        relax = _get_relaxation_dict(shape)

        cpu = MediumRelaxationMaps(
            grid,
            sound_speed.copy(),
            density.copy(),
            beta.copy(),
            {k: v.copy() for k, v in relax.items()},
            use_gpu=False,
        )
        gpu = MediumRelaxationMaps(
            grid,
            sound_speed.copy(),
            density.copy(),
            beta.copy(),
            {k: v.copy() for k, v in relax.items()},
            use_gpu=True,
        )
        np.testing.assert_allclose(_to_np(gpu.bulk_modulus), cpu.bulk_modulus, rtol=1e-12)


class TestInputFileWriterCupyEquivalence:
    """Compare CPU vs GPU for InputFileWriter._set_dc_map and dim calc."""

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        monkeypatch.setattr(medium_module, "check_functions", _dummy_check_functions())

    @pytest.fixture()
    def setup_2d(self):
        import fullwave

        grid = fullwave.Grid(
            domain_size=(1e-2, 1e-2),
            f0=1e6,
            duration=1e-5,
            c0=1540.0,
            ppw=6,
            cfl=0.4,
        )
        shape = (grid.nx, grid.ny)

        rng = np.random.default_rng(777)
        sound_speed = rng.uniform(1400, 1600, shape)
        density = rng.uniform(900, 1100, shape)
        alpha_exp = rng.uniform(0.9, 1.0, shape)
        beta = np.zeros(shape)

        medium = fullwave.MediumExponentialAttenuation(
            grid,
            sound_speed,
            density,
            alpha_exp,
            beta,
            use_gpu=False,
        )
        return grid, medium

    def test_dim_calc(self, setup_2d):
        grid, medium = setup_2d
        # CPU dim
        cpu_dim = int(
            np.rint(medium.sound_speed.max()) - np.rint(medium.sound_speed.min()),
        )
        # GPU dim
        import cupy as cp

        c_gpu = cp.asarray(medium.sound_speed)
        gpu_dim = int(cp.rint(c_gpu.max()) - cp.rint(c_gpu.min()))

        assert cpu_dim == gpu_dim

    def test_dc_map(self, setup_2d, tmp_path):
        import fullwave

        grid, medium = setup_2d

        src_coords = np.array([[grid.nx // 2, grid.ny // 2]])
        source = fullwave.Source(
            p0=np.ones((1, 10)),
            coords=src_coords,
            grid_shape=(grid.nx, grid.ny),
        )
        sensor = fullwave.Sensor(
            coords=src_coords,
            grid_shape=(grid.nx, grid.ny),
        )

        cpu_writer = InputFileWriter(
            work_dir=tmp_path / "cpu",
            grid=grid,
            medium=medium,
            source=source,
            sensor=sensor,
            validate_input=False,
            use_exponential_attenuation=True,
            use_gpu=False,
        )
        gpu_writer = InputFileWriter(
            work_dir=tmp_path / "gpu",
            grid=grid,
            medium=medium,
            source=source,
            sensor=sensor,
            validate_input=False,
            use_exponential_attenuation=True,
            use_gpu=True,
        )
        np.testing.assert_array_equal(_to_np(gpu_writer._dc_map), cpu_writer._dc_map)


class TestPMLBuilderRelaxationCupyEquivalence:
    """Compare CPU vs GPU for PMLBuilder (multiple relaxation path)."""

    @pytest.fixture(autouse=True)
    def _patch(self, monkeypatch):
        monkeypatch.setattr(medium_module, "check_functions", _dummy_check_functions())

    @pytest.fixture()
    def setup_2d(self):
        import fullwave

        grid = fullwave.Grid(
            domain_size=(1e-2, 1e-2),
            f0=1e6,
            duration=1e-5,
            c0=1540.0,
            ppw=6,
            cfl=0.4,
        )
        shape = (grid.nx, grid.ny)

        rng = np.random.default_rng(321)
        sound_speed = rng.uniform(1400, 1600, shape)
        density = rng.uniform(900, 1100, shape)
        beta = rng.uniform(0.5, 1.5, shape)
        relax = _get_relaxation_dict(shape)

        medium = fullwave.MediumRelaxationMaps(
            grid,
            sound_speed,
            density,
            beta,
            relax,
            use_gpu=False,
        )
        src_coords = np.array([[grid.nx // 2, i] for i in range(grid.ny)])
        sen_coords = np.array([[0, i] for i in range(grid.ny)])
        source = fullwave.Source(
            p0=np.ones((src_coords.shape[0], 10)),
            coords=src_coords,
            grid_shape=shape,
        )
        sensor = fullwave.Sensor(
            coords=sen_coords,
            grid_shape=shape,
        )
        return grid, medium, source, sensor

    def test_extended_medium_identical(self, setup_2d):
        grid, medium, source, sensor = setup_2d
        cpu = PMLBuilder(grid, medium, source, sensor, use_gpu=False)
        gpu = PMLBuilder(grid, medium, source, sensor, use_gpu=True)

        np.testing.assert_allclose(
            _to_np(gpu.extended_medium.sound_speed),
            cpu.extended_medium.sound_speed,
            rtol=1e-14,
        )
        np.testing.assert_allclose(
            _to_np(gpu.extended_medium.density),
            cpu.extended_medium.density,
            rtol=1e-14,
        )
        for key in cpu.extended_medium.relaxation_param_dict:
            np.testing.assert_allclose(
                _to_np(gpu.extended_medium.relaxation_param_dict[key]),
                cpu.extended_medium.relaxation_param_dict[key],
                rtol=1e-10,
                err_msg=f"extended relaxation_param_dict[{key}] mismatch",
            )

    def test_run_identical(self, setup_2d):
        grid, medium, source, sensor = setup_2d
        cpu_builder = PMLBuilder(grid, medium, source, sensor, use_gpu=False)
        gpu_builder = PMLBuilder(grid, medium, source, sensor, use_gpu=True)

        cpu_result = cpu_builder.run(use_pml=True)
        gpu_result = gpu_builder.run(use_pml=True)

        np.testing.assert_allclose(
            _to_np(gpu_result.sound_speed),
            cpu_result.sound_speed,
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            _to_np(gpu_result.density),
            cpu_result.density,
            rtol=1e-12,
        )
        for key in cpu_result.relaxation_param_dict_for_fw2:
            np.testing.assert_allclose(
                _to_np(gpu_result.relaxation_param_dict_for_fw2[key]),
                cpu_result.relaxation_param_dict_for_fw2[key],
                rtol=1e-10,
                err_msg=f"run() relaxation_param_dict_for_fw2[{key}] mismatch",
            )


class TestRelaxationParametersGeneratorCupyEquivalence:
    """Compare CPU (Numba) vs GPU (CuPy) for RelaxationParametersGenerator."""

    @pytest.fixture()
    def generator(self):
        from pathlib import Path

        db_path = (
            Path(__file__).parent.parent
            / "fullwave"
            / "solver"
            / "bins"
            / "database"
            / "relaxation_params_database_num_relax=2_20260113_0957.mat"
        )
        if not db_path.exists():
            pytest.skip(f"LUT database not found at {db_path}")
        return RelaxationParametersGenerator(n_relaxation_mechanisms=2, path_database=db_path)

    def test_generate_identical_2d(self, generator):
        """GPU generate must produce identical results to CPU generate for 2D arrays."""
        rng = np.random.default_rng(123)
        shape = (50, 60)
        alpha_coeff = rng.uniform(generator.alpha_min, generator.alpha_max, size=shape)
        alpha_power = rng.uniform(generator.power_min, generator.power_max, size=shape)

        cpu_result = generator._generate_cpu(alpha_coeff, alpha_power)
        gpu_result = generator._generate_gpu(cp.asarray(alpha_coeff), cp.asarray(alpha_power))

        for key in cpu_result:
            np.testing.assert_allclose(
                _to_np(gpu_result[key]),
                cpu_result[key],
                rtol=1e-12,
                err_msg=f"key={key} mismatch (2D)",
            )

    def test_generate_identical_3d(self, generator):
        """GPU generate must produce identical results to CPU generate for 3D arrays."""
        rng = np.random.default_rng(456)
        shape = (20, 25, 15)
        alpha_coeff = rng.uniform(generator.alpha_min, generator.alpha_max, size=shape)
        alpha_power = rng.uniform(generator.power_min, generator.power_max, size=shape)

        cpu_result = generator._generate_cpu(alpha_coeff, alpha_power)
        gpu_result = generator._generate_gpu(cp.asarray(alpha_coeff), cp.asarray(alpha_power))

        for key in cpu_result:
            np.testing.assert_allclose(
                _to_np(gpu_result[key]),
                cpu_result[key],
                rtol=1e-12,
                err_msg=f"key={key} mismatch (3D)",
            )

    def test_generate_dispatches_correctly(self, generator):
        """generate() dispatches to CPU for numpy, GPU for CuPy."""
        rng = np.random.default_rng(789)
        shape = (10, 10)
        alpha_coeff = rng.uniform(generator.alpha_min, generator.alpha_max, size=shape)
        alpha_power = rng.uniform(generator.power_min, generator.power_max, size=shape)

        cpu_result = generator.generate(alpha_coeff, alpha_power)
        gpu_result = generator.generate(cp.asarray(alpha_coeff), cp.asarray(alpha_power))

        for key in cpu_result:
            assert isinstance(cpu_result[key], np.ndarray), f"CPU result {key} should be numpy"
            np.testing.assert_allclose(
                _to_np(gpu_result[key]),
                cpu_result[key],
                rtol=1e-12,
                err_msg=f"key={key} dispatch mismatch",
            )
