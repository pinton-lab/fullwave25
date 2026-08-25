"""Perfectly Matched Layer (PML) setup for Fullwave."""

from __future__ import annotations

import concurrent.futures
import gc
import logging
from collections import OrderedDict
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np

import fullwave

if TYPE_CHECKING:
    from types import ModuleType

    from numpy.typing import NDArray
from fullwave.solver.utils import initialize_relaxation_param_dict
from fullwave.utils import check_functions, plot_utils

logger = logging.getLogger("__main__." + __name__)

_CUPY_AVAILABLE: bool | None = None


def _check_cupy() -> bool:
    """Return True if CuPy is importable; result is cached after the first call."""
    global _CUPY_AVAILABLE  # noqa: PLW0603
    if _CUPY_AVAILABLE is None:
        try:
            import cupy  # noqa: F401, PLC0415

            _CUPY_AVAILABLE = True
        except ImportError:
            _CUPY_AVAILABLE = False
    return _CUPY_AVAILABLE


def _get_array_module(*, use_gpu: bool) -> ModuleType:
    """Return ``cupy`` when *use_gpu* is True and CuPy is available, else ``numpy``."""
    if use_gpu and _check_cupy():
        import cupy  # noqa: PLC0415

        return cupy
    return np


def _smooth_transition_function_part(
    x: NDArray[np.float64],
    xp: ModuleType = np,
) -> NDArray[np.float64]:
    """Smooth bump helper (works with numpy or cupy)."""
    return xp.where(x > 0, xp.exp(-1 / (x + 1e-20)), 0)


def _smooth_transition_function(
    x: NDArray[np.float64],
    xp: ModuleType = np,
) -> NDArray[np.float64]:
    """Smooth transition function (works with numpy or cupy)."""
    return _smooth_transition_function_part(x, xp=xp) / (
        _smooth_transition_function_part(x, xp=xp) + _smooth_transition_function_part(1 - x, xp=xp)
    )


def _linear_transition_function(
    x: NDArray[np.float64],
    xp: ModuleType = np,  # noqa: ARG001
) -> NDArray[np.float64]:
    """Linear transition function."""
    return x


def _n_th_deg_polynomial_function(
    x: NDArray[np.float64],
    n: int = 2,
    xp: ModuleType = np,  # noqa: ARG001
) -> NDArray[np.float64]:
    """N-th degree polynomial transition function."""
    return x**n


def _cosine_transition_function(
    x: NDArray[np.float64],
    xp: ModuleType = np,
) -> NDArray[np.float64]:
    """Cosine transition function (works with numpy or cupy)."""
    return 0.5 * (1 - xp.cos(xp.pi * x))


def _obtain_relax_var_rename_dict(
    n_relaxation_mechanisms: int,
    *,
    is_3d: bool = False,
    use_isotropic_relaxation: bool = False,
) -> dict:
    if use_isotropic_relaxation:
        rename_dict = {
            "kappa_x": "kappa_x2",
            "kappa_u": "kappa_x1",
        }

        for nu in range(1, n_relaxation_mechanisms + 1):
            rename_dict[f"d_u_nu{nu}"] = f"d_x1_nu{nu}"
            rename_dict[f"d_x_nu{nu}"] = f"d_x2_nu{nu}"

            rename_dict[f"alpha_u_nu{nu}"] = f"alpha_x1_nu{nu}"
            rename_dict[f"alpha_x_nu{nu}"] = f"alpha_x2_nu{nu}"
    else:
        rename_dict = {
            "kappa_x": "kappa_x2",
            "kappa_y": "kappa_x2",
            "kappa_u": "kappa_x1",
            "kappa_w": "kappa_x1",
        }
        if is_3d:
            rename_dict.update(
                {
                    "kappa_z": "kappa_x2",
                    "kappa_v": "kappa_x1",
                },
            )
        for nu in range(1, n_relaxation_mechanisms + 1):
            rename_dict[f"d_u_nu{nu}"] = f"d_x1_nu{nu}"
            rename_dict[f"d_w_nu{nu}"] = f"d_x1_nu{nu}"
            rename_dict[f"d_x_nu{nu}"] = f"d_x2_nu{nu}"
            rename_dict[f"d_y_nu{nu}"] = f"d_x2_nu{nu}"

            rename_dict[f"alpha_u_nu{nu}"] = f"alpha_x1_nu{nu}"
            rename_dict[f"alpha_w_nu{nu}"] = f"alpha_x1_nu{nu}"
            rename_dict[f"alpha_x_nu{nu}"] = f"alpha_x2_nu{nu}"
            rename_dict[f"alpha_y_nu{nu}"] = f"alpha_x2_nu{nu}"
            if is_3d:
                rename_dict[f"d_v_nu{nu}"] = f"d_x1_nu{nu}"
                rename_dict[f"d_z_nu{nu}"] = f"d_x2_nu{nu}"
                rename_dict[f"alpha_v_nu{nu}"] = f"alpha_x1_nu{nu}"
                rename_dict[f"alpha_z_nu{nu}"] = f"alpha_x2_nu{nu}"

    return rename_dict


@dataclass
class PMLBuilder:
    """Setup for Perfectly Matched Layers (PML) in fullwave simulations."""

    medium_org: fullwave.Medium
    source_org: fullwave.Source
    sensor_org: fullwave.Sensor

    m_spatial_order: int
    n_pml_layer: int
    n_relaxation: int
    n_transition_layer: int

    extended_grid: fullwave.Grid = field(init=False)
    extended_medium: fullwave.Medium | fullwave.MediumRelaxationMaps = field(init=False)
    extended_source: fullwave.Source = field(init=False)
    extended_sensor: fullwave.Sensor = field(init=False)

    pml_mask_x: NDArray[np.float64] = field(init=False)
    pml_mask_y: NDArray[np.float64] = field(init=False)

    def __init__(  # noqa: PLR0912
        self,
        grid: fullwave.Grid,
        medium: fullwave.Medium,
        source: fullwave.Source,
        sensor: fullwave.Sensor,
        *,
        m_spatial_order: int = 8,
        n_pml_layer: int = 40,
        n_transition_layer: int = 40,
        use_isotropic_relaxation: bool = False,
        use_gpu: bool = False,
        pml_design: str = "decoupled",
        pml_alpha_entrance: float | None = None,
        pml_unstretched: bool = True,
        # pml_alpha_target: float = 1.1,
        # pml_alpha_power_target: float = 1.6,
        # pml_strength_factor: float = 2.0,
        # use_2_relax_mechanisms: bool = False,
    ) -> None:
        """Initialize the PMLSetup with the given medium, source, sensor, and PML parameters.

        Parameters
        ----------
        grid: fullwave.Grid
            The grid configuration.
        medium : fullwave.Medium)
            The medium relaxation maps.
        source : fullwave.Source
            The source configuration.
        sensor : fullwave.Sensor
            The sensor configuration.
        m_spatial_order : int, optional
            fullwave simulation's spatial order (default is 8).
            It depends on the fullwave simulation binary version.
            Fullwave simulation has 2M th order spatial accuracy and fourth order accuracy in time.
            see Pinton, G. (2021) http://arxiv.org/abs/2106.11476 for more detail.
        n_pml_layer : int, optional
            PML layer thickness (default is 40).
        n_transition_layer : int, optional
            Number of transition layers (default is 40).
        pml_unstretched : bool, optional
            Whether to carry the stretching factor of the medium to one across the
            transition layer, before the absorbing layer starts (default is True).
            The absorbing layer otherwise inherits the medium's own factor, and
            its coefficients divide by the square of it, so a fitted factor far
            from one mistunes the layer and it reflects. The interior keeps its
            own factor either way, so a medium whose factor is already one is
            unchanged.
        use_isotropic_relaxation : bool, optional
            Whether to use isotropic relaxation mechanisms for attenuation modeling
            to reduce memory usage while retaining accuracy.
            For 2D it will reduce the memory usage by approximately 15%.
            For 3D it will reduce the memory usage by approximately 25%.
            This option omits the anisotropic relaxation mechanisms to model the attenuation.
            We usually recommend using isotropic relaxation mechanisms
            unless the anisotropic attenuation is required for the simulation.
        use_gpu : bool, optional
            If True, use CuPy for GPU-accelerated PML computation (default is False).
            Requires CuPy to be installed. Falls back to CPU if CuPy is unavailable.
        pml_design : str, optional
            How the two-stage PML is built from the medium's relaxation
            parameters, ``medium_matched`` or ``decoupled``. Defaults to ``decoupled``.
            Pass ``medium_matched`` to reproduce the PML used up to and
            including version 1.2.6. Only the 2D path implements ``decoupled``;
            the 3D path raises rather than silently building the other one.
        pml_alpha_entrance : float, optional
            The relaxation frequency at the inner edge of the PML layer, used by
            ``decoupled`` only. ``None`` means twice the angular evaluation
            frequency, which is where the reflection coefficient was measured to
            be lowest. The optimum is sharp, so change it only with a
            measurement in hand.

        """
        check_functions.check_instance(
            grid,
            fullwave.Grid,
        )
        check_functions.check_instance(
            medium,
            [fullwave.Medium, fullwave.MediumRelaxationMaps],
        )
        check_functions.check_instance(
            source,
            fullwave.Source,
        )
        check_functions.check_instance(
            sensor,
            fullwave.Sensor,
        )

        self.grid_org = grid
        self.medium_org = medium
        self.source_org = source
        self.sensor_org = sensor
        self.is_3d = grid.is_3d
        self.use_isotropic_relaxation = use_isotropic_relaxation

        self.use_gpu = use_gpu
        self.xp: ModuleType = _get_array_module(use_gpu=use_gpu)
        if self.xp is not np:
            logger.info("PMLBuilder: using CuPy GPU backend")
        elif use_gpu:
            logger.warning(
                "PMLBuilder: use_gpu=True but CuPy is not available. Falling back to CPU (numpy)."
            )

        if pml_design not in ("medium_matched", "decoupled"):
            error_msg = f"pml_design must be 'medium_matched' or 'decoupled', got {pml_design!r}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if self.is_3d and pml_design != "medium_matched":
            error_msg = (
                f"pml_design {pml_design!r} is implemented for 2D only. "
                "Pass pml_design='medium_matched' for a 3D simulation."
            )
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
        self.pml_design = pml_design
        self.pml_alpha_entrance = pml_alpha_entrance
        self.pml_unstretched = pml_unstretched
        logger.info("PMLBuilder: pml_design=%s", pml_design)

        self.m_spatial_order = m_spatial_order
        self.n_pml_layer = n_pml_layer
        self.n_transition_layer = n_transition_layer

        domain_size: tuple[float, ...]
        if self.is_3d:
            domain_size = (
                (self.medium_org.sound_speed.shape[0] + 2 * self.num_boundary_points)
                * self.grid_org.dx,
                (self.medium_org.sound_speed.shape[1] + 2 * self.num_boundary_points)
                * self.grid_org.dy,
                (self.medium_org.sound_speed.shape[2] + 2 * self.num_boundary_points)
                * self.grid_org.dz,
            )
        else:
            domain_size = (
                (self.medium_org.sound_speed.shape[0] + 2 * self.num_boundary_points)
                * self.grid_org.dx,
                (self.medium_org.sound_speed.shape[1] + 2 * self.num_boundary_points)
                * self.grid_org.dy,
            )

        logger.debug("building extended grid for pml...")
        self.extended_grid = fullwave.Grid(
            domain_size=domain_size,
            f0=self.grid_org.f0,
            duration=self.grid_org.duration,
            c0=self.grid_org.c0,
            ppw=self.grid_org.ppw,
            cfl=self.grid_org.cfl,
        )
        logger.debug("building extended grid for pml...done")

        logger.debug("building extended medium for pml...")
        if isinstance(self.medium_org, fullwave.MediumRelaxationMaps):
            base_attrs = ["sound_speed", "density", "beta"]
            original_alpha_coeff = getattr(self.medium_org, "alpha_coeff", None)
            if original_alpha_coeff is not None and np.ndim(original_alpha_coeff) != 0:
                base_attrs.append("alpha_coeff")
            relax_attrs = list(self.medium_org.relaxation_param_dict.keys())

            if self.xp is not np:
                # Pass CuPy arrays directly — multi-GPU extension uses D2D copy (NVLink)
                named_arrays = [(name, getattr(self.medium_org, name)) for name in base_attrs]
                named_arrays += [
                    (key, self.medium_org.relaxation_param_dict[key]) for key in relax_attrs
                ]
                extended = self._extend_arrays_gpu(named_arrays)
                # Free original GPU arrays to reclaim memory
                self._ensure_numpy_medium_arrays(base_attrs)
                for key in relax_attrs:
                    import cupy as cp  # noqa: PLC0415

                    val = self.medium_org.relaxation_param_dict[key]
                    if not isinstance(val, np.ndarray):
                        self.medium_org.relaxation_param_dict[key] = cp.asnumpy(val)
                cp.get_default_memory_pool().free_all_blocks()
            else:
                named_arrays = [(name, getattr(self.medium_org, name)) for name in base_attrs] + [
                    (key, self.medium_org.relaxation_param_dict[key]) for key in relax_attrs
                ]
                extended = self._extend_arrays_cpu(named_arrays)

            extended_relaxation_param_dict = {key: extended[key] for key in relax_attrs}
            # Extended arrays are numpy — skip re-upload to GPU to avoid
            # wasting PCIe bandwidth. CPU numexpr handles subsequent computation.
            self.extended_medium = fullwave.MediumRelaxationMaps(
                grid=self.extended_grid,
                sound_speed=extended["sound_speed"],
                density=extended["density"],
                beta=extended["beta"],
                relaxation_param_dict=extended_relaxation_param_dict,
                alpha_coeff=(
                    original_alpha_coeff
                    if np.ndim(original_alpha_coeff) == 0
                    else extended.get("alpha_coeff")
                ),
                lossless_coords=(
                    None
                    if getattr(self.medium_org, "lossless_coords", None) is None
                    else self.medium_org.lossless_coords + self.num_boundary_points
                ),
                air_coords=self.medium_org.air_coords + self.num_boundary_points,
                n_relaxation_mechanisms=self.medium_org.n_relaxation_mechanisms,
                n_jobs=self.medium_org.n_jobs,
                dtype=getattr(self.medium_org, "dtype", np.float64),
                use_gpu=False,
            )
        else:
            attr_names = ["sound_speed", "density", "beta", "alpha_coeff", "alpha_power"]
            if self.xp is not np:
                named_arrays = [(name, getattr(self.medium_org, name)) for name in attr_names]
                extended = self._extend_arrays_gpu(named_arrays)
                self._ensure_numpy_medium_arrays(attr_names)
            else:
                named_arrays = [(name, getattr(self.medium_org, name)) for name in attr_names]
                extended = self._extend_arrays_cpu(named_arrays)

            # Extended arrays are numpy — skip re-upload to GPU to avoid
            # wasting PCIe bandwidth. CPU numexpr handles subsequent computation.
            self.extended_medium = fullwave.Medium(
                grid=self.extended_grid,
                sound_speed=extended["sound_speed"],
                density=extended["density"],
                beta=extended["beta"],
                alpha_coeff=extended["alpha_coeff"],
                alpha_power=extended["alpha_power"],
                air_coords=self.medium_org.air_coords + self.num_boundary_points,
                n_relaxation_mechanisms=self.medium_org.n_relaxation_mechanisms,
                path_relaxation_parameters_database=self.medium_org.path_relaxation_parameters_database,
                attenuation_builder=self.medium_org.attenuation_builder,
                n_jobs=self.medium_org.n_jobs,
                dtype=getattr(self.medium_org, "dtype", np.float64),
                use_gpu=False,
            )
        logger.debug("building extended medium for pml...done")

        logger.debug("building extended source for pml...")
        extended_grid_shape = tuple(
            s + 2 * self.num_boundary_points for s in self.source_org.grid_shape
        )
        incoords_add_ext = (
            self.source_org.incoords_add + self.num_boundary_points
            if getattr(self.source_org, "incoords_add", None) is not None
            else None
        )
        incoords_u_ext = (
            self.source_org.incoords_u + self.num_boundary_points
            if getattr(self.source_org, "incoords_u", None) is not None
            else None
        )
        incoords_v_ext = (
            self.source_org.incoords_v + self.num_boundary_points
            if getattr(self.source_org, "incoords_v", None) is not None
            else None
        )
        incoords_w_ext = (
            self.source_org.incoords_w + self.num_boundary_points
            if getattr(self.source_org, "incoords_w", None) is not None
            else None
        )
        self.extended_source = fullwave.Source(
            p0=self.source_org.p0,
            coords=self.source_org.incoords + self.num_boundary_points,
            grid_shape=extended_grid_shape,
            p0_additive=self.source_org.p0_additive,
            coords_additive=incoords_add_ext,
            u0=getattr(self.source_org, "u0", None),
            coords_u=incoords_u_ext,
            v0=getattr(self.source_org, "v0", None),
            coords_v=incoords_v_ext,
            w0=getattr(self.source_org, "w0", None),
            coords_w=incoords_w_ext,
        )
        logger.debug("building extended source for pml...done")

        logger.debug("building extended sensor for pml...")
        if self.sensor_org.is_sparse_grid:
            # Sparse-grid sensor: no explicit coordinates to shift.
            # Pass mod values through; the binary generates positions at run time.
            self.extended_sensor = fullwave.Sensor(
                mod_x=self.sensor_org.mod_x,
                mod_y=self.sensor_org.mod_y,
                mod_z=self.sensor_org.mod_z,
                sampling_modulus_time=self.sensor_org.sampling_modulus_time,
            )
        else:
            extended_sensor_grid_shape = tuple(
                s + 2 * self.num_boundary_points for s in self.sensor_org.grid_shape
            )
            self.extended_sensor = fullwave.Sensor(
                coords=self.sensor_org.outcoords + self.num_boundary_points,
                grid_shape=extended_sensor_grid_shape,
                sampling_modulus_time=self.sensor_org.sampling_modulus_time,
            )
        logger.debug("building extended sensor for pml...done")
        if self.is_3d:
            self.pml_mask_x, self.pml_mask_y, self.pml_mask_z = self._localize_pml_region()
        else:
            self.pml_mask_x, self.pml_mask_y = self._localize_pml_region()

        self.pml_layer_m = self.extended_grid.dx * self.n_pml_layer
        self.transition_layer_m = self.extended_grid.dx * self.n_transition_layer

        self.n_polynomial = 2
        self.theoritical_reflection_coefficient = 10 ** (-30)

        if self.n_pml_layer == 0:
            self.n_transition_layer = 0

    # ---
    @cached_property
    def num_boundary_points(self) -> int:
        """Returns the number of the boundary points.

        Number of PML layer and ghost cells.
        """
        return self.n_transition_layer + self.n_pml_layer + self.m_spatial_order

    @cached_property
    def nx(self) -> int:
        """Returns the number of grid points in x-direction."""
        return self.extended_grid.nx

    @cached_property
    def ny(self) -> int:
        """Returns the number of grid points in y-direction."""
        return self.extended_grid.ny

    @cached_property
    def nz(self) -> int:
        """Returns the number of grid points in y-direction."""
        return self.extended_grid.nz

    @cached_property
    def nt(self) -> int:
        """Returns the number of time steps."""
        return self.extended_grid.nt

    @cached_property
    def n_sources(self) -> int:
        """Return the number of sources."""
        return self.extended_source.n_sources

    @cached_property
    def n_sensors(self) -> int:
        """Return the number of sources."""
        return self.extended_sensor.n_sensors

    @cached_property
    def n_air(self) -> int:
        """Return the number of air coordinates."""
        return self.extended_medium.n_air

    @cached_property
    def n_coords_zero(self) -> int:
        """Return the number of air coordinates.

        (alias for self.n_air)
        """
        return self.n_air

    # def _extend_map_for_pml(
    #     self,
    #     input_map: NDArray[np.float64 | np.int64 | np.bool],
    #     *,
    #     fill_edge: bool = True,
    # ) -> NDArray[np.float64 | np.int64 | np.bool]:
    #     kwargs = {} if fill_edge else {"constant_values": 0}
    #     return np.pad(
    #         input_map,
    #         pad_width=self.num_boundary_points,
    #         mode="edge" if fill_edge else "constant",
    #         **kwargs,
    #     )

    def _extend_map_for_pml(
        self,
        input_map: NDArray[np.float64 | np.int64 | np.bool_],
        *,
        fill_edge: bool = True,
    ) -> NDArray[np.float64 | np.int64 | np.bool_]:
        """Fast version using pre-allocation and direct assignment instead of np.pad.

        When ``self.use_gpu`` is True and CuPy is available, the computation
        runs on the GPU and the result is returned as a numpy array.
        """
        xp = self.xp
        pad = self.num_boundary_points

        # Ensure array is on the correct device (no-op if already there)
        input_gpu = xp.asarray(input_map)

        # Pre-allocate output array with correct dtype
        if self.is_3d:
            nx, ny, nz = input_gpu.shape
            output = xp.empty((nx + 2 * pad, ny + 2 * pad, nz + 2 * pad), dtype=input_gpu.dtype)

            # Fill center with original data (single copy)
            output[pad : pad + nx, pad : pad + ny, pad : pad + nz] = input_gpu

            if fill_edge:
                # Fill edges efficiently using broadcasting
                # X boundaries
                output[:pad, pad : pad + ny, pad : pad + nz] = input_gpu[0:1, :, :]
                output[pad + nx :, pad : pad + ny, pad : pad + nz] = input_gpu[-1:, :, :]

                # Y boundaries (now includes X corners)
                output[:, :pad, pad : pad + nz] = output[:, pad : pad + 1, pad : pad + nz]
                output[:, pad + ny :, pad : pad + nz] = output[
                    :,
                    pad + ny - 1 : pad + ny,
                    pad : pad + nz,
                ]

                # Z boundaries (now includes all corners)
                output[:, :, :pad] = output[:, :, pad : pad + 1]
                output[:, :, pad + nz :] = output[:, :, pad + nz - 1 : pad + nz]
            else:
                # Fill with zeros
                output[:pad, :, :] = 0
                output[pad + nx :, :, :] = 0
                output[:, :pad, :] = 0
                output[:, pad + ny :, :] = 0
                output[:, :, :pad] = 0
                output[:, :, pad + nz :] = 0
        else:  # 2D case
            nx, ny = input_gpu.shape
            output = xp.empty((nx + 2 * pad, ny + 2 * pad), dtype=input_gpu.dtype)

            # Fill center
            output[pad : pad + nx, pad : pad + ny] = input_gpu

            if fill_edge:
                # Fill edges
                output[:pad, pad : pad + ny] = input_gpu[0:1, :]
                output[pad + nx :, pad : pad + ny] = input_gpu[-1:, :]
                output[:, :pad] = output[:, pad : pad + 1]
                output[:, pad + ny :] = output[:, pad + ny - 1 : pad + ny]
            else:
                output[:pad, :] = 0
                output[pad + nx :, :] = 0
                output[:, :pad] = 0
                output[:, pad + ny :] = 0

        return output

    def _move_named_arrays_to_cpu(
        self,
        named_arrays: list[tuple[str, NDArray]],
        attr_names: list[str] | None = None,
    ) -> list[tuple[str, NDArray]]:
        """Convert CuPy arrays to numpy and free GPU memory pools.

        Also replaces the corresponding ``medium_org`` attributes (given by
        *attr_names*) so the original GPU arrays can be garbage-collected.
        """
        import cupy as cp  # noqa: PLC0415

        numpy_arrays = []
        for name, arr in named_arrays:
            if hasattr(arr, "get"):
                arr_np = arr.get()
                numpy_arrays.append((name, arr_np))
                # Update medium_org so the CuPy array can be freed
                if attr_names and name in attr_names:
                    setattr(self.medium_org, name, arr_np)
            else:
                numpy_arrays.append((name, arr))
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        return numpy_arrays

    def _extend_arrays_gpu(
        self,
        named_arrays: list[tuple[str, NDArray]],
    ) -> dict[str, NDArray]:
        """Extend medium arrays using multi-GPU, single-GPU, or CPU fallback.

        Parameters
        ----------
        named_arrays : list of (name, numpy_array) or (name, cupy_array) pairs
            Arrays to extend.

        Returns
        -------
        dict[str, NDArray]
            Extended arrays as numpy arrays.

        """
        import cupy as cp  # noqa: PLC0415

        n_gpus = cp.cuda.runtime.getDeviceCount()
        logger.info("CUDA devices available: %d", n_gpus)

        attr_names = [name for name, _ in named_arrays]

        # Strategy 1: Multi-GPU parallel
        if n_gpus > 1:
            n_workers = min(n_gpus, len(named_arrays))
            try:
                logger.info(
                    "Extending %d medium arrays using %d GPUs (of %d available).",
                    len(named_arrays),
                    n_workers,
                    n_gpus,
                )
                return self._extend_arrays_multi_gpu(named_arrays, n_gpus)
            except Exception:
                logger.warning(
                    "Multi-GPU extension failed. Falling back to CPU.",
                    exc_info=True,
                )

        # Move arrays to CPU and free GPU memory before CPU fallback.
        # The original medium arrays may still occupy GPU memory.
        named_arrays = self._move_named_arrays_to_cpu(named_arrays, attr_names)

        logger.info("Extending %d medium arrays on CPU.", len(named_arrays))
        return self._extend_arrays_cpu(named_arrays)

    def _extend_arrays_multi_gpu(
        self,
        named_arrays: list[tuple[str, NDArray]],
        n_gpus: int,
    ) -> dict[str, NDArray]:
        """Extend arrays in parallel, each on a different GPU.

        Each thread sets its own CUDA device, transfers data, extends,
        and returns the result as a numpy array.
        Accepts both numpy and CuPy input arrays. CuPy arrays are copied
        device-to-device via NVLink when available (faster than PCIe).
        """
        import cupy as cp  # noqa: PLC0415

        def extend_on_device(
            args: tuple[str, NDArray, int],
        ) -> tuple[str, NDArray]:
            name, arr, device_id = args
            with cp.cuda.Device(device_id):
                # cp.asarray handles both numpy (H2D) and CuPy from
                # another device (D2D via NVLink when available)
                arr_local = cp.asarray(arr)
                result_gpu = self._extend_map_for_pml(arr_local)
                result_np = cp.asnumpy(result_gpu)
                del arr_local, result_gpu
                cp.get_default_memory_pool().free_all_blocks()
                return name, result_np

        items = [(name, arr, i % n_gpus) for i, (name, arr) in enumerate(named_arrays)]

        results = {}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(n_gpus, len(items))) as executor:
            futures = [executor.submit(extend_on_device, item) for item in items]
            for future in concurrent.futures.as_completed(futures):
                name, result = future.result()
                results[name] = result
        return results

    def _extend_arrays_sequential_gpu(
        self,
        named_arrays: list[tuple[str, NDArray]],
    ) -> dict[str, NDArray]:
        """Extend arrays one at a time on GPU, freeing after each."""
        import cupy as cp  # noqa: PLC0415

        results = {}
        pool = cp.get_default_memory_pool()
        for name, arr_np in named_arrays:
            result_gpu = self._extend_map_for_pml(arr_np)
            results[name] = cp.asnumpy(result_gpu)
            del result_gpu
            pool.free_all_blocks()
        return results

    def _extend_arrays_cpu(
        self,
        named_arrays: list[tuple[str, NDArray]],
    ) -> dict[str, NDArray]:
        """Extend arrays on CPU using ThreadPoolExecutor."""
        # Convert any CuPy arrays to numpy before CPU fallback
        numpy_arrays = []
        for name, arr in named_arrays:
            if hasattr(arr, "get"):
                numpy_arrays.append((name, arr.get()))
            else:
                numpy_arrays.append((name, arr))

        orig_xp = self.xp
        self.xp = np
        try:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = {
                    name: executor.submit(self._extend_map_for_pml, arr)
                    for name, arr in numpy_arrays
                }
                return {name: future.result() for name, future in futures.items()}
        finally:
            self.xp = orig_xp

    def _ensure_numpy_medium_arrays(
        self,
        attr_names: list[str],
    ) -> list[tuple[str, NDArray]]:
        """Convert medium arrays to numpy and free GPU memory.

        Returns list of (name, numpy_array) pairs.
        """
        import cupy as cp  # noqa: PLC0415

        named_arrays = []
        for name in attr_names:
            arr = getattr(self.medium_org, name)
            if not isinstance(arr, np.ndarray):
                arr_np = cp.asnumpy(arr)
                setattr(self.medium_org, name, arr_np)
            else:
                arr_np = arr
            named_arrays.append((name, arr_np))
        cp.get_default_memory_pool().free_all_blocks()
        return named_arrays

    def _localize_pml_region(self) -> tuple[NDArray[np.float64], ...]:
        if self.is_3d:
            n_x_extended, n_y_extended, n_z_extended = self.extended_medium.sound_speed.shape

            # Store 1D profiles instead of full 3D arrays to save memory.
            # Each mask is separable: pml_mask_x[i,j,k] only depends on i.
            pml_mask_x = np.zeros(n_x_extended, dtype=np.float64)
            pml_mask_y = np.zeros(n_y_extended, dtype=np.float64)
            pml_mask_z = np.zeros(n_z_extended, dtype=np.float64)

            # PML indices and values
            i = np.arange(self.n_pml_layer, dtype=np.int64)
            vals = i.astype(np.float64) / self.n_pml_layer

            ix1 = i + (n_x_extended - self.m_spatial_order - self.n_pml_layer)
            ix2 = self.m_spatial_order + self.n_pml_layer - i - 1

            iy1 = i + (n_y_extended - self.m_spatial_order - self.n_pml_layer)
            iy2 = self.m_spatial_order + self.n_pml_layer - i - 1

            iz1 = i + (n_z_extended - self.m_spatial_order - self.n_pml_layer)
            iz2 = self.m_spatial_order + self.n_pml_layer - i - 1

            # Fill PML ramps
            pml_mask_x[ix1] = vals
            pml_mask_x[ix2] = vals

            pml_mask_y[iy1] = vals
            pml_mask_y[iy2] = vals

            pml_mask_z[iz1] = vals
            pml_mask_z[iz2] = vals

            # Inner "hard" PML region
            pml_mask_x[0 : self.m_spatial_order] = 1.0
            pml_mask_x[n_x_extended - self.m_spatial_order : n_x_extended] = 1.0

            pml_mask_y[0 : self.m_spatial_order] = 1.0
            pml_mask_y[n_y_extended - self.m_spatial_order : n_y_extended] = 1.0

            pml_mask_z[0 : self.m_spatial_order] = 1.0
            pml_mask_z[n_z_extended - self.m_spatial_order : n_z_extended] = 1.0

            return pml_mask_x, pml_mask_y, pml_mask_z

        # 2D case — store 1D profiles instead of full 2D arrays
        n_x_extended, n_y_extended = self.extended_medium.sound_speed.shape

        pml_mask_x = np.zeros(n_x_extended, dtype=np.float64)
        pml_mask_y = np.zeros(n_y_extended, dtype=np.float64)

        i = np.arange(self.n_pml_layer, dtype=np.int64)
        vals = i.astype(np.float64) / self.n_pml_layer

        ix1 = i + (n_x_extended - self.m_spatial_order - self.n_pml_layer)
        ix2 = self.m_spatial_order + self.n_pml_layer - i - 1

        iy1 = i + (n_y_extended - self.m_spatial_order - self.n_pml_layer)
        iy2 = self.m_spatial_order + self.n_pml_layer - i - 1

        pml_mask_x[ix1] = vals
        pml_mask_x[ix2] = vals

        pml_mask_y[iy1] = vals
        pml_mask_y[iy2] = vals

        pml_mask_x[0 : self.m_spatial_order] = 1.0
        pml_mask_x[n_x_extended - self.m_spatial_order : n_x_extended] = 1.0

        pml_mask_y[0 : self.m_spatial_order] = 1.0
        pml_mask_y[n_y_extended - self.m_spatial_order : n_y_extended] = 1.0

        return pml_mask_x, pml_mask_y

    def _calc_a_and_b(
        self,
        d_x: float | NDArray[np.float64],
        kappa_x: float | NDArray[np.float64],
        alpha_x: float | NDArray[np.float64],
        dt: float | NDArray[np.float64],
        output_dtype: np.dtype | None = None,
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        xp = self.xp

        d_x = xp.asarray(d_x, dtype=xp.float64)
        kappa_x = xp.asarray(kappa_x, dtype=xp.float64)
        alpha_x = xp.asarray(alpha_x, dtype=xp.float64)
        dt = xp.asarray(dt, dtype=xp.float64)

        eps = xp.finfo(xp.float64).eps

        rate = d_x / kappa_x + alpha_x
        two_over_dt = 2.0 / dt
        b = (two_over_dt - rate) / (two_over_dt + rate)
        a = -(d_x / (kappa_x**2 + eps)) / (rate + two_over_dt)

        if output_dtype is not None and output_dtype != xp.float64:
            a = a.astype(output_dtype, copy=False)
            b = b.astype(output_dtype, copy=False)

        return a, b

    def run(self, *, use_pml: bool = True) -> fullwave.MediumRelaxationMaps:
        """Generate perfect matched layer (PML) relaxation parameters.

        It generates the relaxation parameters
        for the PML region considering the given medium and PML parameters.

        Returns
        -------
        Medium
            A Medium instance with the constructed domain properties.

        """
        logger.debug("Running PML builder...")
        if use_pml:
            extended_medium: fullwave.MediumRelaxationMaps = self.extended_medium.build()
            if self.is_3d:
                return self._apply_pml_3d(
                    extended_medium=extended_medium,
                    theoritical_reflection_coefficient=self.theoritical_reflection_coefficient,
                    n_polynomial=self.n_polynomial,
                )

            return self._apply_pml_2d(
                extended_medium=extended_medium,
                theoritical_reflection_coefficient=self.theoritical_reflection_coefficient,
                n_polynomial=self.n_polynomial,
            )

        extended_medium: fullwave.MediumRelaxationMaps = self.extended_medium.build()
        return extended_medium

    def _medium_copy(self, relaxation_param_dict: dict, key: str) -> NDArray[np.float64]:
        """Return a fresh float64 copy of one medium map."""
        return self.xp.array(relaxation_param_dict[key], dtype=self.xp.float64, copy=True)

    def _ramp_on_every_axis(
        self, field: NDArray[np.float64], **kwargs: object
    ) -> NDArray[np.float64]:
        """Apply one ramp along both grid axes in turn."""
        out = field
        for axis_index in (0, 1):
            out = self._apply_transition_and_pml(
                out, array_shape=field.shape, axis=axis_index, is_3d=False, **kwargs
            )
        return out

    def _empty_damping_in_transition_layer(
        self, damping: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Return the damping strength taken to zero across the transition layer."""
        return self._ramp_on_every_axis(
            damping,
            value_target=0.0,
            transition_type="cosine",
            transit_within_transition_layer=True,
        )

    def _raise_damping_in_pml_layer(
        self,
        damping: NDArray[np.float64],
        n_polynomial: float,
        d_target_pml: float,
    ) -> NDArray[np.float64]:
        """Return the damping strength raised to the PML target across the PML layer."""
        return self._ramp_on_every_axis(
            damping,
            value_target=d_target_pml,
            n_polynomial=n_polynomial,
            transition_type="polynomial",
            transit_within_pml_layer=True,
        )

    def _splice_entrance_frequency_into_pml_layer(
        self,
        relaxation_frequency: NDArray[np.float64],
        entrance: float,
    ) -> NDArray[np.float64]:
        """Return the relaxation frequency replaced by the CFS ramp inside the PML layer.

        Replaced rather than ramped into, because ramping a relaxation frequency
        through the evaluation frequency is what confines the damping to the
        outer edge.
        """
        xp = self.xp
        edge = self.m_spatial_order + self.n_pml_layer
        ramp = self._ramp_on_every_axis(
            xp.full_like(relaxation_frequency, entrance),
            value_target=0.0,
            transition_type="linear",
            transit_within_pml_layer=True,
        )
        inside_pml_layer = xp.zeros(relaxation_frequency.shape, dtype=bool)
        for axis_index in (0, 1):
            view = xp.moveaxis(inside_pml_layer, axis_index, 0)
            view[:edge] = True
            view[view.shape[0] - edge :] = True
        return xp.where(inside_pml_layer, xp.asarray(ramp), xp.asarray(relaxation_frequency))

    def _build_decoupled_2d(
        self,
        extended_medium: fullwave.MediumRelaxationMaps,
        rename_dict: dict,
        n_polynomial: float,
        d_target_pml: float,
    ) -> dict:
        """Return every PML array for the `decoupled` design, 2D only.

        Each interior mechanism is emptied inside the transition layer, then a
        complex frequency shifted profile is built inside the PML layer alone,
        so the result does not depend on the tissue.
        """
        entrance = (
            4.0 * np.pi * self.extended_grid.f0
            if self.pml_alpha_entrance is None
            else self.pml_alpha_entrance
        )
        relaxation_param_dict = extended_medium.relaxation_param_dict
        letters = sorted(key.split("_", 1)[1] for key in rename_dict if key.startswith("kappa_"))
        out_dict: dict = {}
        for letter in letters:
            out_dict[f"kappa_{letter}"] = relaxation_param_dict[rename_dict[f"kappa_{letter}"]]
            for nu in range(1, extended_medium.n_relaxation_mechanisms + 1):
                damping = self._medium_copy(
                    relaxation_param_dict, rename_dict[f"d_{letter}_nu{nu}"]
                )
                relaxation_frequency = self._medium_copy(
                    relaxation_param_dict, rename_dict[f"alpha_{letter}_nu{nu}"]
                )
                emptied = self._empty_damping_in_transition_layer(damping)
                if nu > 1:
                    out_dict[f"d_{letter}_nu{nu}"] = emptied
                    out_dict[f"alpha_{letter}_nu{nu}"] = relaxation_frequency
                    continue
                out_dict[f"d_{letter}_nu{nu}"] = self._raise_damping_in_pml_layer(
                    emptied, n_polynomial, d_target_pml
                )
                out_dict[f"alpha_{letter}_nu{nu}"] = self._splice_entrance_frequency_into_pml_layer(
                    relaxation_frequency, entrance
                )
        return out_dict

    def _interior_mask(self, shape: tuple[int, ...]) -> NDArray[np.bool_]:
        """Grid mask excluding the transition layer, the PML layer and the ghost cells."""
        edge = self.num_boundary_points
        mask = self.xp.zeros(shape, dtype=bool)
        mask[tuple(slice(edge, size - edge) for size in shape)] = True
        return mask

    @staticmethod
    def _lossless_value(parameter_name: str) -> float:
        """Return what a lossless voxel holds for one relaxation parameter.

        A stretching factor of 1 leaves the sound speed alone. A damping or a
        relaxation frequency of 0 removes the mechanism. Together these are the
        values of a medium whose wavenumber is exactly omega / c.
        """
        return 1.0 if parameter_name.startswith("kappa") else 0.0

    def _interior_mask(self, shape: tuple[int, ...]) -> NDArray[np.bool_]:
        """Grid mask excluding the transition layer, the PML layer and the ghost cells."""
        edge = self.num_boundary_points
        mask = self.xp.zeros(shape, dtype=bool)
        mask[tuple(slice(edge, size - edge) for size in shape)] = True
        return mask

    def _replicate_interior_outward(self, mask: NDArray[np.bool_]) -> NDArray[np.bool_]:
        """Extend an interior mask into the boundary region, as the medium is extended.

        A voxel of the boundary region takes the value of the interior voxel it
        was copied from, which is its own index clamped into the interior.
        """
        edge = self.num_boundary_points
        clamped = [np.clip(np.arange(size), edge, size - edge - 1) for size in mask.shape]
        return mask[np.ix_(*clamped)]

    def _lossless_mask(self, shape: tuple[int, ...]) -> NDArray[np.bool_] | None:
        """Return the voxels the medium asks to be lossless, or None if there are none.

        The boundary region is included. It inherits from the interior, so a
        lossless region reaching the edge of the domain carries the absorbing
        layer beside it, and the PML ramp is then built on lossless values.
        """
        coords = getattr(self.extended_medium, "lossless_coords", None)
        if coords is not None and len(coords):
            marked = self.xp.zeros(shape, dtype=bool)
            marked[tuple(np.asarray(coords).T)] = True
            return self._replicate_interior_outward(marked & self._interior_mask(shape))

        attenuation_coefficient = getattr(self.extended_medium, "alpha_coeff", None)
        if attenuation_coefficient is None:
            return None
        if np.ndim(attenuation_coefficient) == 0:
            if float(attenuation_coefficient) != 0:
                return None
            return self.xp.ones(shape, dtype=bool)
        return self.xp.asarray(attenuation_coefficient) == 0

    def _apply_zero_attenuation_gate(self, relaxation_parameters: dict) -> int:
        """Make the voxels of zero attenuation lossless and non-dispersive.

        **Called BEFORE the PML coefficients are built**, on the medium's own
        relaxation parameters. The absorbing layer then inherits the lossless
        values, and the PML damping ramp is built on top of them.

        Writing into the finished coefficients instead would delete that ramp.
        Measured on 2026-08-17, a uniform lossless medium reflected -0.02 dB
        when that was tried, against -55.60 dB here.
        """
        if not relaxation_parameters:
            return 0
        shape = next(iter(relaxation_parameters.values())).shape
        lossless = self._lossless_mask(shape)
        if lossless is None:
            return 0
        count = int(lossless.sum())
        if count == 0:
            return 0
        for name, values in relaxation_parameters.items():
            values[lossless] = self._lossless_value(name)
        logger.info("zero-attenuation gate: %d voxels set lossless", count)
        return count

    def _carry_stretching_to_one(
        self,
        relaxation_param_dict: dict[str, NDArray[np.float64]],
        *,
        is_3d: bool = False,
    ) -> None:
        """Carry every stretching factor map to one across the transition layer.

        The absorbing layer builds its coefficients from the medium's stretching
        factor, and the drive divides by the square of it. A fitted factor of
        0.30 therefore multiplies that drive by 11 and the layer reflects.

        The interior keeps its own factor. The factor reaches one before the
        absorbing layer starts, so the layer is tuned to the speed the wave
        actually arrives at.

        ONLY `kappa_x1` MOVES. The impedance constraint holds `kappa_x2` at
        exactly one, and a build that carries that constraint refuses a value
        that is one part in a million away from it. A ramp toward one is not
        exact at every cell, so `kappa_x2` is left alone.

        Parameters
        ----------
        relaxation_param_dict : dict
            The medium's relaxation maps, changed in place. The keys are the
            python side names `kappa_x1` and `kappa_x2`, which the renaming to
            the fullwave 2 names happens after.
        is_3d : bool, optional
            Whether the domain is three dimensional (default is False).
        """
        for key in ("kappa_x1",):
            if key not in relaxation_param_dict:
                continue
            found = np.asarray(relaxation_param_dict[key], dtype=np.float64)
            if found.ndim == 0 or np.allclose(found, 1.0):
                continue
            for axis in range(3 if is_3d else 2):
                found = self._apply_transition_and_pml(
                    found,
                    value_target=1.0,
                    array_shape=found.shape,
                    axis=axis,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=is_3d,
                )
            relaxation_param_dict[key] = found

    def _apply_pml_2d(
        self,
        extended_medium: fullwave.MediumRelaxationMaps,
        theoritical_reflection_coefficient: float,
        n_polynomial: float,
    ) -> fullwave.MediumRelaxationMaps:
        """Apply PML to the extended medium relaxation parameters.

        ref: Komatitsch, D., & Martin, R. (2007).
        An unsplit convolutional perfectly matched layer improved
        at grazing incidence for the seismic wave equation.
        Geophysics, 72(5), SM155-SM167. https://doi.org/10.1190/1.2757586

        Parameters
        ----------
        extended_medium : fullwave.MediumRelaxationMaps
            The extended medium relaxation parameters.
        n_polynomial : float
            The polynomial order for the PML damping parameter.
            it changes the transition function shape from the medium to the PML.
        theoritical_reflection_coefficient : float
            The theoretical reflection coefficient for the PML.
            it changes the PML strength. it gets unstable if it is too low.

        Returns
        -------
        fullwave.MediumRelaxationMaps
            The extended medium relaxation parameters with PML applied.

        """
        if self.pml_unstretched:
            self._carry_stretching_to_one(extended_medium.relaxation_param_dict, is_3d=False)
        logger.debug("Applying 2D PML...")
        self._apply_zero_attenuation_gate(extended_medium.relaxation_param_dict)
        # alpha=0 and d=0 will make a and b in the PML be 0
        # this procedure shrinks the multiple relaxation mechanisms to a single one
        alpha_target_pml = 0
        alpha_target_higher_nu = 0
        d_target_higher_nu = 0

        # see Komatitsch, D., & Martin, R. (2007), SM160
        d_target_pml = (
            -(n_polynomial + 1)
            * self.extended_grid.c0
            * np.log(theoritical_reflection_coefficient)
            / (2 * (self.pml_layer_m + self.transition_layer_m))
            # / (2 * (self.pml_layer_m))
        )
        # alpha_pml_entrance = np.pi * self.extended_grid.f0

        out_dict = {}
        relaxation_param_dict = extended_medium.relaxation_param_dict
        rename_dict = _obtain_relax_var_rename_dict(
            n_relaxation_mechanisms=self.extended_medium.n_relaxation_mechanisms,
            is_3d=self.is_3d,
            use_isotropic_relaxation=self.use_isotropic_relaxation,
        )

        def _compute_one(
            key_fw2: str,
            key_py: str,
            relaxation_param_dict: dict[str, NDArray[np.float64]],
            alpha_target_higher_nu: float,
            d_target_higher_nu: float,
            alpha_target_pml: float,
            d_target_pml: float,
            n_polynomial: float,
            is_3d: bool,  # noqa: FBT001
            apply_transition_and_pml_fn: callable,
        ) -> tuple[str, NDArray[np.float64]]:
            """Return (key_fw2, computed_array). No side effects."""
            arr = relaxation_param_dict[key_py]

            if key_fw2 in ["kappa_x", "kappa_u", "kappa_y", "kappa_w"]:
                return key_fw2, arr

            # helper predicates
            is_alpha = (
                ("alpha_u_nu" in key_fw2)
                or ("alpha_x_nu" in key_fw2)
                or ("alpha_w_nu" in key_fw2)
                or ("alpha_y_nu" in key_fw2)
            )
            is_d = (
                ("d_u_nu" in key_fw2)
                or ("d_x_nu" in key_fw2)
                or ("d_w_nu" in key_fw2)
                or ("d_y_nu" in key_fw2)
            )
            has_nu1 = "nu1" in key_fw2

            if is_alpha and (not has_nu1):
                out = apply_transition_and_pml_fn(
                    arr,
                    value_target=alpha_target_higher_nu,
                    array_shape=arr.shape,
                    axis=0,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=is_3d,
                )
                out = apply_transition_and_pml_fn(
                    out,
                    value_target=alpha_target_higher_nu,
                    array_shape=arr.shape,
                    axis=1,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=is_3d,
                )
                return key_fw2, out

            if is_d and (not has_nu1):
                out = apply_transition_and_pml_fn(
                    arr,
                    value_target=d_target_higher_nu,
                    array_shape=arr.shape,
                    axis=0,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=is_3d,
                )
                out = apply_transition_and_pml_fn(
                    out,
                    value_target=d_target_higher_nu,
                    array_shape=arr.shape,
                    axis=1,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=is_3d,
                )
                return key_fw2, out

            if is_alpha and has_nu1:
                out = apply_transition_and_pml_fn(
                    arr,
                    value_target=alpha_target_pml,
                    array_shape=arr.shape,
                    axis=0,
                    transition_type="linear",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=is_3d,
                )
                out = apply_transition_and_pml_fn(
                    out,
                    value_target=alpha_target_pml,
                    array_shape=arr.shape,
                    axis=1,
                    transition_type="linear",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=is_3d,
                )
                return key_fw2, out

            if is_d and has_nu1:
                out = apply_transition_and_pml_fn(
                    arr,
                    value_target=d_target_pml,
                    array_shape=arr.shape,
                    axis=0,
                    n_polynomial=n_polynomial,
                    transition_type="polynomial",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=is_3d,
                )
                out = apply_transition_and_pml_fn(
                    out,
                    value_target=d_target_pml,
                    array_shape=arr.shape,
                    axis=1,
                    n_polynomial=n_polynomial,
                    transition_type="polynomial",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=is_3d,
                )
                return key_fw2, out

            error_msg = f"Unhandled key_fw2 pattern: {key_fw2}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        items = list(rename_dict.items())

        if self.pml_design == "decoupled":
            results = list(
                self._build_decoupled_2d(
                    extended_medium, rename_dict, n_polynomial, d_target_pml
                ).items()
            )
        elif self.xp is not np:
            # GPU path: run sequentially to avoid CuPy multi-thread CUDA context issues
            results = [
                _compute_one(
                    key_fw2,
                    key_py,
                    relaxation_param_dict,
                    alpha_target_higher_nu,
                    d_target_higher_nu,
                    alpha_target_pml,
                    d_target_pml,
                    n_polynomial,
                    self.is_3d,
                    self._apply_transition_and_pml,
                )
                for key_fw2, key_py in items
            ]
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        _compute_one,
                        key_fw2,
                        key_py,
                        relaxation_param_dict,
                        alpha_target_higher_nu,
                        d_target_higher_nu,
                        alpha_target_pml,
                        d_target_pml,
                        n_polynomial,
                        self.is_3d,
                        self._apply_transition_and_pml,
                    )
                    for key_fw2, key_py in items
                ]
                results = [f.result() for f in futures]
        out_dict = dict(results)

        logger.debug("Calculating PML a and b coefficients...")
        axis_list = ["u", "x"] if self.use_isotropic_relaxation else ["u", "w", "x", "y"]
        # Build independent tasks (flatten nested loops)
        tasks = [
            (nu, axis)
            for nu in range(1, extended_medium.n_relaxation_mechanisms + 1)
            for axis in axis_list
        ]

        medium_dtype = getattr(extended_medium, "dtype", None)

        def _worker(
            nu: int,
            axis: str,
        ) -> tuple[str, NDArray[np.float64], str, NDArray[np.float64]]:
            a, b = self._calc_a_and_b(
                d_x=out_dict[f"d_{axis}_nu{nu}"],
                kappa_x=out_dict[f"kappa_{axis}"],
                alpha_x=out_dict[f"alpha_{axis}_nu{nu}"],
                dt=extended_medium.grid.dt,
                output_dtype=medium_dtype,
            )
            # Return keys + values so parent can update dict safely
            return (f"a_pml_{axis}{nu}", a, f"b_pml_{axis}{nu}", b)

        if self.xp is not np:
            # GPU path: run sequentially
            results = [_worker(nu, axis) for nu, axis in tasks]
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(_worker, nu, axis) for nu, axis in tasks]
                results = [f.result() for f in futures]

        for a_key, a_val, b_key, b_val in results:
            out_dict[a_key] = a_val
            out_dict[b_key] = b_val

        logger.debug("PML a and b coefficients calculation completed.")

        logger.debug("Updating extended medium relaxation parameters...")
        extended_medium.relaxation_param_dict_for_fw2.update(
            out_dict,
        )
        logger.debug("PML application completed.")

        return extended_medium

    def _apply_pml_3d(
        self,
        extended_medium: fullwave.MediumRelaxationMaps,
        theoritical_reflection_coefficient: float,
        n_polynomial: float,
    ) -> fullwave.MediumRelaxationMaps:
        """Apply PML to the extended medium relaxation parameters.

        ref: Komatitsch, D., & Martin, R. (2007).
        An unsplit convolutional perfectly matched layer improved
        at grazing incidence for the seismic wave equation.
        Geophysics, 72(5), SM155-SM167. https://doi.org/10.1190/1.2757586

        Parameters
        ----------
        extended_medium : fullwave.MediumRelaxationMaps
            The extended medium relaxation parameters.
        n_polynomial : float
            The polynomial order for the PML damping parameter.
            it changes the transition function shape from the medium to the PML.
        theoritical_reflection_coefficient : float
            The theoretical reflection coefficient for the PML.
            it changes the PML strength. it gets unstable if it is too low.

        Returns
        -------
        fullwave.MediumRelaxationMaps
            The extended medium relaxation parameters with PML applied.

        """
        if self.pml_unstretched:
            self._carry_stretching_to_one(extended_medium.relaxation_param_dict, is_3d=True)
        logger.debug("Applying 3D PML...")
        self._apply_zero_attenuation_gate(extended_medium.relaxation_param_dict)
        # alpha=0 and d=0 will make a and b in the PML be 0
        # this procedure shrinks the multiple relaxation mechanisms to a single one
        alpha_target_pml = 0
        alpha_target_higher_nu = 0
        d_target_higher_nu = 0

        # see Komatitsch, D., & Martin, R. (2007), SM160
        d_target_pml = (
            -(n_polynomial + 1)
            * self.extended_grid.c0
            * np.log(theoritical_reflection_coefficient)
            / (2 * (self.pml_layer_m + self.transition_layer_m))
            # / (2 * self.pml_layer_m)
        )

        out_dict = {}
        relaxation_param_dict = extended_medium.relaxation_param_dict
        rename_dict = _obtain_relax_var_rename_dict(
            n_relaxation_mechanisms=self.extended_medium.n_relaxation_mechanisms,
            is_3d=self.is_3d,
            use_isotropic_relaxation=self.use_isotropic_relaxation,
        )

        def _compute_one(
            key_fw2: str,
            key_py: str,
            relaxation_param_dict: dict[str, NDArray[np.float64]],
            alpha_target_higher_nu: float,
            d_target_higher_nu: float,
            alpha_target_pml: float,
            d_target_pml: float,
            n_polynomial: float,
            is_3d: bool,  # noqa: FBT001
            apply_transition_and_pml_fn: callable,
        ) -> tuple[str, NDArray[np.float64]]:
            """Return (key_fw2, computed_array). No side effects."""
            arr = relaxation_param_dict[key_py]

            if key_fw2 in ["kappa_x", "kappa_u", "kappa_y", "kappa_v", "kappa_z", "kappa_w"]:
                return key_fw2, arr

            # helper predicates
            is_alpha = (
                ("alpha_u_nu" in key_fw2)
                or ("alpha_v_nu" in key_fw2)
                or ("alpha_w_nu" in key_fw2)
                or ("alpha_x_nu" in key_fw2)
                or ("alpha_y_nu" in key_fw2)
                or ("alpha_z_nu" in key_fw2)
            )
            is_d = (
                ("d_u_nu" in key_fw2)
                or ("d_v_nu" in key_fw2)
                or ("d_w_nu" in key_fw2)
                or ("d_x_nu" in key_fw2)
                or ("d_y_nu" in key_fw2)
                or ("d_z_nu" in key_fw2)
            )
            has_nu1 = "nu1" in key_fw2

            if is_alpha and (not has_nu1):
                out = apply_transition_and_pml_fn(
                    arr,
                    value_target=alpha_target_higher_nu,
                    array_shape=arr.shape,
                    axis=0,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=is_3d,
                )
                out = apply_transition_and_pml_fn(
                    out,
                    value_target=alpha_target_higher_nu,
                    array_shape=arr.shape,
                    axis=1,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=is_3d,
                )
                out = apply_transition_and_pml_fn(
                    out,
                    value_target=alpha_target_higher_nu,
                    array_shape=arr.shape,
                    axis=2,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=is_3d,
                )
                return key_fw2, out
            if is_d and (not has_nu1):
                out = apply_transition_and_pml_fn(
                    arr,
                    value_target=d_target_higher_nu,
                    array_shape=arr.shape,
                    axis=0,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=is_3d,
                )
                out = apply_transition_and_pml_fn(
                    out,
                    value_target=d_target_higher_nu,
                    array_shape=arr.shape,
                    axis=1,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=is_3d,
                )
                out = apply_transition_and_pml_fn(
                    out,
                    value_target=d_target_higher_nu,
                    array_shape=arr.shape,
                    axis=2,
                    transition_type="cosine",
                    transit_within_transition_layer=True,
                    is_3d=is_3d,
                )
                return key_fw2, out
            if is_alpha and has_nu1:
                out = apply_transition_and_pml_fn(
                    arr,
                    value_target=alpha_target_pml,
                    array_shape=arr.shape,
                    axis=0,
                    transition_type="linear",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=is_3d,
                )
                out = apply_transition_and_pml_fn(
                    out,
                    value_target=alpha_target_pml,
                    array_shape=arr.shape,
                    axis=1,
                    transition_type="linear",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=is_3d,
                )
                out = apply_transition_and_pml_fn(
                    out,
                    value_target=alpha_target_pml,
                    array_shape=arr.shape,
                    axis=2,
                    transition_type="linear",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=is_3d,
                )
                return key_fw2, out
            if is_d and has_nu1:
                out = apply_transition_and_pml_fn(
                    arr,
                    value_target=d_target_pml,
                    array_shape=arr.shape,
                    axis=0,
                    n_polynomial=n_polynomial,
                    transition_type="polynomial",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=is_3d,
                )
                out = apply_transition_and_pml_fn(
                    out,
                    value_target=d_target_pml,
                    array_shape=arr.shape,
                    axis=1,
                    n_polynomial=n_polynomial,
                    transition_type="polynomial",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=is_3d,
                )
                out = apply_transition_and_pml_fn(
                    out,
                    value_target=d_target_pml,
                    array_shape=arr.shape,
                    axis=2,
                    n_polynomial=n_polynomial,
                    transition_type="polynomial",
                    transit_within_transition_layer=False,
                    transit_within_pml_layer=False,
                    is_3d=is_3d,
                )
                return key_fw2, out
            error_msg = f"Unhandled key_fw2 pattern: {key_fw2}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        items = list(rename_dict.items())

        if self.xp is not np:
            # GPU path: run sequentially to avoid CuPy multi-thread CUDA context issues
            results = [
                _compute_one(
                    key_fw2,
                    key_py,
                    relaxation_param_dict,
                    alpha_target_higher_nu,
                    d_target_higher_nu,
                    alpha_target_pml,
                    d_target_pml,
                    n_polynomial,
                    self.is_3d,
                    self._apply_transition_and_pml,
                )
                for key_fw2, key_py in items
            ]
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(
                        _compute_one,
                        key_fw2,
                        key_py,
                        relaxation_param_dict,
                        alpha_target_higher_nu,
                        d_target_higher_nu,
                        alpha_target_pml,
                        d_target_pml,
                        n_polynomial,
                        self.is_3d,
                        self._apply_transition_and_pml,
                    )
                    for key_fw2, key_py in items
                ]
                results = [f.result() for f in futures]
        out_dict = dict(results)

        logger.debug("Calculating PML a and b coefficients...")
        axis_list = ["u", "x"] if self.use_isotropic_relaxation else ["u", "v", "w", "x", "y", "z"]

        # Build independent tasks (flatten nested loops)
        tasks = [
            (nu, axis)
            for nu in range(1, extended_medium.n_relaxation_mechanisms + 1)
            for axis in axis_list
        ]

        medium_dtype = getattr(extended_medium, "dtype", None)

        def _worker(
            nu: int,
            axis: str,
        ) -> tuple[str, NDArray[np.float64], str, NDArray[np.float64]]:
            a, b = self._calc_a_and_b(
                d_x=out_dict[f"d_{axis}_nu{nu}"],
                kappa_x=out_dict[f"kappa_{axis}"],
                alpha_x=out_dict[f"alpha_{axis}_nu{nu}"],
                dt=extended_medium.grid.dt,
                output_dtype=medium_dtype,
            )
            # Return keys + values so parent can update dict safely
            return (f"a_pml_{axis}{nu}", a, f"b_pml_{axis}{nu}", b)

        if self.xp is not np:
            # GPU path: run sequentially
            results = [_worker(nu, axis) for nu, axis in tasks]
        else:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(_worker, nu, axis) for nu, axis in tasks]
                results = [f.result() for f in futures]

        for a_key, a_val, b_key, b_val in results:
            out_dict[a_key] = a_val
            out_dict[b_key] = b_val

        logger.debug("PML a and b coefficients calculation completed.")

        logger.debug("Updating extended medium relaxation parameters...")
        extended_medium.relaxation_param_dict_for_fw2.update(
            out_dict,
        )
        logger.debug("PML application completed.")

        return extended_medium

    def _apply_transition_and_pml(  # noqa: PLR0912
        self,
        input_array: NDArray[np.float64],
        value_target: float,
        array_shape: tuple[int, ...],
        axis: int = 0,
        *,
        transition_type: str = "smooth",
        n_polynomial: float = 2,
        transit_within_transition_layer: bool = False,
        transit_within_pml_layer: bool = False,
        disable_the_transition_and_pml: bool = False,
        is_3d: bool = False,
    ) -> NDArray[np.float64]:
        if transit_within_transition_layer and transit_within_pml_layer:
            error_msg = (
                "Both transit_within_transition_layer and transit_within_pml_layer "
                "cannot be True at the same time."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        if disable_the_transition_and_pml:
            return input_array

        if transit_within_transition_layer and self.n_transition_layer == 0:
            error_msg = (
                "Transition layer is not defined. "
                "Set transit_within_transition_layer to False or define n_transition_layer."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Input validation
        if axis not in {0, 1, 2}:
            error_msg = f"Invalid axis value. Expected 0, 1, 2, but got {axis}."
            logger.error(error_msg)
            raise ValueError(error_msg)
        if axis == 2 and not is_3d:
            error_msg = (
                "axis=2 is only valid for 3D cases. Set is_3d=True if you are working with 3D data."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

        # Compute layer parameters
        if transit_within_transition_layer:
            layer_thickness = self.n_transition_layer
            layer_offset = self.n_pml_layer
        elif transit_within_pml_layer:
            layer_thickness = self.n_pml_layer
            layer_offset = 0
        else:
            layer_thickness = self.n_pml_layer + self.n_transition_layer
            layer_offset = 0

        # Compute transition function once
        xp = self.xp
        use_gpu = xp is not np
        transition_linspace = xp.linspace(0, 1, layer_thickness + 1)
        transition_map = {
            "smooth": _smooth_transition_function,
            "linear": _linear_transition_function,
            "polynomial": _n_th_deg_polynomial_function,
            "cosine": _cosine_transition_function,
        }

        if transition_type not in transition_map:
            error_msg = f"Invalid transition type: {transition_type}."
            logger.error(error_msg)
            raise ValueError(
                error_msg,
            )

        if transition_type == "polynomial":
            transition_function = transition_map[transition_type](
                transition_linspace,
                n=n_polynomial,
                xp=xp,
            )
        else:
            transition_function = transition_map[transition_type](transition_linspace, xp=xp)

        n_axis_extended = array_shape[axis]
        m_offset = self.m_spatial_order + layer_offset

        # Pre-compute indices (used multiple times)
        up_end = m_offset + layer_thickness
        down_start = n_axis_extended - m_offset - layer_thickness - 1

        # Transfer to GPU if needed
        if use_gpu:
            input_array = xp.asarray(input_array)

        # Move axis to 0 for uniform processing
        working_array = xp.moveaxis(input_array, axis, 0).copy()

        # Apply boundary conditions
        working_array[: m_offset + layer_thickness] = value_target
        working_array[n_axis_extended - m_offset - layer_thickness :] = value_target

        # Apply transitions (axis-agnostic)
        up_start = m_offset - 1
        down_end = n_axis_extended - m_offset

        # Fetch boundary values
        up_vals = working_array[up_end]
        down_vals = working_array[down_start]

        # Reshape for broadcasting based on dimensionality
        if is_3d:
            # For 3D: shape is (L, H, W) after moveaxis
            up_vals = up_vals[None, :, :]
            down_vals = down_vals[None, :, :]
            trans_up = transition_function[::-1][:, None, None]
            trans_down = transition_function[:, None, None]
        else:
            # For 2D: shape is (L, W) after moveaxis
            up_vals = up_vals[None, :]
            down_vals = down_vals[None, :]
            trans_up = transition_function[::-1][:, None]
            trans_down = transition_function[:, None]

        # Apply transitions
        working_array[up_start:up_end] = up_vals - trans_up * (up_vals - value_target)
        working_array[down_start:down_end] = down_vals - trans_down * (down_vals - value_target)

        # Move axis back
        return xp.moveaxis(working_array, 0, axis)

    @staticmethod
    def _calc_time_constants(
        dx: NDArray[np.float64],
        kappa: NDArray[np.float64],
        alpha: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return dx / kappa + alpha

    def _sort_relaxation_param_dict(
        self,
        relaxation_param_dict: dict[str, NDArray[np.float64]],
        relaxation_param_updates: dict[str, NDArray[np.float64]],
        n_relaxation_mechanisms: int,
    ) -> dict:
        kappa_x1 = relaxation_param_updates["kappa_x1"]
        kappa_x2 = relaxation_param_updates["kappa_x2"]

        d_x1 = []
        alpha_x1 = []
        d_x2 = []
        alpha_x2 = []
        time_const_x1 = []
        time_const_x2 = []
        for nu in range(1, n_relaxation_mechanisms + 1):
            d_x1_nu = relaxation_param_updates[f"d_x1_nu{nu}"]
            alpha_x1_nu = relaxation_param_updates[f"alpha_x1_nu{nu}"]
            d_x2_nu = relaxation_param_updates[f"d_x2_nu{nu}"]
            alpha_x2_nu = relaxation_param_updates[f"alpha_x2_nu{nu}"]

            d_x1.append(d_x1_nu)
            alpha_x1.append(alpha_x1_nu)
            d_x2.append(d_x2_nu)
            alpha_x2.append(alpha_x2_nu)

            time_const_x1_nu = self._calc_time_constants(
                dx=d_x1_nu,
                kappa=kappa_x1,
                alpha=alpha_x1_nu,
            )
            time_const_x2_nu = self._calc_time_constants(
                dx=d_x2_nu,
                kappa=kappa_x2,
                alpha=alpha_x2_nu,
            )
            time_const_x1.append(time_const_x1_nu)
            time_const_x2.append(time_const_x2_nu)

        time_const_x1 = np.stack(time_const_x1, axis=-1)
        time_const_x2 = np.stack(time_const_x2, axis=-1)
        d_x1 = np.stack(d_x1, axis=-1)
        alpha_x1 = np.stack(alpha_x1, axis=-1)
        d_x2 = np.stack(d_x2, axis=-1)
        alpha_x2 = np.stack(alpha_x2, axis=-1)

        # sort the nu values based on the time constants
        sorted_indices_x1 = np.argsort(time_const_x1, axis=-1)
        sorted_indices_x2 = np.argsort(time_const_x2, axis=-1)
        relaxation_param_dict["kappa_x1"] = np.atleast_2d(kappa_x1)
        relaxation_param_dict["kappa_x2"] = np.atleast_2d(kappa_x2)

        for nu in range(1, n_relaxation_mechanisms + 1):
            relaxation_param_dict[f"d_x1_nu{nu}"] = np.atleast_2d(
                np.take_along_axis(
                    d_x1,
                    np.expand_dims(sorted_indices_x1[..., nu - 1], axis=-1),
                    axis=-1,
                ).squeeze(-1),
            )
            relaxation_param_dict[f"alpha_x1_nu{nu}"] = np.atleast_2d(
                np.take_along_axis(
                    alpha_x1,
                    np.expand_dims(sorted_indices_x1[..., nu - 1], axis=-1),
                    axis=-1,
                ).squeeze(-1),
            )
            relaxation_param_dict[f"d_x2_nu{nu}"] = np.atleast_2d(
                np.take_along_axis(
                    d_x2,
                    np.expand_dims(sorted_indices_x2[..., nu - 1], axis=-1),
                    axis=-1,
                ).squeeze(-1),
            )
            relaxation_param_dict[f"alpha_x2_nu{nu}"] = np.atleast_2d(
                np.take_along_axis(
                    alpha_x2,
                    np.expand_dims(sorted_indices_x2[..., nu - 1], axis=-1),
                    axis=-1,
                ).squeeze(-1),
            )
        return relaxation_param_dict

    def plot(
        self,
        export_path: Path | str | None = Path("./temp/temp.png"),
        *,
        show: bool = False,
    ) -> None:
        """Plot the medium fields using matplotlib."""
        relaxation_param_dict_keys = initialize_relaxation_param_dict().keys()

        target_map_dict: OrderedDict = OrderedDict(
            [
                ("Sound speed", self.extended_medium.sound_speed),
                ("Density", self.extended_medium.density),
                ("Beta", self.extended_medium.beta),
                ("Air map", self.extended_medium.air_map),
            ],
        )
        for key in relaxation_param_dict_keys:
            target_map_dict[key] = self.extended_medium.relaxation_param_dict[key]

        target_map_dict.update(
            [
                ("PML mask x", self.pml_mask_x[:, None] * np.ones(self.extended_grid.ny)),
                ("PML mask y", np.ones(self.extended_grid.nx)[:, None] * self.pml_mask_y[None, :]),
                ("Source mask", self.extended_source.mask),
                ("Sensor mask", self.extended_sensor.mask),
            ],
        )

        num_plots = len(target_map_dict)
        # calculate subplot shape to make a square
        n_rows = int(np.sqrt(num_plots))
        n_cols = int(np.ceil(num_plots / n_rows))
        # adjust the fig size
        fig_size = (n_cols * 5, n_rows * 5)

        plt.close("all")
        _, axes = plt.subplots(n_rows, n_cols, figsize=fig_size)

        for ax, (title, map_data) in zip(
            axes.flatten(),
            target_map_dict.items(),
            strict=False,
        ):
            plot_utils.plot_array_on_ax(
                ax,
                map_data,
                title=title,
                xlim=(-5, self.extended_grid.ny + 5),
                ylim=(-5, self.extended_grid.nx + 5),
                reverse_y_axis=True,
            )
        plt.tight_layout()

        if export_path is not None:
            plt.savefig(export_path, dpi=300)
        if show:
            plt.show()
        plt.close("all")


@dataclass
class PMLBuilderExponentialAttenuation(PMLBuilder):
    """A class to set up PML for exponential attenuation media."""

    def __init__(
        self,
        grid: fullwave.Grid,
        medium: fullwave.Medium,
        source: fullwave.Source,
        sensor: fullwave.Sensor,
        *,
        m_spatial_order: int = 8,
        n_pml_layer: int = 40,
        use_gpu: bool = False,
        # n_transition_layer: int = 40,
        # pml_alpha_target: float = 1.1,
        # pml_alpha_power_target: float = 1.6,
        # pml_strength_factor: float = 2.0,
        # use_2_relax_mechanisms: bool = False,
    ) -> None:
        """Initialize the PMLSetup with the given medium, source, sensor, and PML parameters.

        Parameters
        ----------
        grid: fullwave.Grid
            The grid configuration.
        medium : fullwave.Medium)
            The medium relaxation maps.
        source : fullwave.Source
            The source configuration.
        sensor : fullwave.Sensor
            The sensor configuration.
        m_spatial_order : int, optional
            fullwave simulation's spatial order (default is 8).
            It depends on the fullwave simulation binary version.
            Fullwave simulation has 2M th order spatial accuracy and fourth order accuracy in time.
            see Pinton, G. (2021) http://arxiv.org/abs/2106.11476 for more detail.
        n_pml_layer : int, optional
            PML layer thickness (default is 40).
        use_gpu : bool, optional
            If True, use CuPy for GPU-accelerated PML computation (default is False).
            Requires CuPy to be installed. Falls back to CPU if CuPy is unavailable.
        n_transition_layer : int, optional
            Number of transition layers (default is 40).
        pml_alpha_target : float, optional
            Target alpha value for PML (default is 0.5).
            This value is used to calculate the transition layer values.
        pml_alpha_power_target : float, optional
            Target alpha power value for PML (default is 1.0).
            This value is used to calculate the transition layer values.
        pml_strength_factor : float, optional
            Strength factor for PML (default is 2.0).
            This value is used to calculate the PML target values.
        use_2_relax_mechanisms : bool, optional
            If True, use 2 relaxation mechanisms for PML for stability (default is False).
            if True, pml_alpha_target, pml_alpha_power_target, and pml_strength_factor are ignored.

        """
        check_functions.check_instance(
            grid,
            fullwave.Grid,
        )
        check_functions.check_instance(
            medium,
            fullwave.Medium,
        )
        check_functions.check_instance(
            source,
            fullwave.Source,
        )
        check_functions.check_instance(
            sensor,
            fullwave.Sensor,
        )

        self.grid_org = grid
        self.medium_org = medium
        self.source_org = source
        self.sensor_org = sensor
        self.is_3d = grid.is_3d

        self.use_gpu = use_gpu
        self.xp: ModuleType = _get_array_module(use_gpu=use_gpu)
        if self.xp is not np:
            logger.info("PMLBuilderExponentialAttenuation: using CuPy GPU backend")
        elif use_gpu:
            logger.warning(
                "PMLBuilderExponentialAttenuation: use_gpu=True but CuPy is not available. "
                "Falling back to CPU (numpy)."
            )

        self.m_spatial_order = m_spatial_order
        self.n_pml_layer = n_pml_layer
        # self.n_transition_layer = n_transition_layer
        # self.pml_alpha_target = pml_alpha_target
        # self.pml_alpha_power_target = pml_alpha_power_target
        # self.pml_strength_factor = pml_strength_factor
        # self.use_2_relax_mechanisms = use_2_relax_mechanisms

        domain_size: tuple[float, ...]
        if self.is_3d:
            domain_size = (
                (self.medium_org.sound_speed.shape[0] + 2 * self.num_boundary_points)
                * self.grid_org.dx,
                (self.medium_org.sound_speed.shape[1] + 2 * self.num_boundary_points)
                * self.grid_org.dy,
                (self.medium_org.sound_speed.shape[2] + 2 * self.num_boundary_points)
                * self.grid_org.dz,
            )
        else:
            domain_size = (
                (self.medium_org.sound_speed.shape[0] + 2 * self.num_boundary_points)
                * self.grid_org.dx,
                (self.medium_org.sound_speed.shape[1] + 2 * self.num_boundary_points)
                * self.grid_org.dy,
            )
        self.extended_grid = fullwave.Grid(
            domain_size=domain_size,
            f0=self.grid_org.f0,
            duration=self.grid_org.duration,
            c0=self.grid_org.c0,
            ppw=self.grid_org.ppw,
            cfl=self.grid_org.cfl,
        )

        logger.debug("building extended medium for pml...")
        attr_names = ["sound_speed", "density", "beta", "alpha_coeff", "alpha_power"]
        if self.xp is not np:
            named_arrays = [(name, getattr(self.medium_org, name)) for name in attr_names]
            extended = self._extend_arrays_gpu(named_arrays)
            # Free original GPU arrays and replace with numpy to reclaim GPU memory
            self._ensure_numpy_medium_arrays(attr_names)
        else:
            named_arrays = [(name, getattr(self.medium_org, name)) for name in attr_names]
            extended = self._extend_arrays_cpu(named_arrays)

        # Extended arrays are numpy — skip re-upload to GPU to avoid
        # wasting PCIe bandwidth. CPU numexpr handles subsequent computation.
        self.extended_medium = fullwave.Medium(
            grid=self.extended_grid,
            sound_speed=extended["sound_speed"],
            density=extended["density"],
            beta=extended["beta"],
            alpha_coeff=extended["alpha_coeff"],
            alpha_power=extended["alpha_power"],
            air_coords=self.medium_org.air_coords + self.num_boundary_points,
            n_relaxation_mechanisms=self.medium_org.n_relaxation_mechanisms,
            path_relaxation_parameters_database=self.medium_org.path_relaxation_parameters_database,
            attenuation_builder=self.medium_org.attenuation_builder,
            dtype=getattr(self.medium_org, "dtype", np.float64),
            use_gpu=False,
        )
        logger.debug("Extended medium for PML built successfully.")

        extended_grid_shape = tuple(
            s + 2 * self.num_boundary_points for s in self.source_org.grid_shape
        )
        incoords_add_ext = (
            self.source_org.incoords_add + self.num_boundary_points
            if getattr(self.source_org, "incoords_add", None) is not None
            else None
        )
        incoords_u_ext = (
            self.source_org.incoords_u + self.num_boundary_points
            if getattr(self.source_org, "incoords_u", None) is not None
            else None
        )
        incoords_v_ext = (
            self.source_org.incoords_v + self.num_boundary_points
            if getattr(self.source_org, "incoords_v", None) is not None
            else None
        )
        incoords_w_ext = (
            self.source_org.incoords_w + self.num_boundary_points
            if getattr(self.source_org, "incoords_w", None) is not None
            else None
        )
        self.extended_source = fullwave.Source(
            p0=self.source_org.p0,
            coords=self.source_org.incoords + self.num_boundary_points,
            grid_shape=extended_grid_shape,
            p0_additive=self.source_org.p0_additive,
            coords_additive=incoords_add_ext,
            u0=getattr(self.source_org, "u0", None),
            coords_u=incoords_u_ext,
            v0=getattr(self.source_org, "v0", None),
            coords_v=incoords_v_ext,
            w0=getattr(self.source_org, "w0", None),
            coords_w=incoords_w_ext,
        )
        if self.sensor_org.is_sparse_grid:
            self.extended_sensor = fullwave.Sensor(
                mod_x=self.sensor_org.mod_x,
                mod_y=self.sensor_org.mod_y,
                mod_z=self.sensor_org.mod_z,
                sampling_modulus_time=self.sensor_org.sampling_modulus_time,
            )
        else:
            extended_sensor_grid_shape = tuple(
                s + 2 * self.num_boundary_points for s in self.sensor_org.grid_shape
            )
            self.extended_sensor = fullwave.Sensor(
                coords=self.sensor_org.outcoords + self.num_boundary_points,
                grid_shape=extended_sensor_grid_shape,
                sampling_modulus_time=self.sensor_org.sampling_modulus_time,
            )
        logger.debug("Extended source and sensor for PML built successfully.")

        logger.debug("Localizing PML region...")
        if self.is_3d:
            self.pml_mask_x, self.pml_mask_y, self.pml_mask_z = self._localize_pml_region()
        else:
            self.pml_mask_x, self.pml_mask_y = self._localize_pml_region()
        logger.debug("PML region localized successfully.")

        self.pml_layer_m = self.extended_grid.dx * self.n_pml_layer
        # self.transition_layer_m = self.extended_grid.dx * self.n_transition_layer

        self.n_polynomial = 2
        self.theoritical_reflection_coefficient = 10 ** (-30)

        if self.n_pml_layer == 0:
            self.n_transition_layer = 0

    @cached_property
    def num_boundary_points(self) -> int:
        """Returns the number of the boundary points.

        Number of PML layer and ghost cells.
        """
        return self.n_pml_layer + self.m_spatial_order

    def run(self, *, use_pml: bool = True) -> fullwave.MediumExponentialAttenuation:
        """Generate perfect matched layer (PML) relaxation parameters.

        It generates the relaxation parameters
        for the PML region considering the given medium and PML parameters.

        Returns
        -------
        Medium
            A Medium instance with the constructed domain properties.

        """
        if use_pml:
            logger.debug("Building extended medium for PML...")
            extended_medium: fullwave.MediumExponentialAttenuation = (
                self.extended_medium.build_exponential()
            )
            # Free the intermediate Medium's GPU arrays — only alpha_exp is needed
            self._free_extended_medium_gpu()
            logger.debug("Extended medium for PML built successfully.")
            if self.is_3d:
                logger.debug("Applying 3D PML to the extended medium...")
                return self._apply_pml_3d(
                    extended_medium=extended_medium,
                )

            logger.debug("Applying 2D PML to the extended medium...")
            return self._apply_pml_2d(
                extended_medium=extended_medium,
            )

        logger.debug("PML is disabled. Building extended medium without applying PML...")
        extended_medium: fullwave.MediumExponentialAttenuation = (
            self.extended_medium.build_exponential()
        )
        self._free_extended_medium_gpu()
        logger.debug("Extended medium built successfully without applying PML.")
        return extended_medium

    def _free_extended_medium_gpu(self) -> None:
        """Free GPU arrays from extended_medium that are no longer needed.

        After build_exponential(), the Medium's alpha_coeff, alpha_power,
        sound_speed, density, and beta are duplicated in the returned
        MediumExponentialAttenuation. Free them to reclaim GPU memory.
        """
        if self.xp is np:
            return
        import cupy as cp  # noqa: PLC0415

        medium = self.extended_medium
        for attr in ("sound_speed", "density", "beta", "alpha_coeff", "alpha_power"):
            val = getattr(medium, attr, None)
            if val is not None and not isinstance(val, np.ndarray):
                setattr(medium, attr, cp.asnumpy(val))
        cp.get_default_memory_pool().free_all_blocks()

    def _mask_body_2d(self, nx: int, ny: int, n_body: int) -> NDArray[np.float32]:
        """Create a mask for the PML region.

        Parameters
        ----------
        nx : int
            Number of grid points in the x-direction.
        ny : int
            Number of grid points in the y-direction.
        n_body : int
            Thickness of the body region (non-PML region).

        Returns
        -------
        NDArray[np.float32]
            A 2D numpy array representing the PML mask.
            Interior (body) region is 1, PML boundary approaches 0.

        """
        xp = self.xp
        use_gpu = xp is not np

        def edge_distance_1d(n: int, n_body: int) -> NDArray[np.float32]:
            d = xp.zeros(n, dtype=xp.float32)
            if n_body <= 0:
                return d
            d[:n_body] = xp.arange(n_body, 0, -1, dtype=xp.float32)
            d[-n_body:] = xp.arange(1, n_body + 1, dtype=xp.float32)
            return d

        rx = edge_distance_1d(nx, n_body)[:, None]
        ry = edge_distance_1d(ny, n_body)[None, :]

        if use_gpu:
            mask_sq = rx * rx + ry * ry
            mmax = float(xp.sqrt(mask_sq.max()))
            if mmax > 0.0:
                mask_sq = mask_sq / (mmax * mmax)
            return xp.maximum(1 - xp.sqrt(mask_sq), 0)

        rx_np = rx  # noqa: F841
        ry_np = ry  # noqa: F841
        mask_sq = ne.evaluate("rx_np*rx_np + ry_np*ry_np")
        mmax = float(np.sqrt(mask_sq.max()))
        if mmax > 0.0:
            mask_sq = ne.evaluate("mask_sq / (mmax*mmax)")
        result = ne.evaluate("1 - sqrt(mask_sq)")
        return np.maximum(result, 0)

    def _mask_body_3d(self, nx: int, ny: int, nz: int, n_body: int) -> NDArray[np.float32]:
        """Create a mask for the PML region.

        Parameters
        ----------
        nx : int
            Number of grid points in the x-direction.
        ny : int
            Number of grid points in the y-direction.
        nz : int
            Number of grid points in the z-direction.
        n_body : int
            Thickness of the body region (non-PML region).

        Returns
        -------
        NDArray[np.float64]
            A 3D numpy array representing the PML mask.

        """
        xp = self.xp
        use_gpu = xp is not np

        def edge_distance_1d(n: int, n_body: int) -> NDArray[np.float32]:
            d = xp.zeros(n, dtype=xp.float32)
            if n_body <= 0:
                return d
            d[:n_body] = xp.arange(n_body, 0, -1, dtype=xp.float32)
            d[-n_body:] = xp.arange(1, n_body + 1, dtype=xp.float32)
            return d

        rx = edge_distance_1d(nx, n_body)[:, None, None]
        ry = edge_distance_1d(ny, n_body)[None, :, None]
        rz = edge_distance_1d(nz, n_body)[None, None, :]

        if use_gpu:
            mask_sq = rx * rx + ry * ry + rz * rz
            mmax = float(xp.sqrt(mask_sq.max()))
            if mmax > 0.0:
                mask_sq = mask_sq / (mmax * mmax)
            return xp.maximum(1 - xp.sqrt(mask_sq), 0)

        rx_np = rx  # noqa: F841
        ry_np = ry  # noqa: F841
        rz_np = rz  # noqa: F841
        # 1) compute squared distance with numexpr (no reduction here)
        mask_sq = ne.evaluate("rx_np*rx_np + ry_np*ry_np + rz_np*rz_np")

        # 2) reduction done by NumPy, then scalar sqrt
        mmax = float(np.sqrt(mask_sq.max()))
        if mmax > 0.0:
            # 3) normalize in squared space (still via numexpr, elementwise only)
            mask_sq = ne.evaluate("mask_sq / (mmax*mmax)")

        # 4) final sqrt elementwise
        result = ne.evaluate("1 - sqrt(mask_sq)")
        return np.maximum(result, 0)

    def _apply_pml_3d(
        self,
        extended_medium: fullwave.MediumExponentialAttenuation,
    ) -> fullwave.MediumExponentialAttenuation:
        a_mask = self._mask_body_3d(
            nx=extended_medium.alpha_exp.shape[0],
            ny=extended_medium.alpha_exp.shape[1],
            nz=extended_medium.alpha_exp.shape[2],
            n_body=self.num_boundary_points,
        )
        if isinstance(extended_medium.alpha_exp, np.ndarray) and not isinstance(a_mask, np.ndarray):
            import cupy as cp  # noqa: PLC0415

            a_mask = cp.asnumpy(a_mask)
        extended_medium.alpha_exp *= a_mask
        return extended_medium

    def _apply_pml_2d(
        self,
        extended_medium: fullwave.MediumExponentialAttenuation,
    ) -> fullwave.MediumExponentialAttenuation:
        a_mask = self._mask_body_2d(
            nx=extended_medium.alpha_exp.shape[0],
            ny=extended_medium.alpha_exp.shape[1],
            n_body=self.num_boundary_points,
        )
        if isinstance(extended_medium.alpha_exp, np.ndarray) and not isinstance(a_mask, np.ndarray):
            import cupy as cp  # noqa: PLC0415

            a_mask = cp.asnumpy(a_mask)
        extended_medium.alpha_exp *= a_mask
        return extended_medium
