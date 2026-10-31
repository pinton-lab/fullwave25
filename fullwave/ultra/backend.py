"""The experimental fullwave2-ultra backend of ``Solver(use_ultra=True)``: shared parts and 3D."""

from __future__ import annotations

import inspect
import time
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import numpy as np

from fullwave.solver.binary_markers import check_binary_holds_source
from fullwave.solver.source_type import (
    HARD_RELEASED_AFTER,
    SOURCE_TYPES,
    as_additive_source,
    is_hard_held,
    is_soft,
)
from fullwave.utils import run_directory_in_memory
from fullwave.utils.signal_filter import apply_filter

from . import binary as binary_module
from . import dat_io
from .errors import FullwaveUltraExperimentalWarning, UltraUnsupportedError
from .medium_to_aexp import back_faces_of_source
from .time_axis import check_time_axis, delay_additive_signal
from .writer_3d import write_run_directory_3d, write_sources_3d

if TYPE_CHECKING:
    from tempfile import TemporaryDirectory

    from numpy.typing import NDArray

    import fullwave

CFL_LIMIT_3D = 0.30
SPEED_OPTIMIZED_PPW = 6

PACKAGE_DIR = Path(__file__).parents[1]

_shown_warnings: set[str] = set()


def _stacklevel_of_first_caller_outside_package() -> int:
    """Return the ``warnings.warn`` stacklevel of the first frame outside the fullwave package."""
    frame = inspect.currentframe()
    stacklevel = 0
    while frame is not None and Path(frame.f_code.co_filename).is_relative_to(PACKAGE_DIR):
        frame = frame.f_back
        stacklevel += 1
    return stacklevel


def _warn_once(*, name: str, message: str, category: type[Warning]) -> None:
    """Warn with ``message`` the first time ``name`` is seen in this process.

    The warning points at the line of the user's code, however deep the call is.
    """
    if name in _shown_warnings:
        return
    _shown_warnings.add(name)
    warnings.warn(message, category, stacklevel=_stacklevel_of_first_caller_outside_package())


def warn_sponge_without_ultra() -> None:
    """Warn once per process that ``ultra_sponge_thickness_px`` needs ``use_ultra=True``."""
    _warn_once(
        name="sponge_without_ultra",
        message=(
            "ultra_sponge_thickness_px is ignored without use_ultra=True. The native "
            "absorber is sized by pml_layer_thickness_px (relaxation model) or "
            "exponential_attenuation_pml_thickness_px (exponential model)."
        ),
        category=UserWarning,
    )


def warn_ppw_above_speed_optimum(*, ppw: int) -> None:
    """Warn once per process when ``ppw`` is above the ppw the backend is optimized for."""
    if ppw <= SPEED_OPTIMIZED_PPW:
        return
    _warn_once(
        name="ppw",
        message=(
            f"use_ultra=True is optimized for speed at ppw={SPEED_OPTIMIZED_PPW}; "
            f"ppw={ppw} runs, but more slowly."
        ),
        category=UserWarning,
    )


def warn_experimental() -> None:
    """Warn once per process that ``use_ultra=True`` is experimental."""
    _warn_once(
        name="experimental",
        message=(
            "Solver(use_ultra=True) is experimental and may change or be removed without notice."
        ),
        category=FullwaveUltraExperimentalWarning,
    )


def warn_ignored(*, option: str, detail: str = "") -> None:
    """Warn once per process that ``use_ultra=True`` ignores ``option``."""
    suffix = f"; {detail}" if detail else ""
    _warn_once(
        name=f"ignored:{option}",
        message=f"{option} is ignored with use_ultra=True{suffix}.",
        category=UserWarning,
    )


def warn_one_device_only(*, device: int) -> None:
    """Warn once per process that ``use_ultra=True`` runs on one GPU, the first one listed."""
    _warn_once(
        name="one_device_only",
        message=f"cuda_device_id lists several GPUs; use_ultra=True runs on GPU {device} only.",
        category=UserWarning,
    )


def refuse(message: str) -> None:
    """Raise ``UltraUnsupportedError`` with ``message``."""
    raise UltraUnsupportedError(message)


def driven_source(
    source: fullwave.Source,
    *,
    grid: fullwave.Grid,
    medium: fullwave.Medium,
    source_type: str,
) -> fullwave.Source:
    """Return ``source`` as the Solver drives it.

    A soft source is scaled and additive. A hard source is returned as it is: the writers
    hold it with the flag ``hold``.

    Raises
    ------
    ValueError
        When a soft source already carries an additive signal.
    """
    if not is_soft(source_type):
        return source
    if source.p0_additive is not None:
        msg = (
            "this source already carries an additive signal. source_type='soft' "
            "converts a hard source into an additive one, and this source holds no "
            "hard source positions, so its signal would be lost."
        )
        raise ValueError(msg)
    return as_additive_source(source, grid, medium, use_exponential_attenuation=True)


def to_host(array: object) -> NDArray:
    """Return ``array`` as a NumPy array, copying it down from the GPU if needed."""
    getter = getattr(array, "get", None)
    return np.asarray(getter() if callable(getter) else array)


def single_device(cuda_device_id: str | int | list | None) -> int:
    """Return the one CUDA device index that ``cuda_device_id`` names."""
    if cuda_device_id is None:
        return 0
    if isinstance(cuda_device_id, str):
        devices = [d for d in cuda_device_id.split(",") if d.strip()]
    elif isinstance(cuda_device_id, (list, tuple)):
        devices = list(cuda_device_id)
    else:
        devices = [cuda_device_id]
    if not devices:
        refuse("cuda_device_id lists no device")
    device = int(devices[0])
    if len(devices) > 1:
        warn_one_device_only(device=device)
    return device


def work_directory(
    *, work_dir: Path, run_on_memory: bool | None
) -> tuple[Path, TemporaryDirectory | None]:
    """Return the parent of the run directories and the temporary directory that owns it.

    None or True for ``run_on_memory`` gives a RAM-backed temporary directory; the second
    value is None otherwise.
    """
    if run_on_memory is None or run_on_memory:
        tempdir = run_directory_in_memory()
        return Path(tempdir.name), tempdir
    work_dir = Path(work_dir)
    work_dir.mkdir(exist_ok=True, parents=True)
    return work_dir, None


def check_filters(
    *,
    load_results: bool,
    highpass_cutoff_mhz: float | None,
    bandpass_cutoff_mhz: tuple[float, float] | None,
) -> None:
    """Raise ``ValueError`` for filter options that cannot be applied."""
    if highpass_cutoff_mhz is not None and bandpass_cutoff_mhz is not None:
        msg = "highpass_cutoff_mhz and bandpass_cutoff_mhz cannot both be specified"
        raise ValueError(msg)
    filtered = highpass_cutoff_mhz is not None or bandpass_cutoff_mhz is not None
    if filtered and not load_results:
        msg = "Filtering requires load_results=True"
        raise ValueError(msg)


def filter_traces(
    traces: NDArray,
    *,
    dt: float,
    highpass_cutoff_mhz: float | None,
    bandpass_cutoff_mhz: tuple[float, float] | None,
) -> NDArray:
    """Return ``traces`` ``(n_sensors, n_frames)`` after the chosen filter, if any."""
    if highpass_cutoff_mhz is not None:
        return apply_filter(traces, dt, f_low_hz=highpass_cutoff_mhz * 1e6)
    if bandpass_cutoff_mhz is not None:
        low, high = bandpass_cutoff_mhz
        return apply_filter(traces, dt, f_low_hz=low * 1e6, f_high_hz=high * 1e6)
    return traces


class UltraBackend3D:
    """Write, run and read one 3D fullwave2-ultra simulation.

    Parameters
    ----------
    work_dir : Path
        Parent of the run directory when ``run_on_memory`` is False.
    grid, medium, source, sensor
        The fullwave25 objects.
    binary : Path
        The ``bench_3d_opt`` binary.
    run_on_memory : bool | None
        None or True runs in a RAM-backed temporary directory.
    nbdy : int | None
        Sponge width in cells; None gives 4 wavelengths.
    cuda_device_id : str | int | list | None
        One CUDA device.
    verify_gpu : bool
        Check that the device is visible before the launch.
    source_type : str
        ``"hard"`` sets the pressure at the source cells for the whole run.
        ``"hard_released_after"`` sets it while the signal plays, then lets the cells evolve
        freely. ``"soft"`` adds the scaled signal to the pressure of the same cells and leaves
        them transparent.
    """

    def __init__(
        self,
        *,
        work_dir: Path,
        grid: fullwave.Grid,
        medium: fullwave.Medium,
        source: fullwave.Source,
        sensor: fullwave.Sensor,
        binary: Path,
        run_on_memory: bool | None,
        nbdy: int | None,
        cuda_device_id: str | int | list | None,
        verify_gpu: bool,
        source_type: str = HARD_RELEASED_AFTER,
    ) -> None:
        self.grid = grid
        self.medium = medium
        self.source = driven_source(source, grid=grid, medium=medium, source_type=source_type)
        self.hold = is_hard_held(source_type)
        self.sensor = sensor
        self.binary = Path(binary)
        self.nbdy = nbdy
        self.device = single_device(cuda_device_id)
        self.verify_gpu = verify_gpu
        self.last_run: dict = {}
        self._check_inputs()
        self._check_binary()
        self.work_dir, self._tempdir = work_directory(
            work_dir=work_dir, run_on_memory=run_on_memory
        )

    @property
    def holds_hard_source(self) -> bool:
        """Whether the hard cells stay at zero pressure after the drive ends."""
        return self.hold and self.source.n_sources > 0 and self.source.p0.shape[1] < self.grid.nt

    def _check_inputs(self) -> None:
        if not self.grid.is_3d:
            refuse("this backend runs a 3D grid")
        if self.grid.cfl > CFL_LIMIT_3D + 1e-12:
            refuse(f"cfl {self.grid.cfl} is above the 3D stability limit {CFL_LIMIT_3D}")
        if type(self.medium).__name__ != "Medium":
            refuse("use_ultra=True needs fullwave.Medium (exponential attenuation)")
        if not np.all(to_host(self.medium.alpha_power) == 1.0):
            refuse("use_ultra=True needs alpha_power == 1 everywhere")
        if any(v is not None for v in (self.source.u0, self.source.v0, self.source.w0)):
            refuse("velocity sources are not supported by use_ultra=True")
        if self.source.n_sources + self.source.n_sources_add == 0:
            refuse("use_ultra=True needs a pressure source, hard or additive")
        if getattr(self.sensor, "is_sparse_grid", False):
            refuse("sparse-grid sensors are not supported by use_ultra=True")

    def _check_binary(self) -> None:
        features = binary_module.inspect_binary(self.binary)
        if not features.is_3d:
            refuse(f"{self.binary} does not read nZ.dat, so it is not a 3D binary (bench_3d_opt)")
        if len(self.medium.air_coords) > 0 and not features.has_air_cells_3d:
            refuse(
                "air cells need a 3D binary that reads icczero.dat and the window file "
                f"nzero0.dat; {self.binary} does not read both"
            )
        if self.source.n_sources_add and not (
            features.has_additive_channel and features.has_additive_signals
        ):
            refuse(
                f"{self.binary} has no additive source channel (it does not read both "
                "ncoords_add.dat and icmat_add.dat) and would run without the additive source"
            )
        if self.holds_hard_source:
            check_binary_holds_source(
                self.binary,
                marker=binary_module.MARKER_HOLD_SOURCE_3D,
                error_type=UltraUnsupportedError,
                fix="rebuild bench_3d_opt from the fullwave2-ultra source that reads nszero0.dat",
            )

    def _checked_source_signals(self) -> tuple[NDArray | None, NDArray | None]:
        """Return the hard and additive signals after checking they fit the time axis."""
        n_t = check_time_axis(self.grid)
        hard = np.asarray(self.source.p0) if self.source.n_sources else None
        additive = np.asarray(self.source.p0_additive) if self.source.n_sources_add else None
        for signal in (hard, additive):
            if signal is not None and signal.shape[1] > n_t:
                msg = (
                    f"the source has {signal.shape[1]} steps, more than the {n_t} of the simulation"
                )
                raise ValueError(msg)
        return hard, additive

    def run_batch(self, *_args: object, **_kwargs: object) -> None:
        """Refuse: a 3D grid runs one transmit at a time."""
        refuse("run_batch is not supported by use_ultra=True in 3D")

    def run(
        self,
        simulation_dir_name: str | Path = "txrx_0",
        *,
        is_static_map: bool = False,
        record_whole_domain: bool = False,
        load_results: bool = True,
        generate_input_only: bool = False,
        highpass_cutoff_mhz: float | None = None,
        bandpass_cutoff_mhz: tuple[float, float] | None = None,
    ) -> NDArray | Path:
        """Write the run directory, run the binary and return the traces.

        Returns
        -------
        NDArray | Path
            Traces ``(n_sensors, n_frames)``; the path of ``genout.dat`` if ``load_results`` is
            False; the run directory if ``generate_input_only`` is True.
        """
        if is_static_map or record_whole_domain:
            refuse("is_static_map and record_whole_domain are not supported by use_ultra=True")
        check_filters(
            load_results=load_results,
            highpass_cutoff_mhz=highpass_cutoff_mhz,
            bandpass_cutoff_mhz=bandpass_cutoff_mhz,
        )

        grid, medium = self.grid, self.medium
        hard, additive = self._checked_source_signals()
        n_tic = (additive if hard is None else hard).shape[1]
        if additive is not None:
            additive = delay_additive_signal(additive)
        rundir = self.work_dir / simulation_dir_name
        scalars = write_run_directory_3d(
            rundir,
            c0=grid.c0,
            omega0=grid.omega,
            dur=grid.duration,
            ppw=grid.ppw,
            cfl=grid.cfl,
            maps={
                "cmap": to_host(medium.sound_speed),
                "rmap": to_host(medium.density),
                "nmap": to_host(medium.beta),
                "amap": to_host(medium.alpha_coeff),
            },
            incoords=self.source.incoords,
            outcoords=self.sensor.outcoords,
            n_tic=n_tic,
            mod_t=self.sensor.sampling_modulus_time,
            nbdy=self.nbdy,
            air_coords=to_host(medium.air_coords) if len(medium.air_coords) else None,
            additive_incoords=self.source.incoords_add if additive is not None else None,
            n_tic_additive=None if additive is None else additive.shape[1],
            back_faces=back_faces_of_source(
                incoords=self.source.incoords,
                shape=tuple(medium.sound_speed.shape),
                held=self.holds_hard_source,
            ),
        )
        write_sources_3d(
            rundir,
            signal=hard,
            additive_signal=additive,
            n_t=scalars["nT"],
            hold=self.holds_hard_source,
        )
        self.last_run = {"writer": scalars, "run_dir": str(rundir)}
        if generate_input_only:
            return rundir

        started = time.perf_counter()
        stdout = binary_module.run_binary(
            self.binary, rundir, device=self.device, verify_gpu=self.verify_gpu
        )
        self.last_run.update(run_seconds=time.perf_counter() - started, stdout=stdout)
        genout = rundir / "genout.dat"
        if not load_results:
            return genout
        traces = dat_io.read_genout(genout, ncoordsout=scalars["ncoordsout"]).T
        return filter_traces(
            traces,
            dt=grid.dt,
            highpass_cutoff_mhz=highpass_cutoff_mhz,
            bandpass_cutoff_mhz=bandpass_cutoff_mhz,
        )


SPONGE_HINT = "set the sponge width with ultra_sponge_thickness_px"


class NativeOnlyOption(NamedTuple):
    """An option of the native solver that ``use_ultra=True`` ignores."""

    solver_default: object
    hint: str = ""


NATIVE_ONLY_OPTIONS: dict[str, NativeOnlyOption] = {
    "pml_layer_thickness_px": NativeOnlyOption(solver_default=None, hint=SPONGE_HINT),
    "n_transition_layer": NativeOnlyOption(solver_default=None, hint=SPONGE_HINT),
    "pml_alpha_entrance": NativeOnlyOption(solver_default=None, hint=SPONGE_HINT),
    "exponential_attenuation_pml_thickness_px": NativeOnlyOption(
        solver_default=None, hint=SPONGE_HINT
    ),
    "use_gpu_pml": NativeOnlyOption(solver_default=False),
    "save_gpu_memory": NativeOnlyOption(solver_default=False),
    "write_maps_the_solver_does_not_read": NativeOnlyOption(solver_default=False),
    "use_isotropic_relaxation": NativeOnlyOption(solver_default=True),
}


def warn_native_only_options(*, options: dict[str, object]) -> None:
    """Warn once per process for each native-only option set to a non-default value."""
    for name, option in NATIVE_ONLY_OPTIONS.items():
        if options[name] != option.solver_default:
            warn_ignored(option=name, detail=option.hint)


def warn_relaxation_model_requested(*, use_exponential_attenuation: bool) -> None:
    """Warn once per process when the relaxation model is requested, as the native default does."""
    if use_exponential_attenuation:
        return
    _warn_once(
        name="attenuation_model",
        message=(
            "use_ultra=True runs the exponential attenuation model, whose attenuation is "
            "frequency-independent (the same at every frequency), not the relaxation model that "
            "use_exponential_attenuation=False requests, whose attenuation is "
            "frequency-dependent."
        ),
        category=UserWarning,
    )


def check_solver_options(
    *,
    use_pml: bool,
    m_spatial_order: int,
    use_gpu: bool,
    source_type: str,
    native_only_options: dict[str, object],
    use_exponential_attenuation: bool,
) -> None:
    """Refuse the ``Solver`` options that ``use_ultra=True`` cannot honor; warn on the rest."""
    if not use_pml:
        refuse("use_pml=False is not supported by use_ultra=True")
    if m_spatial_order != 8:
        refuse("use_ultra=True supports m_spatial_order=8 only")
    if not use_gpu:
        refuse("use_ultra=True needs the GPU")
    if source_type not in SOURCE_TYPES:
        refuse(f"source_type must be one of {SOURCE_TYPES}, not {source_type!r}")
    warn_native_only_options(options=native_only_options)
    warn_relaxation_model_requested(use_exponential_attenuation=use_exponential_attenuation)
