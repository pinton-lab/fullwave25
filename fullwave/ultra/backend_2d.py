"""The 2D fullwave2-ultra backend of ``Solver(use_ultra=True)``; a run is a batch of one."""

from __future__ import annotations

import logging
import shutil
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from fullwave.solver.binary_markers import check_binary_holds_source
from fullwave.solver.source_type import HARD_RELEASED_AFTER, is_hard_held
from fullwave.utils.gpu_memory import (
    free_gpu_memory_bytes,
    members_that_fit,
    release_gpu_memory_pools,
)

from . import binary as binary_module
from . import dat_io
from .backend import (
    check_filters,
    driven_source,
    filter_traces,
    refuse,
    single_device,
    to_host,
    work_directory,
)
from .binary import OUTPUT_TAIL_CHARS
from .errors import UltraBinaryError, UltraUnsupportedError
from .medium_to_aexp import back_faces_of_source, face_margins
from .time_axis import check_time_axis, delay_additive_signal
from .writer_2d import (
    extension_cells,
    frozen_shell_2d,
    write_run_directory_2d,
    write_sources_2d,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray

    import fullwave

logger = logging.getLogger("__main__." + __name__)

FLOAT_BYTES = 4
BYTES_PER_CELL_PER_MEMBER = 6 * FLOAT_BYTES  # pressure and two velocities at two time levels
BYTES_PER_CELL_SHARED = 172  # material maps, sponge coefficients and sponge memory variables
BYTES_PER_AXIS_CELL = 2 * FLOAT_BYTES
BYTES_PER_COORDINATE = 2 * np.dtype(np.int32).itemsize
STENCIL_TABLE_BYTES = 9 * 2 * FLOAT_BYTES


def device_bytes_2d(
    *,
    n_x: int,
    n_y: int,
    n_t: int,
    mod_t: int,
    ncoords: int,
    n_tic: int,
    ncoordsout: int,
    ncoords_add: int = 0,
    n_tic_add: int = 0,
    ncoordszero: int = 0,
) -> tuple[int, int]:
    """Return the GPU memory of a 2D batch, in bytes, without the CUDA context.

    Parameters
    ----------
    n_x, n_y : int
        Cells of the extended grid, which includes the sponge and the stencil margin.
    n_t, mod_t : int
        Number of time steps and sensor decimation in time.
    ncoords, n_tic : int
        Hard source cells and source length in steps.
    ncoordsout : int
        Sensor points.
    ncoords_add, n_tic_add : int
        Additive source cells and source length in steps.
    ncoordszero : int
        Cells held at zero pressure.

    Returns
    -------
    tuple[int, int]
        The bytes shared by all members and the bytes added by each member.
    """
    cells = n_x * n_y
    frames = (n_t - 1) // mod_t
    n_coordinates = ncoords + ncoordszero + ncoordsout + ncoords_add
    per_member = (
        BYTES_PER_CELL_PER_MEMBER * cells
        + FLOAT_BYTES * ncoords * n_tic
        + FLOAT_BYTES * ncoords_add * n_tic_add
        + FLOAT_BYTES * frames * ncoordsout
    )
    shared = (
        BYTES_PER_CELL_SHARED * cells
        + BYTES_PER_AXIS_CELL * (n_x + n_y)
        + BYTES_PER_COORDINATE * n_coordinates
        + FLOAT_BYTES * ncoordsout
        + STENCIL_TABLE_BYTES
    )
    return shared, per_member


def _members_per_run(n_members: int, max_members_per_run: int | None) -> list[range]:
    """Split the members into the fewest runs of at most ``max_members_per_run``."""
    if max_members_per_run is None:
        return [range(n_members)]
    n_runs = -(-n_members // max_members_per_run)
    return [
        range(int(part[0]), int(part[-1]) + 1) for part in np.array_split(range(n_members), n_runs)
    ]


def check_genout_files(
    genout_paths: Sequence[Path], *, ncoordsout: int, rundir: Path, stdout: str
) -> None:
    """Check that every output file of a run is complete.

    Parameters
    ----------
    genout_paths : sequence of Path
        The output file of each batch member.
    ncoordsout : int
        Number of sensor points per frame.
    rundir : Path
        The run directory, named in the error message.
    stdout : str
        Standard output of the binary, of which the end goes into the error message.

    Raises
    ------
    UltraBinaryError
        If a file is missing or empty, does not hold whole ``float32`` frames, or differs in
        size from the other files of the run.
    """
    frame_bytes = FLOAT_BYTES * ncoordsout
    sizes = {path.name: path.stat().st_size if path.is_file() else None for path in genout_paths}
    common_size = max((size for size in sizes.values() if size), default=0)
    incomplete = [
        name
        for name, size in sizes.items()
        if not size or size % frame_bytes != 0 or size != common_size
    ]
    if not incomplete:
        return
    descriptions = [
        f"{name} (missing)" if sizes[name] is None else f"{name} ({sizes[name]} bytes)"
        for name in incomplete
    ]
    msg = (
        "The fullwave2-ultra binary exited with code 0 but its output is incomplete: "
        f"{', '.join(descriptions)}. "
        f"Each output file must be non-empty, a multiple of {frame_bytes} bytes and of equal size. "
        f"Run directory: {rundir}. Last output of the binary:\n{stdout[-OUTPUT_TAIL_CHARS:]}"
    )
    raise UltraBinaryError(msg)


class UltraBackend2D:
    """Write, run and read 2D fullwave2-ultra simulations of one medium and several transmits.

    Parameters
    ----------
    work_dir : Path
        Parent of the run directories when ``run_on_memory`` is False.
    grid, medium, source, sensor
        The fullwave25 objects. All transmits share the medium, the sensor and the source cells.
    binary : Path
        The ``bench_2d_batch`` binary.
    run_on_memory : bool | None
        None or True runs in a RAM-backed temporary directory.
    nbdy : int | None
        Sponge width in cells; None gives 6 wavelengths.
    cuda_device_id : str | int | list | None
        One CUDA device.
    verify_gpu : bool
        Check that the device is visible before the run.
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
        self.source_type = source_type
        self.hold = is_hard_held(source_type)
        self.source = self._as_driven(source)
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

    def _as_driven(self, source: fullwave.Source) -> fullwave.Source:
        """Return ``source`` as the Solver drives it: scaled and additive for a soft source."""
        return driven_source(
            source, grid=self.grid, medium=self.medium, source_type=self.source_type
        )

    def _check_inputs(self) -> None:
        if self.grid.is_3d:
            refuse("this backend runs a 2D grid")
        if type(self.medium).__name__ != "Medium":
            refuse("use_ultra=True needs fullwave.Medium (exponential attenuation)")
        if not np.all(to_host(self.medium.alpha_power) == 1.0):
            refuse("use_ultra=True needs alpha_power == 1 everywhere")
        self._check_source(self.source)
        if getattr(self.sensor, "is_sparse_grid", False):
            refuse("sparse-grid sensors are not supported by use_ultra=True")

    def _check_source(self, source: fullwave.Source) -> None:
        if any(v is not None for v in (source.u0, source.v0, source.w0)):
            refuse("velocity sources are not supported by use_ultra=True")
        if source.n_sources + source.n_sources_add == 0:
            refuse("use_ultra=True needs a pressure source, hard or additive")

    def _check_binary(self) -> None:
        features = binary_module.inspect_binary(self.binary)
        if features.is_3d:
            refuse(
                f"{self.binary} reads nZ.dat, so it is a 3D binary; a 2D grid needs bench_2d_batch"
            )
        if len(self.medium.air_coords) > 0 and not features.has_air_hold_2d:
            refuse(
                "air cells need a 2D binary that reads nTzero.dat or nzero0.dat; "
                f"{self.binary} reads neither"
            )
        if self.source.n_sources_add and not features.has_additive_channel:
            refuse(
                f"{self.binary} has no additive source channel (it does not read ncoords_add.dat) "
                "and would run without the additive source"
            )
        if (
            self.source.n_sources
            and self.source.p0.shape[1] < self.grid.nt
            and not features.has_source_release_2d
        ):
            if self.hold:
                check_binary_holds_source(
                    self.binary,
                    marker=binary_module.MARKER_SOURCE_RELEASE_2D,
                    error_type=UltraUnsupportedError,
                    fix="rebuild bench_2d_batch from the fullwave2-ultra source that reads it",
                )
            refuse(
                f"{self.binary} does not read nszero0.dat, so it cannot release the source "
                "after its signal ends; pad p0 to grid.nt"
            )

    def _check_member(
        self, member: fullwave.Source, *, index: int
    ) -> tuple[NDArray, NDArray | None]:
        """Return the hard and additive signals of one transmit after checking it fits the batch."""
        self._check_source(member)
        reference = self.source
        same_cells = np.array_equal(member.incoords, reference.incoords) and np.array_equal(
            member.incoords_add, reference.incoords_add
        )
        hard = np.asarray(member.p0)
        additive = None if member.p0_additive is None else np.asarray(member.p0_additive)
        same_length = hard.shape == np.shape(reference.p0) and np.shape(additive) == np.shape(
            reference.p0_additive
        )
        if not same_cells or not same_length:
            msg = (
                f"source {index} of the batch differs from the Solver's source in its cells or "
                "its signal length; the transmits of a batch differ only in their signals"
            )
            raise ValueError(msg)
        return hard, additive

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
        """Run the Solver's source as a batch of one.

        Returns
        -------
        NDArray | Path
            Traces ``(n_sensors, n_frames)``; the path of ``genout.dat`` if ``load_results`` is
            False; the run directory if ``generate_input_only`` is True.
        """
        outputs = self._run_driven(
            [self.source],
            simulation_dir_name,
            is_static_map=is_static_map,
            record_whole_domain=record_whole_domain,
            load_results=load_results,
            generate_input_only=generate_input_only,
            highpass_cutoff_mhz=highpass_cutoff_mhz,
            bandpass_cutoff_mhz=bandpass_cutoff_mhz,
        )
        return outputs[0]

    def run_batch(
        self,
        sources: Sequence[fullwave.Source],
        simulation_dir_name: str | Path = "txrx_batch",
        *,
        max_members_per_run: int | None = None,
        is_static_map: bool = False,
        record_whole_domain: bool = False,
        load_results: bool = True,
        generate_input_only: bool = False,
        highpass_cutoff_mhz: float | None = None,
        bandpass_cutoff_mhz: tuple[float, float] | None = None,
    ) -> NDArray | list[Path]:
        """Run the transmits in runs of at most ``max_members_per_run`` (None: one run).

        The free GPU memory can lower the limit, with a warning. The traces do not depend on
        the split.

        Returns
        -------
        NDArray | list[Path]
            Traces ``(n_transmits, n_sensors, n_frames)``; one ``genout`` path per transmit if
            ``load_results`` is False; the run directories if ``generate_input_only`` is True.
        """
        return self._run_driven(
            [self._as_driven(source) for source in sources],
            simulation_dir_name,
            max_members_per_run=max_members_per_run,
            is_static_map=is_static_map,
            record_whole_domain=record_whole_domain,
            load_results=load_results,
            generate_input_only=generate_input_only,
            highpass_cutoff_mhz=highpass_cutoff_mhz,
            bandpass_cutoff_mhz=bandpass_cutoff_mhz,
        )

    def _run_driven(
        self,
        sources: Sequence[fullwave.Source],
        simulation_dir_name: str | Path,
        *,
        max_members_per_run: int | None = None,
        is_static_map: bool,
        record_whole_domain: bool,
        load_results: bool,
        generate_input_only: bool,
        highpass_cutoff_mhz: float | None,
        bandpass_cutoff_mhz: tuple[float, float] | None,
    ) -> NDArray | list[Path]:
        """Run transmits that are already in the form the Solver drives them."""
        if is_static_map or record_whole_domain:
            refuse("is_static_map and record_whole_domain are not supported by use_ultra=True")
        check_filters(
            load_results=load_results,
            highpass_cutoff_mhz=highpass_cutoff_mhz,
            bandpass_cutoff_mhz=bandpass_cutoff_mhz,
        )
        if len(sources) == 0:
            msg = "run_batch needs at least one source."
            raise ValueError(msg)
        if max_members_per_run is not None and max_members_per_run < 1:
            msg = f"max_members_per_run must be 1 or more, not {max_members_per_run}"
            raise ValueError(msg)
        n_t = check_time_axis(self.grid)
        transmits = [self._check_member(source, index=i) for i, source in enumerate(sources)]
        signals = [hard for hard, _ in transmits]
        additive_signals = [
            None if additive is None else delay_additive_signal(additive)
            for _, additive in transmits
        ]
        n_steps = max(signal.shape[1] for signal in transmits[0] if signal is not None)
        if n_steps > n_t:
            msg = f"the source has {n_steps} steps, more than the {n_t} of the simulation"
            raise ValueError(msg)

        has_additive = self.source.n_sources_add > 0
        if generate_input_only:
            members_per_run_limit = max_members_per_run
        else:
            release_gpu_memory_pools()
            members_per_run_limit = self._members_per_run_limit(
                n_members=len(signals),
                max_members_per_run=max_members_per_run,
                n_t=n_t,
                n_tic=signals[0].shape[1],
                n_tic_add=additive_signals[0].shape[1] if has_additive else 0,
            )
        runs = _members_per_run(len(signals), members_per_run_limit)
        outputs = []
        for index, members in enumerate(runs):
            name = simulation_dir_name if len(runs) == 1 else f"{simulation_dir_name}_{index}"
            outputs.append(
                self._run_one(
                    name,
                    [signals[m] for m in members],
                    [additive_signals[m] for m in members] if has_additive else None,
                    n_t=n_t,
                    load_results=load_results,
                    generate_input_only=generate_input_only,
                    highpass_cutoff_mhz=highpass_cutoff_mhz,
                    bandpass_cutoff_mhz=bandpass_cutoff_mhz,
                )
            )
        if load_results and not generate_input_only:
            return np.concatenate(outputs, axis=0)
        return [path for run_paths in outputs for path in run_paths]

    def _members_per_run_limit(
        self,
        *,
        n_members: int,
        max_members_per_run: int | None,
        n_t: int,
        n_tic: int,
        n_tic_add: int,
    ) -> int | None:
        """Return the most members one run may hold, or None when nothing limits it.

        A run that does not fit into the free GPU memory ends with empty output files.
        """
        wanted = n_members if max_members_per_run is None else min(max_members_per_run, n_members)
        free = free_gpu_memory_bytes([str(self.device)])
        if free is None:
            logger.warning(
                "The free GPU memory is unknown, so the %d transmits per run are not checked "
                "against it.",
                wanted,
            )
            return max_members_per_run
        shared_bytes, per_member_bytes = self._member_bytes(
            n_t=n_t, n_tic=n_tic, n_tic_add=n_tic_add
        )
        fit = max(
            members_that_fit(
                free_bytes=free[str(self.device)],
                shared_bytes=shared_bytes,
                per_member_bytes=per_member_bytes,
            ),
            1,
        )
        if fit < wanted:
            logger.warning(
                "The free GPU memory fits %d transmits per run, not %d. "
                "The %d transmits run in runs of at most %d.",
                fit,
                wanted,
                n_members,
                fit,
            )
        return min(wanted, fit)

    def _back_faces(self) -> frozenset[str]:
        """Return the faces behind a held source that touches the grid edge."""
        return back_faces_of_source(
            incoords=self.source.incoords,
            shape=tuple(self.medium.sound_speed.shape),
            held=self.hold and self.source.n_sources > 0,
        )

    def _member_bytes(self, *, n_t: int, n_tic: int, n_tic_add: int) -> tuple[int, int]:
        """Return the GPU bytes shared by all members and the bytes added by each member."""
        nbdy, mext = extension_cells(ppw=self.grid.ppw, nbdy=self.nbdy)
        margins = face_margins(
            ndim=2, mext=mext, shell=frozen_shell_2d(), nbdy=nbdy, back_faces=self._back_faces()
        )
        n_x = self.medium.sound_speed.shape[0] + margins["x_low"] + margins["x_high"]
        n_y = self.medium.sound_speed.shape[1] + margins["y_low"] + margins["y_high"]
        return device_bytes_2d(
            n_x=n_x,
            n_y=n_y,
            n_t=n_t,
            mod_t=self.sensor.sampling_modulus_time,
            ncoords=len(self.source.incoords),
            n_tic=n_tic,
            ncoordsout=len(self.sensor.outcoords),
            ncoords_add=len(self.source.incoords_add) if n_tic_add else 0,
            n_tic_add=n_tic_add,
            ncoordszero=len(self.medium.air_coords),
        )

    def _run_one(
        self,
        simulation_dir_name: str | Path,
        signals: list[NDArray],
        additive_signals: list[NDArray] | None,
        *,
        n_t: int,
        load_results: bool,
        generate_input_only: bool,
        highpass_cutoff_mhz: float | None,
        bandpass_cutoff_mhz: tuple[float, float] | None,
    ) -> NDArray | list[Path]:
        grid, medium = self.grid, self.medium
        rundir = self.work_dir / simulation_dir_name
        scalars = write_run_directory_2d(
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
            n_tic=signals[0].shape[1],
            mod_t=self.sensor.sampling_modulus_time,
            nbdy=self.nbdy,
            air_coords=medium.air_coords if len(medium.air_coords) else None,
            additive_incoords=self.source.incoords_add if additive_signals else None,
            n_tic_additive=additive_signals[0].shape[1] if additive_signals else None,
            back_faces=self._back_faces(),
        )
        write_sources_2d(
            rundir,
            signals=signals,
            n_t=n_t,
            additive_signals=additive_signals,
            hold=self.hold and self.source.n_sources > 0 and signals[0].shape[1] < n_t,
        )
        self.last_run = {"writer": scalars, "run_dir": str(rundir)}
        if generate_input_only:
            return [rundir]

        started = time.perf_counter()
        stdout = binary_module.run_binary(
            self.binary, rundir, device=self.device, verify_gpu=self.verify_gpu
        )
        self.last_run.update(run_seconds=time.perf_counter() - started, stdout=stdout)
        genout_paths = [rundir / dat_io.genout_name(m) for m in range(len(signals))]
        check_genout_files(
            genout_paths, ncoordsout=scalars["ncoordsout"], rundir=rundir, stdout=stdout
        )
        if not load_results:
            return genout_paths
        traces = np.stack(
            [
                filter_traces(
                    dat_io.read_genout(path, ncoordsout=scalars["ncoordsout"]).T,
                    dt=grid.dt,
                    highpass_cutoff_mhz=highpass_cutoff_mhz,
                    bandpass_cutoff_mhz=bandpass_cutoff_mhz,
                )
                for path in genout_paths
            ]
        )
        if self._tempdir is not None:
            shutil.rmtree(rundir)
        return traces
