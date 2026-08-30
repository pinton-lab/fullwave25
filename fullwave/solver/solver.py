"""solver module."""

import gc
import logging
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import fullwave
from fullwave.solver.input_file_writer import InputFileWriter
from fullwave.solver.launcher import Launcher
from fullwave.solver.pml_builder import PMLBuilder, PMLBuilderExponentialAttenuation
from fullwave.solver.source_type import (
    CLAMPED,
    SOURCE_TYPES,
    as_additive_source,
    is_additive,
)
from fullwave.utils import (
    MemoryTempfile,
    check_functions,
)
from fullwave.utils.signal_filter import apply_filter

from .binary_manager import ensure_binary
from .cuda_utils import get_cuda_architecture, retrieve_cuda_version

logger = logging.getLogger("__main__." + __name__)

COMPATIBLE_CUDA_ARCHITECTURES = [
    "sm_50",  # Maxwell: GTX 750, GTX 750 Ti
    "sm_52",  # Maxwell: GTX 980, GTX 970
    "sm_60",  # Pascal: Tesla P100
    "sm_61",  # Pascal: GTX 10*0
    "sm_70",  # Volta: V100, GTX 1180
    "sm_75",  # Turing: RTX 20*0
    "sm_80",  # Ampere: A100
    "sm_86",  # Ampere: RTX 3080, RTX 3090, etc
    "sm_89",  # Ada: RTX 4090, L40, RTX6000
    "sm_90",  # Hopper: H100, H200
    "sm_100",  # Blackwell: RTX 50 series
    "sm_101",  # Blackwell: RTX 50 series
    "sm_120",  # Blackwell: RTX 50 series
    "sm_121",  # Blackwell: RTX 50 series
]

VERIFIED_CUDA_ARCHITECTURES = [
    "sm_80",  # Ampere: A100
    "sm_86",  # Ampere: RTX 3080, RTX 3090, etc
    "sm_89",  # Ada: RTX 4090, L40, RTX6000
    "sm_120",  # Blackwell: RTX 50 series
    "sm_75",  # Turing: RTX 20*0, T4
    "sm_90",  # Hopper: H100, H200
]


COMPATIBLE_CUDA_VERSIONS = [
    11.8,
    12.4,
    12.9,
    13.0,
    13.1,
]

COMPATIBLE_CUDA_RANGES = [
    (11.8, 13.1),
]

VERIFIED_CUDA_VERSIONS = [
    12.4,
    12.9,
]

COMPATIBLE_CUDA_VERSIONS_ARCHITECTURES_set = {
    (11.8, "sm_50"),
    (11.8, "sm_52"),
    (11.8, "sm_60"),
    (11.8, "sm_61"),
    (11.8, "sm_70"),
    (11.8, "sm_75"),
    (11.8, "sm_80"),
    (11.8, "sm_86"),
    (11.8, "sm_89"),
    (11.8, "sm_90"),
    # ---
    (12.4, "sm_50"),
    (12.4, "sm_52"),
    (12.4, "sm_60"),
    (12.4, "sm_61"),
    (12.4, "sm_70"),
    (12.4, "sm_75"),
    (12.4, "sm_80"),
    (12.4, "sm_86"),
    (12.4, "sm_89"),
    (12.4, "sm_90"),
    # ---
    (12.9, "sm_50"),
    (12.9, "sm_52"),
    (12.9, "sm_60"),
    (12.9, "sm_61"),
    (12.9, "sm_70"),
    (12.9, "sm_75"),
    (12.9, "sm_80"),
    (12.9, "sm_86"),
    (12.9, "sm_89"),
    (12.9, "sm_90"),
    (12.9, "sm_100"),
    (12.9, "sm_101"),
    (12.9, "sm_120"),
    (12.9, "sm_121"),
    # ---
    (13.0, "sm_75"),
    (13.0, "sm_80"),
    (13.0, "sm_86"),
    (13.0, "sm_89"),
    (13.0, "sm_90"),
    (13.0, "sm_100"),
    (13.0, "sm_103"),
    (13.0, "sm_120"),
    (13.0, "sm_121"),
}


def _make_cuda_arch_option(*, use_gpu: bool = True) -> str:
    cuda_archtecture_dict = get_cuda_architecture()[0]  # Get the first device's architecture
    arch_option = (
        "sm_"
        + str(cuda_archtecture_dict["compute_capability"][0])
        + str(cuda_archtecture_dict["compute_capability"][1])
    )

    if use_gpu and arch_option not in COMPATIBLE_CUDA_ARCHITECTURES:
        error_msg = (
            f"CUDA architecture {arch_option} is not compatible. "
            f"Please use one of the following architectures: "
            f"{COMPATIBLE_CUDA_ARCHITECTURES}"
        )
        logger.error(error_msg)
        raise ValueError(error_msg)
    if use_gpu and arch_option not in VERIFIED_CUDA_ARCHITECTURES:
        warning_msg = (
            f"Warning: CUDA architecture {arch_option} is not verified. "
            f"The simulation may work, but it has not been tested extensively. \n"
            f"Verified architectures are: {VERIFIED_CUDA_ARCHITECTURES}. \n"
        )
        logger.warning(warning_msg)
    return arch_option


def _make_cuda_version_option(*, use_gpu: bool = True) -> tuple[str, float]:
    cuda_version: float = retrieve_cuda_version()
    if use_gpu and cuda_version == -1:
        error_msg = (
            "Could not retrieve CUDA version. "
            "Please ensure that the CUDA toolkit is properly installed."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # range check
    if use_gpu and not any(start <= cuda_version <= end for start, end in COMPATIBLE_CUDA_RANGES):
        error_msg = (
            f"CUDA version {cuda_version} is not in the compatible ranges: "
            f"{COMPATIBLE_CUDA_RANGES}. Please install a compatible CUDA version."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    # find the a compatible cuda version below the system cuda
    if use_gpu and cuda_version not in COMPATIBLE_CUDA_VERSIONS:
        compatible_versions_below = [v for v in COMPATIBLE_CUDA_VERSIONS if v < cuda_version]
        if compatible_versions_below:
            closest_version = max(compatible_versions_below)
            message = (
                f"Warning: CUDA version {cuda_version} is not in the compatible versions: "
                f"{COMPATIBLE_CUDA_VERSIONS}. "
                f"Using the closest compatible version {closest_version} instead."
            )
            logger.warning(message)
            cuda_version = closest_version
        else:
            error_msg = (
                f"No compatible CUDA versions found below {cuda_version}. "
                "Please install a compatible CUDA version."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    if use_gpu and cuda_version not in VERIFIED_CUDA_VERSIONS:
        warning_msg = (
            f"Warning: CUDA version {cuda_version} is not in the verified versions: "
            f"{VERIFIED_CUDA_VERSIONS}. The simulation may not run correctly."
        )
        logger.warning(warning_msg)

    return ("cuda" + str(cuda_version).replace(".", ""), cuda_version)


def _check_compatible_set(cuda_version: float, cuda_arch: str) -> bool:
    return (cuda_version, cuda_arch) in COMPATIBLE_CUDA_VERSIONS_ARCHITECTURES_set


def _relaxation_binary_path(
    *,
    dimension: str,
    cuda_version_option: str,
    isotropic_str: str = "",
) -> Path:
    """Return where the relaxation solver binary sits.

    One binary serves every mechanism count, because the kernel reads the count
    from ``n_relax.dat`` at run time rather than from its own name.

    Parameters
    ----------
    dimension : str
        Either "2d" or "3d".
    cuda_version_option : str
        The CUDA tag the binary name ends with, such as "cuda124".
    isotropic_str : str, optional
        The anisotropy tag the name carries, empty for the isotropic solver.

    Returns
    -------
    Path
        The bundled path of the binary, which may not exist.

    """
    root = Path(__file__).parent / "bins" / "gpu" / dimension
    return root / f"fullwave2_{dimension}_n_relax{isotropic_str}_multi_gpu_{cuda_version_option}"


def _retrieve_fullwave_simulation_path(
    *,
    use_gpu: bool = True,
    is_3d: bool = False,
    use_exponential_attenuation: bool = False,
    use_isotropic_relaxation: bool = True,
) -> Path:
    arch_option = _make_cuda_arch_option(use_gpu=use_gpu)
    cuda_version_option, cuda_version = _make_cuda_version_option(use_gpu=use_gpu)
    if use_isotropic_relaxation is False:
        error_msg = (
            "Currently, only isotropic relaxation is supported. "
            "Please set use_isotropic_relaxation to True for the simulation."
        )
        raise NotImplementedError(error_msg)

    # isotropic_str = "_isotropic" if use_isotropic_relaxation else ""
    isotropic_str = ""

    _check_compatible_set(
        cuda_version=cuda_version,
        cuda_arch=arch_option,
    )
    if use_exponential_attenuation:
        if is_3d and use_gpu:
            path_fullwave_simulation_bin = (
                Path(__file__).parent
                / "bins"
                / "exponential_attenuation"
                / "gpu"
                / "3d"
                / f"fullwave2_3d_exponential_attenuation_multi_gpu_{cuda_version_option}"
            )
        elif not is_3d and use_gpu:
            path_fullwave_simulation_bin = (
                Path(__file__).parent
                / "bins"
                / "exponential_attenuation"
                / "gpu"
                / "2d"
                / f"fullwave2_2d_exponential_attenuation_multi_gpu_{cuda_version_option}"
            )
        else:
            error_msg = (
                "Currently, exponential attenuation model is only supported in GPU mode. "
                "Please use GPU mode for exponential attenuation simulations."
            )
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
    elif is_3d:
        if use_gpu:
            path_fullwave_simulation_bin = _relaxation_binary_path(
                dimension="3d",
                cuda_version_option=cuda_version_option,
                isotropic_str=isotropic_str,
            )
        else:
            path_fullwave_simulation_bin = (
                Path(__file__).parent / "bins" / "cpu" / "3d" / "fullwave2_3d_2_relax_multi_cpu"
            )
            error_msg = (
                "Currently, 3D simulation is not supported in CPU mode. "
                "Please use GPU mode for 3D simulations."
            )
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
    else:  # noqa: PLR5501
        if use_gpu:
            path_fullwave_simulation_bin = _relaxation_binary_path(
                dimension="2d",
                cuda_version_option=cuda_version_option,
                isotropic_str=isotropic_str,
            )
        else:
            path_fullwave_simulation_bin = (
                Path(__file__).parent / "bins" / "cpu" / "2d" / "fullwave2_2d_2_relax_multi_cpu"
            )
            error_msg = (
                "Currently, 2D simulation is not supported in CPU mode. "
                "Please use GPU mode for 3D simulations."
            )
            logger.error(error_msg)
            raise NotImplementedError(error_msg)
    return path_fullwave_simulation_bin


class Solver:
    """Solver for fullwave simulation tasks.

    The Solver class manages the setup, input validation, and execution of a fullwave simulation.
    It configures the simulation environment based on the provided
    grid, medium, source, sensor, or transducer,
    generates the required input files, and runs the simulation executable.
    """

    def __init__(  # noqa: PLR0912
        self,
        work_dir: Path,
        grid: fullwave.Grid,
        medium: fullwave.Medium,
        source: fullwave.Source | None = None,
        sensor: fullwave.Sensor | None = None,
        *,
        transducer: fullwave.Transducer | None = None,
        path_fullwave_simulation_bin: Path | None = None,
        use_pml: bool = True,
        m_spatial_order: int = 8,
        pml_layer_thickness_px: int | None = None,
        n_transition_layer: int | None = None,
        exponential_attenuation_pml_thickness_px: int | None = None,
        run_on_memory: bool | None = None,
        use_gpu: bool = True,
        use_exponential_attenuation: bool = False,
        use_isotropic_relaxation: bool = True,
        cuda_device_id: str | int | list | None = None,
        save_gpu_memory: bool = False,
        verify_gpu: bool = True,
        use_gpu_pml: bool = False,
        pml_alpha_entrance: float | None = None,
        source_type: str = CLAMPED,
    ) -> None:
        """Initialize a Solver instance for the fullwave simulation.

        This initializer sets up the simulation
        by assigning the provided grid, medium, source, sensor, and
        transducer (if provided).
        It validates input consistency, generates necessary working directories,
        and prepares the input generator and simulation launcher.

        Parameters
        ----------
        work_dir : (Path)
            Directory to store simulation data and temporary files.
        grid : (fullwave.Grid)
            Instance representing the simulation computational grid.
        medium : (fullwave.MediumRelaxationMaps)
            Instance representing the physical medium where simulations occur.
        source : (fullwave.Source, optional)
            Source defining the simulation input. Optional if a transducer is given.
        sensor : (fullwave.Sensor, optional)
            Sensor defining the simulation output. Optional if a transducer is given.
        transducer : (fullwave.Transducer, optional)
            Transducer instance combining source and sensor information.
            Must not be provided together with source or sensor.
        path_fullwave_simulation_bin : (Path, optional):
            Path to the fullwave simulation binary executable.
            Defaults to a binary in the 'bins' directory relative to this file.
        use_pml : (bool, optional)
            Flag indicating whether to use Perfectly Matched Layer (PML) boundaries.
            Defaults to True.
        m_spatial_order : int, optional
            fullwave simulation's spatial order (default is 8).
            It depends on the fullwave simulation binary version.
            Fullwave simulation has 2M th order spatial accuracy and fourth order accuracy in time.
            see Pinton, G. (2021) http://arxiv.org/abs/2106.11476 for more detail.
        pml_layer_thickness_px : int, optional
            Width of the margin between the interior and the grid edge, in grid
            points. For the relaxation model the default is 4 wavelengths.
            For the exponential attenuation model the default is the depth of
            whichever absorber is in use, so no cell of the margin is idle.
        n_transition_layer : int, optional
            Number of transition layers (default is 3 ppw).
        exponential_attenuation_pml_thickness_px : int, optional
            Thickness of the C-PML absorbing layer in grid points.
            The C-PML belongs to the exponential attenuation model, so this
            argument is refused when ``use_exponential_attenuation`` is False.
            The relaxation model absorbs with its own two stage layer, which
            ``pml_layer_thickness_px`` sizes.
            None, the default, gives 2 wavelengths.
            0 turns the layer off and tapers ``alpha_exp`` in the margin instead,
            which is what every release before this one did.
        run_on_memory : bool, optional
            Flag indicating whether to run the simulation in memory.
            Defaults to True, which keeps the input files and the recorded
            field off the disk. A run writes several arrays of the whole grid
            and reads back one array for each recorded step, and none of that
            needs to survive the run, because ``run`` returns the field.
            If True, a temporary directory is created in memory.
            it uses the /run/user/{uid} directory if available.
            the maximum size depends on the system configuration.
            if needed, increase the size of /run/user/{uid} using the following website:
            https://wiki.archlinux.org/title/Profile-sync-daemon#Allocate_more_memory_to_accommodate_profiles_in_/run/user/xxxx
            It falls back to the disk when no such directory is available.
            None, the default, means memory, and it steps aside for a static
            map, which needs the files on a disk. An explicit True refuses a
            static map rather than stepping aside. Set it to False to keep the
            simulation directory.
        use_gpu : bool, optional
            Whether to use GPU for the simulation.
            Currently, only GPU version is supported.
            Defaults to True.
            In the future support the simulation will be run on multi-core CPU version if False.
        use_exponential_attenuation : bool, optional
            Whether to use exponential attenuation model.
            Defaults to False. If True, the simulation will use exponential attenuation.
            Exponential attenuation is memory efficient and faster
            than the relaxation mechanism model at the cost of attenuation accuracy.
            The exponential attenuation model does not use relaxation mechanisms
            and does not supports frequency power law attenuation.
        use_isotropic_relaxation : bool, optional
            Whether to use isotropic relaxation mechanisms for attenuation modeling
            to reduce memory usage while retaining accuracy.
            For 2D it will reduce the GPU memory usage by approximately 15%.
            For 3D it will reduce the GPU memory usage by approximately 30%
            and CPU memory usage by approximately 20%.
            This option omits the anisotropic relaxation mechanisms to model the attenuation.
            We usually recommend using isotropic relaxation mechanisms
            unless the anisotropic attenuation is required for the simulation.
        cuda_device_id : str | int | list | None, optional
            The CUDA device ID(s) to use for the simulation.
            Defaults to None. If None, the default device ID "0" will be used.
            for multiple GPUs, provide a list of device IDs.
            example 1: [0, 1] for using GPU 0 and GPU 1. or "0,1" as a string.
            example 2: 2 for using GPU 2 or "2" as a string.
        save_gpu_memory : bool, optional
            Whether to save GPU memory by using ICMAT_MEMORY_SAVING flag in the simulation.
        use_gpu_pml : bool, optional
            Whether to use CuPy for GPU-accelerated PML computation (default is False).
            Requires CuPy to be installed. Falls back to CPU if CuPy is unavailable.
            This accelerates the PML array padding and transition computations
            using the GPU, which is especially beneficial for large 3D grids.
            The simulation does not load initial conditions into GPU memory and
            it loads the slice of the wavefield needed for the current time step
            from CPU memory at each time step.
            Defaults to False. If True, it may significantly reduce GPU memory usage,
            but it may also increase the simulation time
            due to the overhead of data transfer between CPU and GPU
            depending on the hardware and the simulation settings.
            useful in 3D simulations with large grid sizes
            where GPU memory is a limiting factor.
        verify_gpu : bool, optional
            Whether to verify that the specified CUDA devices exist on the system.
            Defaults to True. Set to False when generating input files only
            (``generate_input_only=True``) on a machine that may not have
            the target GPUs available.

        Raises
        ------
        ValueError:
            If neither a source nor a transducer is provided,
            if neither a sensor nor a transducer is provided,
            or if both source and transducer (or sensor
            and transducer) are defined simultaneously.

        """
        if exponential_attenuation_pml_thickness_px is not None and not use_exponential_attenuation:
            error_msg = (
                "exponential_attenuation_pml_thickness_px belongs to the exponential "
                "attenuation model, and this run uses the relaxation model. Set "
                "use_exponential_attenuation=True, or size the relaxation PML with "
                "pml_layer_thickness_px."
            )
            raise ValueError(error_msg)

        # type hints
        self.source: fullwave.Source
        self.sensor: fullwave.Sensor
        self.medium: fullwave.Medium
        self.grid: fullwave.Grid
        self.input_file_writer: InputFileWriter
        self.save_gpu_memory = save_gpu_memory

        # None means memory, and it steps aside for a static map. An explicit
        # True refuses one instead, because the caller asked for memory.
        self.run_on_memory_is_stated = run_on_memory is not None
        self.work_dir_on_disk = Path(work_dir)
        run_on_memory = True if run_on_memory is None else run_on_memory
        self.run_on_memory = run_on_memory
        if run_on_memory:
            message = (
                "\nrun_on_memory is set to True."
                "\nThis simulation will be executed in RAM-based temporary directory. "
                "\n"
                "\nIt speeds up the simulation significantly, "
                "\nhowever you need to ensure that sufficient memory"
                "is available for the simulation. "
                "\n"
                "\nIf you encounter memory issues, consider setting run_on_memory to False. "
                "\n"
                "\nThe temporary directory will be created in /run/user/{uid} if available. "
                f"\nThe simulation output will not be saved in {work_dir}. "
                "\n"
                "\nThe maximum size depends on the system configuration. "
                "\nIf needed, increase the size of /run/user/{uid} using the following website: "
                "\nhttps://wiki.archlinux.org/title/Profile-sync-daemon#Allocate_more_memory_to_accommodate_profiles_in_/run/user/xxxx"
                "\n"
            )
            logger.info(message)
            self.tempfile = MemoryTempfile(
                preferred_paths=["/run/user/{uid}"],
                remove_paths=["/dev/shm", "/run/shm"],  # noqa: S108
                additional_paths=["/var/run"],
                filesystem_types=["tmpfs"],
                fallback=True,
            )
            self.tempdir = self.tempfile.TemporaryDirectory()
            self.work_dir = Path(self.tempdir.name)
        else:
            self.work_dir = work_dir
            self.work_dir.mkdir(exist_ok=True, parents=True)

        self.grid = grid
        self.is_3d = grid.is_3d
        self.use_gpu = use_gpu
        self.use_exponential_attenuation = use_exponential_attenuation
        self.use_isotropic_relaxation = use_isotropic_relaxation

        self.n_relax_mechanisms = medium.n_relaxation_mechanisms

        if path_fullwave_simulation_bin is None:
            local_path = _retrieve_fullwave_simulation_path(
                use_gpu=use_gpu,
                is_3d=self.is_3d,
                use_exponential_attenuation=self.use_exponential_attenuation,
                use_isotropic_relaxation=use_isotropic_relaxation,
            )
            path_fullwave_simulation_bin = ensure_binary(local_path)
        else:
            check_functions.check_path_exists(path_fullwave_simulation_bin)

        self._check_input(
            grid,
            medium,
            source,
            sensor,
            transducer,
            path_fullwave_simulation_bin,
        )

        self.medium = medium
        if use_isotropic_relaxation:
            if self.medium.use_isotropic_relaxation is False:
                message = (
                    "Solver is set to use isotropic relaxation, "
                    "but the provided medium is using anisotropic relaxation. "
                    "Overriding the medium to use isotropic relaxation. "
                )
                # warning
                logger.warning(message, UserWarning)
            self.medium.use_isotropic_relaxation = True
        else:
            if self.medium.use_isotropic_relaxation is True:
                message = (
                    "Solver is set to use anisotropic relaxation, "
                    "but the provided medium is using isotropic relaxation. "
                    "Overriding the medium to use anisotropic relaxation. "
                )
                logger.warning(message, UserWarning)
            self.medium.use_isotropic_relaxation = False

        self.use_pml = use_pml
        if not use_pml:
            if exponential_attenuation_pml_thickness_px:
                error_msg = (
                    "use_pml=False asks for no PML, and "
                    "exponential_attenuation_pml_thickness_px="
                    f"{exponential_attenuation_pml_thickness_px} asks for one. Set use_pml=True, "
                    "or drop the thickness."
                )
                raise ValueError(error_msg)
            exponential_attenuation_pml_thickness_px = 0
            pml_layer_thickness_px = 0
            n_transition_layer = 0

        # The exponential attenuation builder sizes its own margin, because the
        # margin holds one absorber and nothing else, and only that builder
        # knows which of its two absorbers is in use.
        if pml_layer_thickness_px is None and not use_exponential_attenuation:
            pml_layer_thickness_px = self.grid.ppw * 4
        if n_transition_layer is None:
            n_transition_layer = self.grid.ppw * 2

        if source is not None:
            self.source = source
        elif transducer is not None:
            self.source = transducer.source
        else:
            error_msg = "source or transducer must be provided"
            raise ValueError(error_msg)

        if source_type not in SOURCE_TYPES:
            error_msg = f"source_type {source_type!r} is not one of {SOURCE_TYPES}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if is_additive(source_type):
            self.source = as_additive_source(self.source, self.grid, self.medium)
            logger.info("source_type=%s, the source is driven by addition", source_type)

        if sensor is not None:
            self.sensor = sensor
        elif transducer is not None:
            self.sensor = transducer.sensor
        else:
            error_msg = "sensor or transducer must be provided"
            logger.error(error_msg)
            raise ValueError(error_msg)

        self.transducer: fullwave.Transducer | None = transducer

        self.path_fullwave_simulation_bin = path_fullwave_simulation_bin
        self.cuda_device_id = cuda_device_id
        self.use_gpu_pml = use_gpu_pml

        self.fullwave_launcher = Launcher(
            path_fullwave_simulation_bin,
            is_3d=self.is_3d,
            use_gpu=self.use_gpu,
            cuda_device_id=self.cuda_device_id,
            save_gpu_memory=self.save_gpu_memory,
            verify_gpu=verify_gpu,
        )

        if use_exponential_attenuation:
            self.pml_builder = PMLBuilderExponentialAttenuation(
                grid=self.grid,
                medium=self.medium,
                source=self.source,
                sensor=self.sensor,
                m_spatial_order=m_spatial_order,
                n_pml_layer=pml_layer_thickness_px,
                exponential_attenuation_pml_thickness_px=exponential_attenuation_pml_thickness_px,
                use_gpu=use_gpu_pml,
            )
        else:
            self.pml_builder = PMLBuilder(
                grid=self.grid,
                medium=self.medium,
                source=self.source,
                sensor=self.sensor,
                m_spatial_order=m_spatial_order,
                n_pml_layer=pml_layer_thickness_px,
                n_transition_layer=n_transition_layer,
                use_isotropic_relaxation=use_isotropic_relaxation,
                use_gpu=use_gpu_pml,
                pml_alpha_entrance=pml_alpha_entrance,
            )

    @staticmethod
    def _release_gpu_memory_pools() -> None:
        """Release all CuPy GPU memory pool blocks back to CUDA.

        Call ``gc.collect()`` first so that Python releases references to
        CuPy arrays, then drain both the device and pinned memory pools.
        This prevents stale allocations from causing memory pressure when
        subsequent operations allocate large GPU arrays.
        """
        gc.collect()
        try:
            import cupy as cp  # noqa: PLC0415

            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()
        except ImportError:
            pass

    @staticmethod
    def _check_input(
        grid: fullwave.Grid,
        medium: fullwave.Medium,
        source: fullwave.Source | None,
        sensor: fullwave.Sensor | None,
        transducer: fullwave.Transducer | None,
        path_fullwave_simulation_bin: Path,
    ) -> None:
        """Check the input values.

        Raises
        ------
        ValueError
            If neither source nor transducer is defined,
            if neither sensor nor transducer is defined,
            or if both source and transducer (or sensor and transducer) are defined simultaneously.

        """
        # check if the source and sensor have value or transducer has value
        if source is None and transducer is None:
            error_msg = "source or transducer must be defined"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if sensor is None and transducer is None:
            error_msg = "sensor or transducer must be defined"
            logger.error(error_msg)
            raise ValueError(error_msg)
        if transducer is not None and source is not None:
            error_msg = "source and transducer cannot be defined at the same time"
            logger.error(error_msg)
            raise ValueError(error_msg)

        if transducer is not None and sensor is not None:
            warning_msg = (
                "sensor and transducer are defined at the same time. "
                "It uses sensor instead of transducer.sensor."
            )
            logger.warning(warning_msg)

        if source is not None and transducer is None:
            check_functions.check_instance(source, fullwave.Source)
        if sensor is not None and transducer is None:
            check_functions.check_instance(sensor, fullwave.Sensor)
        if transducer is not None:
            check_functions.check_instance(transducer, fullwave.Transducer)

        # validate the instances
        check_functions.check_instance(grid, fullwave.Grid)
        check_functions.check_instance(medium, [fullwave.Medium, fullwave.MediumRelaxationMaps])

        if source is not None:
            grid_shape = (grid.nx, grid.ny, grid.nz) if grid.is_3d else (grid.nx, grid.ny)
            source.validate(grid_shape=grid_shape)
        if sensor is not None:
            grid_shape = (grid.nx, grid.ny, grid.nz) if grid.is_3d else (grid.nx, grid.ny)
            sensor.validate(grid_shape=grid_shape)

        error_msg = f"{path_fullwave_simulation_bin} does not exist"
        assert path_fullwave_simulation_bin.exists(), error_msg

    @staticmethod
    def _validate_filter_params(
        highpass_cutoff_mhz: float | None,
        bandpass_cutoff_mhz: tuple[float, float] | None,
        *,
        load_results: bool,
    ) -> None:
        """Validate high-pass / band-pass filter arguments passed to run().

        Raises
        ------
        ValueError
            If both filter options are set simultaneously, or if a filter is
            requested without ``load_results=True``.

        """
        if highpass_cutoff_mhz is not None and bandpass_cutoff_mhz is not None:
            error_msg = (
                "highpass_cutoff_mhz and bandpass_cutoff_mhz cannot both be specified. "
                "Use highpass_cutoff_mhz for a simple high-pass filter or "
                "bandpass_cutoff_mhz for a band-pass filter."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)
        if (highpass_cutoff_mhz is not None or bandpass_cutoff_mhz is not None) and (
            not load_results
        ):
            error_msg = (
                "Filtering requires load_results=True. "
                "Set load_results=True or disable the filter options."
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    @staticmethod
    def _apply_output_filter(
        result: NDArray[np.float64],
        dt: float,
        highpass_cutoff_mhz: float | None,
        bandpass_cutoff_mhz: tuple[float, float] | None,
    ) -> NDArray[np.float64]:
        """Apply the optional frequency filter to the reshaped sensor output.

        Parameters
        ----------
        result : NDArray[np.float64]
            Sensor data shaped ``[n_sensors, n_t]``.
        dt : float
            Grid time step in seconds.
        highpass_cutoff_mhz : float | None
            High-pass edge in MHz, or ``None``.
        bandpass_cutoff_mhz : tuple[float, float] | None
            ``(f_low_mhz, f_high_mhz)`` band-pass edges, or ``None``.

        Returns
        -------
        NDArray[np.float64]
            Filtered (or unchanged) sensor data.

        """
        if highpass_cutoff_mhz is not None:
            logger.info("Applying high-pass filter at %.4g MHz...", highpass_cutoff_mhz)
            return apply_filter(result, dt, f_low_hz=highpass_cutoff_mhz * 1e6)
        if bandpass_cutoff_mhz is not None:
            f_low_hz = bandpass_cutoff_mhz[0] * 1e6
            f_high_hz = bandpass_cutoff_mhz[1] * 1e6
            logger.info(
                "Applying band-pass filter %.4g-%.4g MHz...",
                bandpass_cutoff_mhz[0],
                bandpass_cutoff_mhz[1],
            )
            return apply_filter(result, dt, f_low_hz=f_low_hz, f_high_hz=f_high_hz)
        return result

    @staticmethod
    def _reshape_sensor_data(
        raw_sensor_output: NDArray[np.float64],
        sensor: fullwave.Sensor,
        *,
        n_t: int | None = None,
    ) -> NDArray[np.float64]:
        """Reshape the raw sensor output data.

        Parameters
        ----------
        raw_sensor_output: NDArray[np.float64]
            The raw sensor output data from the simulation. [nt*ncoordsout, 1]
        sensor: fullwave.Sensor
            The sensor object used in the simulation.
        n_t: int | None
            Number of time steps in the extended grid.  Required for sparse-grid
            sensors because n_sensors is not known at Python time.

        Returns
        -------
        NDArray[np.float64]: The reshaped sensor output data. [ncoordsout, nt]

        """
        if sensor.is_sparse_grid:
            if n_t is None:
                msg = "n_t is required to reshape sparse-grid sensor output"
                raise ValueError(msg)
            n_t_recorded = -(-n_t // sensor.sampling_modulus_time)  # ceiling division
            return raw_sensor_output.reshape(n_t_recorded, -1).T
        return raw_sensor_output.reshape(-1, sensor.n_sensors).T

    def run(
        self,
        simulation_dir_name: str | Path = "txrx_0",
        *,
        is_static_map: bool = False,
        recalculate_pml: bool = True,
        record_whole_domain: bool = False,
        sampling_modulus_time_whole_domain: int = 1,
        load_results: bool = True,
        generate_input_only: bool = False,
        release_after_write: bool = False,
        highpass_cutoff_mhz: float | None = None,
        bandpass_cutoff_mhz: tuple[float, float] | None = None,
        gpu_memory_estimate: bool = True,
    ) -> NDArray[np.float64] | Path:
        r"""Run the fullwave simulation and return the result as a NumPy array.

        This method generates the simulation input via the input generator,
        launches the simulation through the external executable,
        and retrieves the output data.
        The simulation directory may be customized,
        and additional parameters control the simulation behavior
        such as static map generation
        and recalculation of the Perfectly Matched Layer (PML).

        Parameters
        ----------
        simulation_dir_name : Path
            The directory name where simulation files will be stored.
            The directory will be created under the work directory.
            This is the directory, where Fullwave2 will be executed
        is_static_map : bool
            Flag indicating if a static map is used.\n
            static map is a map that does not change
            during the transmission events such as plane wave and synthetic aperture sequence.\n
            non-static map is a map that changes
            during the transmission events such as walking aperture implementation
            for focused transmit implementation.\n
            if it is a static map, the input files are stored inside the work directory and
            symbolic links are created in the simulation directory.\n
        recalculate_pml : bool
            Flag indicating whether to re-calculate PML parameters.
            default is True.
            you can store the value false
            if you are using the same PML parameters in case of static map simulation.
            set True if you are using different PML parameters for each transmit event
            such as walking aperture.
            set False if you are using the same PML parameters for each transmit event
            such as plane wave
            AND this is the second or later transmit event.
        record_whole_domain : bool
            Flag indicating whether to record the whole domain.
            If True, the simulation will record data for the entire grid.
        sampling_modulus_time_whole_domain : int
            Sampling modulus in time. Default is 1 (record at every time step).
            Changing this value to n will record the pressure every n time steps.
            It reduces the size of the output data.
            This will only change the sensor class if record_whole_domain is True.
            If record_whole_domain is False,
            the sampling sampling_modulus_time_whole_domain is ignored.
        load_results : bool
            Whether to load the results from genout.dat after the simulation.
            Default is True. If set to False, it returns the genout.dat file path instead.
        generate_input_only : bool
            If True, only generate the input files in the simulation directory and
            skip launching the external Fullwave executable.
            In this case the method returns the simulation directory path.
            Default is False.
        release_after_write : bool
            Whether to release the input files after writing them.
            If True, the memory used by the input files will be released after writing them to disk.
            This is useful when run_on_memory is True to free up memory space for the simulation
            or when the input files are large. Default is False.
        highpass_cutoff_mhz : float | None
            Apply a high-pass filter to the sensor recordings after the simulation.
            Removes low-frequency PML drift by attenuating frequencies below this value (in MHz).
            Uses a cosine (Hann) taper to avoid Gibbs ringing.
            Cannot be combined with ``bandpass_cutoff_mhz``.
            Requires ``load_results=True``.  Default is ``None`` (no filtering).
        bandpass_cutoff_mhz : tuple[float, float] | None
            Apply a band-pass filter ``(f_low_mhz, f_high_mhz)`` to the sensor recordings
            after the simulation.  Retains only frequencies inside the specified band.
            Uses cosine (Hann) tapers on both edges.
            Cannot be combined with ``highpass_cutoff_mhz``.
            Requires ``load_results=True``.  Default is ``None`` (no filtering).
        gpu_memory_estimate : bool
            Whether to estimate GPU memory usage before running the simulation.
            Default is True. If True, it estimates the GPU memory usage.

        Returns
        -------
        NDArray[np.float64] | Path
            The simulation output data as a NumPy array when load_results is True
            and generate_input_only is False.
            Otherwise, a Path to either the 'genout.dat' file
            (when load_results is False) or the simulation directory
            (when generate_input_only is True).

        Raises
        ------
        ValueError
            If run_on_memory is True when is_static_map is True.
            Static map simulations require input files to be stored on a disk.
            run_on_memory, on the other hand, removes the input files
            after the simulation is complete.
            Also raised if both ``highpass_cutoff_mhz`` and ``bandpass_cutoff_mhz`` are given,
            or if either filter option is set but ``load_results=False``.

        """
        # self._save_data_for_beamforming()

        # pml setup
        message = f"Starting Fullwave 2.5 v{fullwave.__version__}..."
        logger.info(message)

        message = f"simulation settings overview: \n{self!s}"
        logger.debug(message)

        if self.run_on_memory and is_static_map:
            if self.run_on_memory_is_stated:
                error_msg = (
                    "run_on_memory cannot be True when is_static_map is True. "
                    "Static map simulations require input files to be stored on a disk. "
                    "run_on_memory, on the other hand, removes the input files after the "
                    "simulation is complete. Please set run_on_memory to False when using "
                    "static map."
                )
                logger.error(error_msg)
                raise ValueError(error_msg)
            message = (
                "A static map needs the input files on a disk, so this run uses work_dir "
                f"{self.work_dir_on_disk} rather than memory. Pass run_on_memory=False to "
                "say so, or run_on_memory=True to refuse a static map."
            )
            logger.warning(message)
            self.run_on_memory = False
            self.work_dir = self.work_dir_on_disk
            self.work_dir.mkdir(exist_ok=True, parents=True)

        self._validate_filter_params(
            highpass_cutoff_mhz,
            bandpass_cutoff_mhz,
            load_results=load_results,
        )

        start_time = time.time()
        extended_medium = self.pml_builder.run(use_pml=self.use_pml)
        end_pml_builder_time = time.time()
        message = f"PML building completed in {end_pml_builder_time - start_time:.2e} seconds."
        logger.debug(message)

        if sampling_modulus_time_whole_domain != 1 and record_whole_domain is False:
            warning_msg = (
                f"Warning: sampling_modulus_time_whole_domain value "
                f"{sampling_modulus_time_whole_domain} is ignored "
                "when record_whole_domain is False. "
                f"The sampling_modulus_time {self.sensor.sampling_modulus_time} "
                "in the sensor object is prioritized."
            )
            logger.warning(warning_msg)

        if record_whole_domain:
            mod_x = 1
            mod_y = 1
            mod_z = 1 if self.is_3d else None

            sensor = fullwave.Sensor(
                mod_x=mod_x,
                mod_y=mod_y,
                mod_z=mod_z,
                sampling_modulus_time=sampling_modulus_time_whole_domain,
            )
        else:
            sensor = self.pml_builder.extended_sensor

        # pml_thickness = PML + transition layers on each side, excluding ghost cells.
        # Used by the binary to locate the interior domain when building a sparse sensor grid.
        interior_offset = self.pml_builder.num_boundary_points - self.pml_builder.m_spatial_order
        pml_thickness = 0 if record_whole_domain else interior_offset
        exponential_attenuation_pml_thickness_px = (
            self.pml_builder.exponential_attenuation_pml_thickness_px
        )

        start_input_file_writer_time = time.time()
        input_file_writer = InputFileWriter(
            work_dir=self.work_dir,
            grid=self.pml_builder.extended_grid,
            medium=extended_medium,
            source=self.pml_builder.extended_source,
            sensor=sensor,
            path_fullwave_simulation_bin=self.path_fullwave_simulation_bin,
            use_exponential_attenuation=self.use_exponential_attenuation,
            use_isotropic_relaxation=self.use_isotropic_relaxation,
            release_after_write=release_after_write,
            pml_thickness=pml_thickness,
            exponential_attenuation_pml_thickness_px=exponential_attenuation_pml_thickness_px,
            exponential_attenuation_pml_interior_offset_px=interior_offset,
            use_gpu=self.use_gpu_pml,
        )
        simulation_dir = input_file_writer.run(
            simulation_dir_name,
            is_static_map=is_static_map,
            recalculate_pml=recalculate_pml,
        )
        end_input_file_writer_time = time.time()
        message = (
            f"Input file writing completed in "
            f"{end_input_file_writer_time - start_input_file_writer_time:.2e} seconds."
        )
        logger.debug(message)

        if gpu_memory_estimate:
            self._estimate_gpu_memory(sensor)

        if generate_input_only:
            logger.info(
                "Input data generation completed in %s. Skipping simulation execution.",
                simulation_dir,
            )
            self._release_gpu_memory_pools()
            return simulation_dir

        self._release_gpu_memory_pools()

        sim_result = self.fullwave_launcher.run(
            simulation_dir,
            load_results=load_results,
        )

        if load_results:
            logger.info("reshaping the result...")

            start_loading_time = time.time()
            result = self._reshape_sensor_data(
                sim_result,
                sensor=sensor,
                n_t=self.pml_builder.extended_grid.nt,
            )
            end_loading_time = time.time()
            message = (
                f"Result reshaping completed in "
                f"{end_loading_time - start_loading_time:.2e} seconds."
            )
            logger.info(message)

            return self._apply_output_filter(
                result,
                self.grid.dt,
                highpass_cutoff_mhz,
                bandpass_cutoff_mhz,
            )
        # if load_results is False, return the raw result
        # which is a list of file names
        return sim_result

    def _estimate_gpu_memory(
        self,
        sensor: fullwave.Sensor,
    ) -> None:
        """Estimate and log GPU memory usage per device.

        Provides a pre-launch estimate so users can verify that the simulation
        fits in GPU memory before the binary starts allocating.

        Parameters
        ----------
        sensor : fullwave.Sensor
            The sensor that will actually be written to the input files (may
            differ from ``self.sensor`` when ``record_whole_domain=True``).

        """
        # show that this is an experimental feature
        logger.info("Estimating GPU memory usage... (experimental feature, may be inaccurate)")
        device_ids = self.fullwave_launcher.cuda_device_id.split(",")
        n_gpus = len(device_ids)

        grid = self.pml_builder.extended_grid
        source = self.pml_builder.extended_source
        medium = self.pml_builder.extended_medium

        depth = grid.nx
        lateral = grid.ny * grid.nz if self.is_3d else grid.ny
        halo_depth = 8

        float_bytes = 4
        int_bytes = 4

        c_map = medium.sound_speed
        c_range = int(np.rint(c_map.max()) - np.rint(c_map.min()))
        n_deriv_levels = 1 if c_range == 0 else c_range + 1

        n_source_timesteps = source.icmat.shape[1]

        gb = 1024.0**3

        base_depth = depth // n_gpus
        remainder = depth % n_gpus

        for rank, dev_id in enumerate(device_ids):
            depth_this = base_depth + (1 if rank < remainder else 0)

            if n_gpus == 1:
                n_halo_sides = 0
            elif rank == 0 or rank == n_gpus - 1:
                n_halo_sides = 1
            else:
                n_halo_sides = 2
            local_depth = depth_this + n_halo_sides * halo_depth
            slab = local_depth * lateral

            n_sources = max(source.n_sources // n_gpus, 0)
            n_sensors = max(sensor.n_sensors // n_gpus, 0)
            n_air_local = max(medium.n_air // n_gpus, 0)

            if self.use_exponential_attenuation:
                total = self._mem_exponential(
                    slab,
                    n_deriv_levels,
                    n_sources,
                    n_source_timesteps,
                    save_gpu_memory=self.save_gpu_memory,
                    n_sensors=n_sensors,
                    float_bytes=float_bytes,
                    int_bytes=int_bytes,
                    is_3d=self.is_3d,
                )
            else:
                total = self._mem_relaxation(
                    slab,
                    n_deriv_levels,
                    n_sources,
                    n_source_timesteps,
                    save_gpu_memory=self.save_gpu_memory,
                    n_air=n_air_local,
                    n_sensors=n_sensors,
                    n_relax=self.n_relax_mechanisms,
                    float_bytes=float_bytes,
                    int_bytes=int_bytes,
                    is_3d=self.is_3d,
                )

            mode = "exponential" if self.use_exponential_attenuation else "relaxation"
            saving = ", save_gpu_memory=True" if self.save_gpu_memory else ""
            logger.info(
                "GPU memory estimate [GPU %s] (%s mode%s): "
                "%.2f GB  (depth=%d +%d halo, lateral=%d)",
                dev_id.strip(),
                mode,
                saving,
                total / gb,
                depth_this,
                n_halo_sides * halo_depth,
                lateral,
            )

    @staticmethod
    def _mem_exponential(
        slab: int,
        n_deriv_levels: int,
        n_sources: int,
        n_source_timesteps: int,
        *,
        save_gpu_memory: bool,
        n_sensors: int,
        float_bytes: int,
        int_bytes: int,
        is_3d: bool,
    ) -> int:
        """Return estimated GPU bytes for exponential-attenuation solver.

        Parameters
        ----------
        slab : int
            Grid points per GPU slab (local_depth * lateral).
        n_deriv_levels : int
            Number of derivative-map levels.
        n_sources : int
            Approximate source count on this GPU.
        n_source_timesteps : int
            Number of source time steps.
        save_gpu_memory : bool
            Whether memory-saving mode is active.
        n_sensors : int
            Approximate sensor count on this GPU.
        float_bytes : int
            Bytes per float (4).
        int_bytes : int
            Bytes per int (4).
        is_3d: bool,
            Whether the simulation is 3D (affects sensor memory).

        Returns
        -------
        int
            Total estimated bytes.

        """
        fb = float_bytes
        ib = int_bytes
        ndim = 3 if is_3d else 2
        n_fields = 4 if is_3d else 3  # p, u, [v], w

        # wave fields: n_fields pairs x 2 time levels
        mem = n_fields * 2 * slab * fb
        # material: rho + K + beta + a_exp
        mem += 4 * slab * fb
        # derivative maps (dmap + dcmap)
        mem += 9 * 2 * n_deriv_levels * fb + slab * ib
        # source (icmat + coords)
        if n_sources > 0:
            mem += n_sources * fb if save_gpu_memory else n_sources * n_source_timesteps * fb
            mem += ndim * n_sources * ib
        # sensor (genoutframe + coordsout_local + p_idx_array)
        if n_sensors > 0:
            mem += n_sensors * fb
            mem += (ndim + 1) * n_sensors * ib
            mem += n_sensors * ib
        return mem

    @staticmethod
    def _mem_relaxation(
        slab: int,
        n_deriv_levels: int,
        n_sources: int,
        n_source_timesteps: int,
        *,
        save_gpu_memory: bool,
        n_air: int,
        n_sensors: int,
        n_relax: int,
        float_bytes: int,
        int_bytes: int,
        is_3d: bool,
    ) -> int:
        """Return estimated GPU bytes for relaxation (power-law) solver.

        Parameters
        ----------
        slab : int
            Grid points per GPU slab (local_depth * lateral).
        n_deriv_levels : int
            Number of derivative-map levels.
        n_sources : int
            Approximate source count on this GPU.
        n_source_timesteps : int
            Number of source time steps.
        save_gpu_memory : bool
            Whether memory-saving mode is active.
        n_air : int
            Number of zero-pressure (air) coordinates on this GPU.
        n_sensors : int
            Approximate sensor count on this GPU.
        n_relax : int
            Number of relaxation mechanisms.
        float_bytes : int
            Bytes per float (4).
        int_bytes : int
            Bytes per int (4).
        is_3d: bool,
            Whether the simulation is 3D (affects sensor memory).

        Returns
        -------
        int
            Total estimated bytes.

        """
        fb = float_bytes
        ib = int_bytes
        ndim = 3 if is_3d else 2
        n_fields = 4 if is_3d else 3  # p, u, [v], w

        # wave fields: n_fields pairs x 2 time levels
        mem = n_fields * 2 * slab * fb
        # relaxation psi:
        mem += 2 * (ndim * n_relax * 2 * slab * fb)
        # material: rho + K + beta
        mem += 3 * slab * fb
        # kappa: 2 arrays (kappa_x1, kappa_x2)
        mem += 2 * slab * fb
        # PML: pml_x1 + pml_x2, each has 2 * n_relax arrays
        mem += 2 * (2 * n_relax) * slab * fb
        # (dmap + dcmap)
        mem += 9 * 2 * n_deriv_levels * fb + slab * ib
        # source (icmat + coords)
        if n_sources > 0:
            mem += n_sources * fb if save_gpu_memory else n_sources * n_source_timesteps * fb
            mem += ndim * n_sources * ib

        # air
        if n_air > 0:
            mem += ndim * n_air * ib
        # sensor
        if n_sensors > 0:
            mem += n_sensors * fb
            mem += (ndim + 1) * n_sensors * ib
            mem += n_sensors * ib
        return mem

    def print_info(self) -> None:
        """Print the Solver instance information."""
        print(str(self))

    def summary(self) -> None:
        """Alias for print_info."""
        self.print_info()

    def __str__(self) -> str:
        """Return a string representation of the Solver instance.

        Returns
        -------
        str
            A formatted string containing the Solver's attributes.

        """
        n_transition_layer = (
            self.pml_builder.n_transition_layer
            if hasattr(self.pml_builder, "n_transition_layer")
            else 0
        )
        return (
            f"\nSolver(\n"
            f"  version={fullwave.__version__}\n"
            f"  work_dir={self.work_dir}\n\n"
            f"  medium={self.medium}\n"
            f"  source={self.source}\n"
            f"  sensor={self.sensor}\n"
            f"  transducer={self.transducer}\n\n"
            f"  path_fullwave_simulation_bin={self.path_fullwave_simulation_bin}\n"
            f"  use_pml={self.use_pml}\n"
            f"  pml_thickness_px={self.pml_builder.n_pml_layer}\n"
            f"  n_transition_layer={n_transition_layer}\n"
            f"  is_3d={self.is_3d}\n"
            f"  use_gpu={self.use_gpu}\n"
            f"  use_exponential_attenuation={self.use_exponential_attenuation}\n"
            f"  use_isotropic_relaxation={self.use_isotropic_relaxation}\n"
            f")"
        )

    def __repr__(self) -> str:
        """Return a string representation of the Solver instance.

        Returns
        -------
        str
            A formatted string containing the Solver's attributes.

        """
        return self.__str__()
