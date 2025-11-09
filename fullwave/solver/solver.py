"""solver module."""

import logging
import time
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

import fullwave
from fullwave.solver.input_file_writer import InputFileWriter
from fullwave.solver.launcher import Launcher
from fullwave.solver.pml_builder import PMLBuilder, PMLBuilderExponentialAttenuation
from fullwave.utils import (
    MemoryTempfile,
    check_functions,
)

from .cuda_utils import get_cuda_architecture, retrieve_cuda_version

logger = logging.getLogger("__main__." + __name__)

COMPATIBLE_CUDA_ARCHITECTURES = [
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
]

VERIFIED_CUDA_ARCHITECTURES = [
    "sm_80",  # Ampere: A100
    "sm_89",  # Ada: RTX 4090, L40, RTX6000
    "sm_120",  # Blackwell: RTX 50 series
    "sm_75",  # Turing: RTX 20*0, T4
]

COMPATIBLE_CUDA_VERSIONS = [
    11.8,
    12.4,
    12.6,
    12.9,
]

COMPATIBLE_CUDA_RANGES = [
    (11.8, 12.9),
]

VERIFIED_CUDA_VERSIONS = [
    12.4,
    12.6,
    12.9,
]

COMPATIBLE_CUDA_VERSIONS_ARCHITECTURES_set = {
    (11.8, "sm_61"),
    (11.8, "sm_70"),
    (11.8, "sm_75"),
    (11.8, "sm_80"),
    (11.8, "sm_86"),
    (11.8, "sm_89"),
    (11.8, "sm_90"),
    # ---
    (12.4, "sm_61"),
    (12.4, "sm_70"),
    (12.4, "sm_75"),
    (12.4, "sm_80"),
    (12.4, "sm_86"),
    (12.4, "sm_89"),
    (12.4, "sm_90"),
    # ---
    (12.6, "sm_61"),
    (12.6, "sm_70"),
    (12.6, "sm_75"),
    (12.6, "sm_80"),
    (12.6, "sm_86"),
    (12.6, "sm_89"),
    (12.6, "sm_90"),
    # ---
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


def _retrieve_fullwave_simulation_path(
    *,
    use_gpu: bool = True,
    is_3d: bool = False,
    use_exponential_attenuation: bool = False,
    use_isotropic_relaxation: bool = True,
    n_relax_mechanisms: int = 2,
) -> Path:
    arch_option = _make_cuda_arch_option(use_gpu=use_gpu)
    cuda_version_option, cuda_version = _make_cuda_version_option(use_gpu=use_gpu)
    isotropic_str = "_isotropic" if use_isotropic_relaxation else ""

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
                / f"fullwave2_3d_exponential_attenuation_gpu_{arch_option}_{cuda_version_option}"
            )
        elif not is_3d and use_gpu:
            path_fullwave_simulation_bin = (
                Path(__file__).parent
                / "bins"
                / "exponential_attenuation"
                / "gpu"
                / "2d"
                / f"fullwave2_2d_exponential_attenuation_gpu_{arch_option}_{cuda_version_option}"
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
            if n_relax_mechanisms != 2:
                error_msg = (
                    "Currently, only 2 relaxation mechanisms are supported in 3D simulations. "
                    "Please set n_relax_mechanisms to 2 for 3D simulations."
                )
                logger.error(error_msg)
                raise NotImplementedError(error_msg)

            path_fullwave_simulation_bin = (
                Path(__file__).parent
                / "bins"
                / "gpu"
                / "3d"
                / f"num_relax={n_relax_mechanisms}"
                / (
                    f"fullwave2_3d_{n_relax_mechanisms}_relax{isotropic_str}"
                    f"_multi_gpu_{arch_option}_{cuda_version_option}"
                )
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
            path_fullwave_simulation_bin = (
                Path(__file__).parent
                / "bins"
                / "gpu"
                / "2d"
                / f"num_relax={n_relax_mechanisms}"
                / (
                    f"fullwave2_2d_{n_relax_mechanisms}_relax{isotropic_str}"
                    f"_multi_gpu_{arch_option}_{cuda_version_option}"
                )
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

    def __init__(  # noqa: PLR0912, PLR0915, C901
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
        run_on_memory: bool = False,
        use_gpu: bool = True,
        use_exponential_attenuation: bool = False,
        use_isotropic_relaxation: bool = True,
        cuda_device_id: str | int | list | None = None,
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
            PML layer thickness (default is 3 ppw).
        n_transition_layer : int, optional
            Number of transition layers (default is 3 ppw).
        run_on_memory : bool, optional
            Flag indicating whether to run the simulation in memory.
            If True, a temporary directory is created in memory.
            it uses the /run/user/{uid} directory if available.
            the maximum size depends on the system configuration.
            if needed, increase the size of /run/user/{uid} using the following website:
            https://wiki.archlinux.org/title/Profile-sync-daemon#Allocate_more_memory_to_accommodate_profiles_in_/run/user/xxxx
            If False, a temporary directory is created on disk.
            Defaults to False.
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

        Raises
        ------
        ValueError:
            If neither a source nor a transducer is provided,
            if neither a sensor nor a transducer is provided,
            or if both source and transducer (or sensor
            and transducer) are defined simultaneously.

        """
        # type hints
        self.source: fullwave.Source
        self.sensor: fullwave.Sensor
        self.medium: fullwave.Medium
        self.grid: fullwave.Grid
        self.input_file_writer: InputFileWriter

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
            path_fullwave_simulation_bin = _retrieve_fullwave_simulation_path(
                use_gpu=use_gpu,
                is_3d=self.is_3d,
                use_exponential_attenuation=self.use_exponential_attenuation,
                use_isotropic_relaxation=use_isotropic_relaxation,
                n_relax_mechanisms=self.n_relax_mechanisms,
            )
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
            pml_layer_thickness_px = 0
            n_transition_layer = 0

        if pml_layer_thickness_px is None:
            pml_layer_thickness_px = self.grid.ppw * 3
        if n_transition_layer is None:
            n_transition_layer = self.grid.ppw * 3

        if source is not None:
            self.source = source
        elif transducer is not None:
            self.source = transducer.source
        else:
            error_msg = "source or transducer must be provided"
            raise ValueError(error_msg)

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

        self.fullwave_launcher = Launcher(
            path_fullwave_simulation_bin,
            is_3d=self.is_3d,
            use_gpu=self.use_gpu,
            cuda_device_id=self.cuda_device_id,
        )

        if use_exponential_attenuation:
            self.pml_builder = PMLBuilderExponentialAttenuation(
                grid=self.grid,
                medium=self.medium,
                source=self.source,
                sensor=self.sensor,
                m_spatial_order=m_spatial_order,
                n_pml_layer=pml_layer_thickness_px,
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
            )

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
    def _reshape_sensor_data(
        raw_sensor_output: NDArray[np.float64],
        sensor: fullwave.Sensor,
    ) -> NDArray[np.float64]:
        """Reshape the raw sensor output data.

        Parameters
        ----------
        raw_sensor_output: NDArray[np.float64]
            The raw sensor output data from the simulation. [nt*ncoordsout, 1]
        sensor: fullwave.Sensor
            The sensor object used in the simulation.

        Returns
        -------
        NDArray[np.float64]: The reshaped sensor output data. [ncoordsout, nt]

        """
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

        Returns
        -------
            NDArray[np.float64]: The simulation output data as a NumPy array.

        """
        # self._save_data_for_beamforming()

        # pml setup
        message = f"Starting Fullwave 2.5 v{fullwave.__version__}..."
        logger.info(message)

        message = f"simulation settings overview: \n{self!s}"
        logger.debug(message)

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

        sensor_mask: NDArray[np.bool_]
        if record_whole_domain:
            if self.is_3d:
                sensor_mask = np.zeros(
                    (
                        self.pml_builder.extended_grid.nx,
                        self.pml_builder.extended_grid.ny,
                        self.pml_builder.extended_grid.nz,
                    ),
                    dtype=bool,
                )
            else:
                sensor_mask = np.zeros(
                    (self.pml_builder.extended_grid.nx, self.pml_builder.extended_grid.ny),
                    dtype=bool,
                )
            sensor_mask[:, :] = True
            sensor = fullwave.Sensor(
                mask=sensor_mask,
                sampling_modulus_time=sampling_modulus_time_whole_domain,
            )
        else:
            sensor = self.pml_builder.extended_sensor

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
            )
            end_loading_time = time.time()
            message = (
                f"Result reshaping completed in "
                f"{end_loading_time - start_loading_time:.2e} seconds."
            )
            logger.info(message)
            return result
        # if load_results is False, return the raw result
        # which is a list of file names
        return sim_result

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
            f"  n_transition_layer={self.pml_builder.n_transition_layer}\n"
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
