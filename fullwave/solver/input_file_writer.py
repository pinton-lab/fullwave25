"""input generator modules."""

import concurrent.futures
import logging
import os
import time
from pathlib import Path

import numpy as np
from numpy.typing import DTypeLike, NDArray

import fullwave
from fullwave.utils import check_functions
from fullwave.utils.numerical import matlab_round

logger = logging.getLogger("__main__." + __name__)


def _as_written_float32(variable_mat: np.ndarray) -> NDArray[np.float32]:
    """Return a map as the float32 the solver reads from its file.

    Parameters
    ----------
    variable_mat : np.ndarray
        The map, on the host or on the accelerator.

    Returns
    -------
    NDArray[np.float32]
        The same map, cast the way `_write_matrix` casts it.

    """
    if not isinstance(variable_mat, np.ndarray):
        try:
            variable_mat = variable_mat.get()
        except AttributeError:
            variable_mat = np.asarray(variable_mat)
    return np.ascontiguousarray(variable_mat, dtype=np.float32)


class InputFileWriter:
    """Base class for Fullwave input data generation.

    if you want to make your own InputGenerator,
    you can inherit this class and override the methods such as "__init__", "run".
    """

    def __init__(
        self,
        work_dir: Path,
        grid: fullwave.Grid,
        medium: fullwave.MediumRelaxationMaps | fullwave.MediumExponentialAttenuation,
        source: fullwave.Source,
        sensor: fullwave.Sensor,
        *,
        path_fullwave_simulation_bin: Path = Path(__file__).parent / "bins" / "fullwave_solver_gpu",
        validate_input: bool = True,
        use_exponential_attenuation: bool = False,
        use_isotropic_relaxation: bool = False,
        release_after_write: bool = False,
        pml_thickness: int = 0,
        exponential_attenuation_pml_thickness_px: int = 0,
        exponential_attenuation_pml_interior_offset_px: int = 0,
        use_gpu: bool = False,
        write_maps_the_solver_does_not_read: bool = False,
    ) -> None:
        """Initialize the InputGeneratorBase instance.

        Parameters
        ----------
        work_dir : Path
            The working directory of the whole simulation.
            the simulation directory will be generated under work_dir.
        grid : fullwave.Grid
            The computational grid.
        medium : fullwave.MediumRelaxationMaps
            The MediumRelaxationMaps properties.
        source : fullwave.Source
            The source configuration.
        sensor : fullwave.Sensor
            The sensor configuration.
        path_fullwave_simulation_bin : Path, optional
            The path to the fullwave simulation binary.
        validate_input: bool, optional
            Flag indicating whether to validate the input data.
            default is True.
        use_exponential_attenuation: bool, optional
            Flag indicating whether to use exponential attenuation.
            default is False.
            If True, the medium should be an instance of MediumExponentialAttenuation.
            If False, the medium should be an instance of MediumRelaxationMaps.
        use_isotropic_relaxation : bool, optional
            Whether to use isotropic relaxation mechanisms for attenuation modeling
            to reduce memory usage while retaining accuracy.
            For 2D it will reduce the memory usage by approximately 15%.
            For 3D it will reduce the memory usage by approximately 25%.
            This option omits the anisotropic relaxation mechanisms to model the attenuation.
            We usually recommend using isotropic relaxation mechanisms
            unless the anisotropic attenuation is required for the simulation.
        release_after_write : bool, optional
            Whether to release the variable from memory after writing to file.
            This can help reduce memory usage when generating input files for large simulations.
            default is False.
        pml_thickness : int, optional
            PML boundary thickness in grid points (n_pml_layer + n_transition_layer).
            Required by the solver binary when using sparse grid (mod_x/mod_y != 0) to determine
            the interior domain boundaries. Also written when mod_x == 0 for future use.
        exponential_attenuation_pml_thickness_px : int, optional
            Thickness of the C-PML absorbing layer in grid points.
            0, the default, writes no file, and the solver then places no layer.
        exponential_attenuation_pml_interior_offset_px : int, optional
            Where the interior starts, counted from the last ghost cell.
            The layer needs it to place its inner face, and it is written
            only with the thickness above. It is a second value because
            pml_thickness carries the sparse recording window, which is 0 for
            a whole domain recording while the interior still starts where it
            always did.
        write_maps_the_solver_does_not_read : bool, optional
            Whether to write the momentum operator maps the n-relaxation solver
            never reads, which are kappax and, above the first mechanism, apmlx
            and bpmlx. False, the default, skips them. True writes them all,
            which an older solver binary needs. The record the skipped maps
            replace is impedance_constraint.dat, and this writer always writes it
            on the relaxation path.

        """
        logger.debug("Initializing InputFileWriter instance.")

        self._work_dir = Path(work_dir)
        self.path_fullwave_simulation_bin = path_fullwave_simulation_bin
        self.use_isotropic_relaxation = use_isotropic_relaxation
        self.write_maps_the_solver_does_not_read = write_maps_the_solver_does_not_read
        self.use_gpu = use_gpu

        if validate_input:
            check_functions.check_path_exists(self.path_fullwave_simulation_bin)
            check_functions.check_instance(grid, fullwave.Grid)
            if use_exponential_attenuation:
                check_functions.check_instance(medium, fullwave.MediumExponentialAttenuation)
            else:
                check_functions.check_instance(medium, fullwave.MediumRelaxationMaps)
            check_functions.check_instance(source, fullwave.Source)
            check_functions.check_instance(sensor, fullwave.Sensor)

        self.grid = grid
        self.medium: fullwave.MediumRelaxationMaps | fullwave.MediumExponentialAttenuation = medium
        self.source = source
        self.sensor = sensor
        self.is_3d = self.grid.is_3d
        self.use_exponential_attenuation = use_exponential_attenuation
        self.release_after_write = release_after_write
        self.pml_thickness = pml_thickness
        self.exponential_attenuation_pml_thickness_px = exponential_attenuation_pml_thickness_px
        self.exponential_attenuation_pml_interior_offset_px = (
            exponential_attenuation_pml_interior_offset_px
        )

        self._precomputed_bulk_modulus: NDArray[np.float32] | None = None

        if self.use_gpu:
            try:
                import cupy as cp  # noqa: PLC0415

                n_gpus = cp.cuda.runtime.getDeviceCount()
                if n_gpus > 1 and self.medium.sound_speed.ndim == 3:
                    c_min_val, c_max_val = self._compute_dc_map_and_bulk_modulus_multi_gpu(
                        n_gpus,
                    )
                else:
                    c_min_val, c_max_val = self._compute_dc_map_and_bulk_modulus_single_gpu()
                self._dim = int(round(c_max_val) - round(c_min_val))
                self._dc_map_ready = True
            except ImportError:
                self._dim = int(
                    np.rint(self.medium.sound_speed.max()) - np.rint(self.medium.sound_speed.min()),
                )
                c_min_val = float(self.medium.sound_speed.min())
                self._dc_map_ready = False
        else:
            self._dim = int(
                np.rint(self.medium.sound_speed.max()) - np.rint(self.medium.sound_speed.min()),
            )
            c_min_val = float(self.medium.sound_speed.min())
            self._dc_map_ready = False

        self._set_d_mat()
        self._set_d_map(self._dim, self.medium.sound_speed, c_min=c_min_val)
        if not self._dc_map_ready:
            self._set_dc_map(self.medium.sound_speed)
        logger.debug("InputFileWriter instance created.")

    def run(
        self,
        simulation_dir_name: Path | str,
        *,
        is_static_map: bool = False,
        recalculate_pml: bool = True,
    ) -> Path:
        r"""Run the input data generation and return the simulation directory path.

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

        Returns
        -------
        Path: The simulation directory.

        """
        logger.info("Generating input files for simulation...")
        time_start = time.time()
        simulation_dir = self._work_dir / simulation_dir_name
        simulation_dir.mkdir(parents=True, exist_ok=True)

        self._init_pending_writes()

        self._queue_ic_write(
            simulation_dir / "icmat.dat",
            np.transpose(self.source.icmat),
        )
        p0_additive = getattr(self.source, "p0_additive", None)
        incoords_add = getattr(self.source, "incoords_add", None)
        if p0_additive is not None:
            self._queue_ic_write(
                simulation_dir / "icmat_add.dat",
                np.transpose(p0_additive),
            )
        if incoords_add is not None:
            self._queue_coords_write(simulation_dir / "icc_add.dat", incoords_add)
        for _vel_suffix, _vel_attr in (("u", "u0"), ("v", "v0"), ("w", "w0")):
            _signal_vel = getattr(self.source, _vel_attr, None)
            _incoords_vel = getattr(self.source, f"incoords_{_vel_suffix}", None)
            if _signal_vel is not None:
                self._queue_ic_write(
                    simulation_dir / f"icmat_{_vel_suffix}.dat",
                    np.transpose(_signal_vel),
                )
            if _incoords_vel is not None:
                self._queue_coords_write(
                    simulation_dir / f"icc_{_vel_suffix}.dat",
                    _incoords_vel,
                )
        self._queue_coords_write(simulation_dir / "icc.dat", self.source.incoords)
        self._copy_simulation_bin_file(simulation_dir)

        if not self.use_exponential_attenuation:
            if recalculate_pml:
                dat_output_dir = self._work_dir if is_static_map else simulation_dir

                self._save_variables_into_dat_file(
                    simulation_dir=dat_output_dir,
                    relaxation_param_map_dict_for_fw2=self.medium.relaxation_param_dict_for_fw2,
                    dim=self._dim,
                )
            if is_static_map:
                self._flush_writes()
                self._build_symbolic_links_for_dat_files(
                    src_dir=self._work_dir.resolve(),
                    dst_dir=simulation_dir.resolve(),
                )
        else:
            if is_static_map:
                message = (
                    "Using exponential attenuation with static map is not supported. "
                    "Setting is_static_map to False."
                )
                logger.warning(message)
                is_static_map = False

            dat_output_dir = (
                self._work_dir if is_static_map else simulation_dir
            )  # retain this if for future use. currently is_static_map is forced to False
            self._save_variables_into_dat_file_exponential_attenuation(
                simulation_dir=dat_output_dir,
                dim=self._dim,
            )

        self._flush_writes()
        end_file_writer_time = time.time()
        message = f"Input files generated in {end_file_writer_time - time_start:.2e} seconds."
        logger.info(message)
        return simulation_dir

    # --- constructor utils ---

    def _set_d_mat(self) -> None:
        logger.debug("Setting d matrix for stencil coefficients.")
        # For 2D modeling:
        self._d = np.zeros((9, 2))
        if self.is_3d:
            self._d[1, 0] = self._horner7(
                self.grid.cfl,
                3.26627215252963e-3,
                -7.91679373564790e-4,
                1.08663532410570e-3,
                2.54974226454794e-2,
                3.23083288193913e-5,
                -3.97704676886853e-1,
                7.95584310128586e-8,
                1.25425295688331,
            )
            self._d[2, 0] = self._horner7(
                self.grid.cfl,
                -2.83291379048757e-3,
                8.52796449228369e-4,
                -9.45353822586534e-4,
                -8.82015372858580e-3,
                -2.81364895458027e-5,
                6.73021045987599e-2,
                -6.93180036837075e-8,
                -1.23448809066664e-1,
            )
            self._d[3, 0] = self._horner7(
                self.grid.cfl,
                2.32775473203342e-3,
                -5.56793042789852e-4,
                7.77649035879584e-4,
                2.45547234243566e-3,
                2.31537892801923e-5,
                1.61900960524164e-2,
                5.70523152308121e-8,
                3.46683979649506e-2,
            )
            self._d[4, 0] = self._horner7(
                self.grid.cfl,
                -1.68883462553539e-3,
                3.03535823592644e-4,
                -5.64777117315819e-4,
                2.44582905523866e-4,
                -1.68215579314751e-5,
                -2.62344345204941e-2,
                -4.14559953526389e-8,
                -1.19918511290930e-2,
            )
            self._d[5, 0] = self._horner7(
                self.grid.cfl,
                1.08994931098070e-3,
                -1.41445142143525e-4,
                3.64794490139160e-4,
                -8.86057426195227e-4,
                1.08681882832738e-5,
                2.07238558666603e-2,
                2.67876079477806e-8,
                4.17058420250698e-3,
            )
            self._d[6, 0] = self._horner7(
                self.grid.cfl,
                -6.39950124405340e-4,
                6.06079815415080e-5,
                -2.14633466007892e-4,
                6.84580412267934e-4,
                -6.39907927898092e-6,
                -1.29825288653404e-2,
                -1.57775422151124e-8,
                -1.29998325971518e-3,
            )
            self._d[7, 0] = self._horner7(
                self.grid.cfl,
                2.92716539609611e-4,
                -1.87446062803024e-5,
                9.85389372183761e-5,
                -2.40360290348543e-4,
                2.94166215515130e-6,
                5.57066438452790e-3,
                7.25741366376659e-9,
                3.18698432679400e-4,
            )
            self._d[8, 0] = self._horner7(
                self.grid.cfl,
                -6.42183857909518e-5,
                3.38552867751042e-6,
                -2.17377151411164e-5,
                4.98269067389945e-5,
                -6.50197868987757e-7,
                -1.19096089679178e-3,
                -1.60559948991172e-9,
                -4.57795411807702e-5,
            )
            self._d[1, 1] = self._horner7(
                self.grid.cfl,
                -4.47723278782936e-5,
                -7.69502473399932e-5,
                -1.41765498250133e-5,
                -2.54672045901272e-3,
                -4.14343385915353e-7,
                5.00210047924752e-2,
                -1.01220354410507e-9,
                -8.07139347787336e-8,
            )
        else:
            self._d[1, 0] = self._horner7(
                self.grid.cfl,
                -8.74634088067635e-4,
                -1.80530560296097e-3,
                -4.40512972481673e-4,
                4.74018847663366e-3,
                -1.93097802254349e-5,
                -2.92328221171893e-1,
                -6.58101498708345e-8,
                1.25420636437969,
            )
            self._d[2, 0] = self._horner7(
                self.grid.cfl,
                7.93317828964018e-4,
                1.61433256585486e-3,
                3.97244786277123e-4,
                5.46057645976549e-3,
                1.73781972873916e-5,
                5.88754971188371e-2,
                5.91706982879834e-8,
                -1.23406473759703e-1,
            )
            self._d[3, 0] = self._horner7(
                self.grid.cfl,
                -6.50217700538851e-4,
                -1.16449260340413e-3,
                -3.24403734066325e-4,
                -9.11483710059994e-3,
                -1.41739982312600e-5,
                2.33184077551615e-2,
                -4.82326094707544e-8,
                3.46342451534453e-2,
            )
            self._d[4, 0] = self._horner7(
                self.grid.cfl,
                4.67529510541428e-4,
                7.32736676632388e-4,
                2.32444388955328e-4,
                8.46419766685254e-3,
                1.01438593426278e-5,
                -3.17586249260511e-2,
                3.44988852042879e-8,
                -1.19674942518101e-2,
            )
            self._d[5, 0] = self._horner7(
                self.grid.cfl,
                -2.98416281187033e-4,
                -3.99380750669364e-4,
                -1.48203388388213e-4,
                -6.01788793192501e-3,
                -6.46543538517443e-6,
                2.41912754935119e-2,
                -2.19855171569984e-8,
                4.15554391204146e-3,
            )
            self._d[6, 0] = self._horner7(
                self.grid.cfl,
                1.67882669698981e-4,
                1.88195874702691e-4,
                8.30579218603960e-5,
                3.48461963201376e-3,
                3.61873162287129e-6,
                -1.49875789940005e-2,
                1.22979142197165e-8,
                -1.29213888778954e-3,
            )
            self._d[7, 0] = self._horner7(
                self.grid.cfl,
                -6.22209937489143e-5,
                -6.44890425871692e-5,
                -3.02936928954918e-5,
                -1.33386143898282e-3,
                -1.31215186728213e-6,
                6.70228205200379e-3,
                -4.44653967516776e-9,
                3.15659916047599e-4,
            )
            self._d[8, 0] = self._horner7(
                self.grid.cfl,
                6.84740881090240e-6,
                1.14082245705934e-5,
                3.03727593705750e-6,
                2.36122782444105e-4,
                1.26768491232397e-7,
                -1.53347270556276e-3,
                4.21617557752767e-10,
                -4.51948990428065e-5,
            )
            self._d[1, 1] = self._horner7(
                self.grid.cfl,
                2.13188763071246e-6,
                -7.41025068776257e-5,
                2.31652037371554e-6,
                -2.59495924602038e-3,
                1.20637183170338e-7,
                5.21123771632193e-2,
                4.42258843694177e-10,
                -4.20967682664542e-7,
            )
        logger.debug("d matrix for stencil coefficients set.")

    @staticmethod
    def _horner7(
        x: np.ndarray | float,
        a7: float,
        a6: float,
        a5: float,
        a4: float,
        a3: float,
        a2: float,
        a1: float,
        a0: float,
    ) -> np.ndarray | float:
        """Evaluate a 7th degree polynomial using Horner's method.

        Parameters
        ----------
        x : np.ndarray | float
            The input value(s) at which to evaluate the polynomial.
        a7, a6, a5, a4, a3, a2, a1, a0 : float
            The coefficients of the polynomial, where a7 is the coefficient of x^7
            and a0 is the constant term.

        Returns
        -------
        np.ndarray | float
            The evaluated polynomial value(s).

        """
        # (((((((a7*x + a6)*x + a5)*x + a4)*x + a3)*x + a2)*x + a1)*x + a0)
        return ((((((a7 * x + a6) * x + a5) * x + a4) * x + a3) * x + a2) * x + a1) * x + a0

    def _set_d_map(self, dim: int, c_map: np.ndarray, *, c_min: float | None = None) -> None:
        self._d_map = np.zeros((9, 2, dim + 1), dtype=np.float64)

        cmin = c_min if c_min is not None else float(np.min(c_map))
        scale = self.grid.dt / self.grid.dx  # compute once
        i = np.arange(dim + 1, dtype=np.float64)
        r = (i + cmin) * scale

        if self.is_3d:
            self._d_map[1, 0, :] = self._horner7(
                r,
                3.26627215252963e-3,
                -7.91679373564790e-4,
                1.08663532410570e-3,
                2.54974226454794e-2,
                3.23083288193913e-5,
                -3.97704676886853e-1,
                7.95584310128586e-8,
                1.25425295688331,
            )
            self._d_map[2, 0, :] = self._horner7(
                r,
                -2.83291379048757e-3,
                8.52796449228369e-4,
                -9.45353822586534e-4,
                -8.82015372858580e-3,
                -2.81364895458027e-5,
                6.73021045987599e-2,
                -6.93180036837075e-8,
                -1.23448809066664e-1,
            )
            self._d_map[3, 0, :] = self._horner7(
                r,
                2.32775473203342e-3,
                -5.56793042789852e-4,
                7.77649035879584e-4,
                2.45547234243566e-3,
                2.31537892801923e-5,
                1.61900960524164e-2,
                5.70523152308121e-8,
                3.46683979649506e-2,
            )
            self._d_map[4, 0, :] = self._horner7(
                r,
                -1.68883462553539e-3,
                3.03535823592644e-4,
                -5.64777117315819e-4,
                2.44582905523866e-4,
                -1.68215579314751e-5,
                -2.62344345204941e-2,
                -4.14559953526389e-8,
                -1.19918511290930e-2,
            )
            self._d_map[5, 0, :] = self._horner7(
                r,
                1.08994931098070e-3,
                -1.41445142143525e-4,
                3.64794490139160e-4,
                -8.86057426195227e-4,
                1.08681882832738e-5,
                2.07238558666603e-2,
                2.67876079477806e-8,
                4.17058420250698e-3,
            )
            self._d_map[6, 0, :] = self._horner7(
                r,
                -6.39950124405340e-4,
                6.06079815415080e-5,
                -2.14633466007892e-4,
                6.84580412267934e-4,
                -6.39907927898092e-6,
                -1.29825288653404e-2,
                -1.57775422151124e-8,
                -1.29998325971518e-3,
            )
            self._d_map[7, 0, :] = self._horner7(
                r,
                2.92716539609611e-4,
                -1.87446062803024e-5,
                9.85389372183761e-5,
                -2.40360290348543e-4,
                2.94166215515130e-6,
                5.57066438452790e-3,
                7.25741366376659e-9,
                3.18698432679400e-4,
            )
            self._d_map[8, 0, :] = self._horner7(
                r,
                -6.42183857909518e-5,
                3.38552867751042e-6,
                -2.17377151411164e-5,
                4.98269067389945e-5,
                -6.50197868987757e-7,
                -1.19096089679178e-3,
                -1.60559948991172e-9,
                -4.57795411807702e-5,
            )
            self._d_map[1, 1, :] = self._horner7(
                r,
                -4.47723278782936e-5,
                -7.69502473399932e-5,
                -1.41765498250133e-5,
                -2.54672045901272e-3,
                -4.14343385915353e-7,
                5.00210047924752e-2,
                -1.01220354410507e-9,
                -8.07139347787336e-8,
            )
        else:
            self._d_map[1, 0, :] = self._horner7(
                r,
                -8.74634088067635e-4,
                -1.80530560296097e-3,
                -4.40512972481673e-4,
                4.74018847663366e-3,
                -1.93097802254349e-5,
                -2.92328221171893e-1,
                -6.58101498708345e-8,
                1.25420636437969,
            )
            self._d_map[2, 0, :] = self._horner7(
                r,
                7.93317828964018e-4,
                1.61433256585486e-3,
                3.97244786277123e-4,
                5.46057645976549e-3,
                1.73781972873916e-5,
                5.88754971188371e-2,
                5.91706982879834e-8,
                -1.23406473759703e-1,
            )
            self._d_map[3, 0, :] = self._horner7(
                r,
                -6.50217700538851e-4,
                -1.16449260340413e-3,
                -3.24403734066325e-4,
                -9.11483710059994e-3,
                -1.41739982312600e-5,
                2.33184077551615e-2,
                -4.82326094707544e-8,
                3.46342451534453e-2,
            )
            self._d_map[4, 0, :] = self._horner7(
                r,
                4.67529510541428e-4,
                7.32736676632388e-4,
                2.32444388955328e-4,
                8.46419766685254e-3,
                1.01438593426278e-5,
                -3.17586249260511e-2,
                3.44988852042879e-8,
                -1.19674942518101e-2,
            )
            self._d_map[5, 0, :] = self._horner7(
                r,
                -2.98416281187033e-4,
                -3.99380750669364e-4,
                -1.48203388388213e-4,
                -6.01788793192501e-3,
                -6.46543538517443e-6,
                2.41912754935119e-2,
                -2.19855171569984e-8,
                4.15554391204146e-3,
            )
            self._d_map[6, 0, :] = self._horner7(
                r,
                1.67882669698981e-4,
                1.88195874702691e-4,
                8.30579218603960e-5,
                3.48461963201376e-3,
                3.61873162287129e-6,
                -1.49875789940005e-2,
                1.22979142197165e-8,
                -1.29213888778954e-3,
            )
            self._d_map[7, 0, :] = self._horner7(
                r,
                -6.22209937489143e-5,
                -6.44890425871692e-5,
                -3.02936928954918e-5,
                -1.33386143898282e-3,
                -1.31215186728213e-6,
                6.70228205200379e-3,
                -4.44653967516776e-9,
                3.15659916047599e-4,
            )
            self._d_map[8, 0, :] = self._horner7(
                r,
                6.84740881090240e-6,
                1.14082245705934e-5,
                3.03727593705750e-6,
                2.36122782444105e-4,
                1.26768491232397e-7,
                -1.53347270556276e-3,
                4.21617557752767e-10,
                -4.51948990428065e-5,
            )
            self._d_map[1, 1, :] = self._horner7(
                r,
                2.13188763071246e-6,
                -7.41025068776257e-5,
                2.31652037371554e-6,
                -2.59495924602038e-3,
                1.20637183170338e-7,
                5.21123771632193e-2,
                4.42258843694177e-10,
                -4.20967682664542e-7,
            )

    def _set_dc_map(self, c_map: np.ndarray) -> None:
        """Compute the discretised sound-speed index map.

        For large 3-D maps the naive approach allocates a full float64 copy
        and makes several sequential passes.
        GPU path: uses CuPy for the entire computation in one pass.
        CPU path: processes the first axis in chunks using a thread pool.
        """
        logger.debug("Setting dc map for stencil coefficients.")

        if self.use_gpu:
            try:
                import cupy as cp  # noqa: PLC0415

                c_gpu = cp.asarray(c_map, dtype=cp.float64)
                c_min_rounded = float(matlab_round(float(c_gpu.min())))
                offset = -c_min_rounded + 1
                c_gpu += 1e-9
                cp.rint(c_gpu, out=c_gpu)
                c_gpu += offset
                self._dc_map = cp.asnumpy(c_gpu.astype(cp.int32))
                logger.debug("dc map for stencil coefficients set (GPU).")
                return
            except ImportError:
                pass

        c_min_rounded = matlab_round(c_map.min())
        offset = float(-c_min_rounded + 1)

        result = np.empty(c_map.shape, dtype=np.int32)

        n_slabs = c_map.shape[0]
        # Each chunk is one or more slabs along axis-0.  Keep chunks large
        # enough to amortise thread overhead but small enough to fit in cache.
        max_workers = min(n_slabs, os.cpu_count() or 4)
        chunk_size = max(1, -(-n_slabs // max_workers))  # ceil division

        def _process_chunk(start: int) -> None:
            end = min(start + chunk_size, n_slabs)
            chunk = c_map[start:end].astype(np.float64)
            chunk += 1e-9
            np.rint(chunk, out=chunk)
            chunk += offset
            result[start:end] = chunk.astype(np.int32)

        if n_slabs <= chunk_size:
            # Small array - no threading overhead.
            _process_chunk(0)
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = [pool.submit(_process_chunk, s) for s in range(0, n_slabs, chunk_size)]
                for f in concurrent.futures.as_completed(futures):
                    f.result()  # propagate exceptions

        self._dc_map = result
        logger.debug("dc map for stencil coefficients set.")

    # --- batch write utils ---

    def _compute_dc_map_and_bulk_modulus_single_gpu(self) -> tuple[float, float]:
        """Compute dc_map and bulk_modulus on a single GPU.

        Returns
        -------
        tuple[float, float]
            (c_min_val, c_max_val) of the sound speed.

        """
        import cupy as cp  # noqa: PLC0415

        c_gpu = cp.asarray(self.medium.sound_speed, dtype=cp.float64)
        c_min_val = float(c_gpu.min())
        c_max_val = float(c_gpu.max())

        # dc_map: rint(sound_speed + 1e-9) - rint(c_min) + 1
        c_tmp = c_gpu.copy()
        c_min_rounded = float(matlab_round(c_min_val))
        offset = -c_min_rounded + 1
        c_tmp += 1e-9
        cp.rint(c_tmp, out=c_tmp)
        c_tmp += offset
        self._dc_map = cp.asnumpy(c_tmp.astype(cp.int32))
        del c_tmp, c_gpu

        # bulk_modulus: sound_speed^2 * density
        c_f32 = cp.asarray(self.medium.sound_speed, dtype=cp.float32)
        rho_f32 = cp.asarray(self.medium.density, dtype=cp.float32)
        bulk = cp.multiply(c_f32 * c_f32, rho_f32)
        self._precomputed_bulk_modulus = cp.asnumpy(bulk)
        del c_f32, rho_f32, bulk
        cp.get_default_memory_pool().free_all_blocks()

        logger.debug("dc map and bulk modulus set (single GPU).")
        return c_min_val, c_max_val

    def _compute_dc_map_and_bulk_modulus_multi_gpu(
        self,
        n_gpus: int,
    ) -> tuple[float, float]:
        """Compute dc_map and bulk_modulus in parallel across multiple GPUs.

        Each GPU processes a slice of the 3D array along axis 0.

        Returns
        -------
        tuple[float, float]
            (c_min_val, c_max_val) of the sound speed.

        """
        from concurrent.futures import ThreadPoolExecutor, as_completed  # noqa: PLC0415

        import cupy as cp  # noqa: PLC0415

        sound_speed_np = self.medium.sound_speed
        density_np = self.medium.density

        # Ensure numpy for slicing
        if not isinstance(sound_speed_np, np.ndarray):
            sound_speed_np = cp.asnumpy(sound_speed_np)
        if not isinstance(density_np, np.ndarray):
            density_np = cp.asnumpy(density_np)

        n_slabs = sound_speed_np.shape[0]
        n_workers = min(n_gpus, n_slabs)
        chunk_size = -(-n_slabs // n_workers)  # ceil division

        logger.info(
            "Computing dc_map and bulk_modulus using %d GPUs (of %d available).",
            n_workers,
            n_gpus,
        )

        # Phase 1+2: Each GPU computes local min/max, dc_map chunk, bulk_modulus chunk
        dc_map_result = np.empty(sound_speed_np.shape, dtype=np.int32)
        bulk_result = np.empty(sound_speed_np.shape, dtype=np.float32)
        local_mins = np.empty(n_workers, dtype=np.float64)
        local_maxs = np.empty(n_workers, dtype=np.float64)

        def _process_on_device(worker_id: int, device_id: int) -> None:
            start = worker_id * chunk_size
            end = min(start + chunk_size, n_slabs)
            with cp.cuda.Device(device_id):
                c_chunk = cp.asarray(sound_speed_np[start:end], dtype=cp.float64)
                local_mins[worker_id] = float(c_chunk.min())
                local_maxs[worker_id] = float(c_chunk.max())

                # dc_map for this chunk (offset applied after global min is known)
                c_chunk += 1e-9
                cp.rint(c_chunk, out=c_chunk)
                dc_chunk_f64 = c_chunk  # reuse buffer
                dc_map_result[start:end] = cp.asnumpy(dc_chunk_f64.astype(cp.int32))
                del dc_chunk_f64

                # bulk_modulus for this chunk
                c_f32 = cp.asarray(sound_speed_np[start:end], dtype=cp.float32)
                rho_f32 = cp.asarray(density_np[start:end], dtype=cp.float32)
                bulk_chunk = cp.multiply(c_f32 * c_f32, rho_f32)
                bulk_result[start:end] = cp.asnumpy(bulk_chunk)
                del c_f32, rho_f32, bulk_chunk
                cp.get_default_memory_pool().free_all_blocks()

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [executor.submit(_process_on_device, i, i % n_gpus) for i in range(n_workers)]
            for future in as_completed(futures):
                future.result()

        c_min_val = float(local_mins.min())
        c_max_val = float(local_maxs.max())

        # Apply global offset to dc_map
        c_min_rounded = int(matlab_round(c_min_val))
        offset = np.int32(-c_min_rounded + 1)
        dc_map_result += offset

        self._dc_map = dc_map_result
        self._precomputed_bulk_modulus = bulk_result

        logger.debug("dc map and bulk modulus set (multi-GPU, %d devices).", n_workers)
        return c_min_val, c_max_val

    def _init_pending_writes(self) -> None:
        """Initialize the pending writes list for batch I/O."""
        self._pending_writes: list[tuple] = []

    def _queue_matrix_write(
        self,
        var_type: DTypeLike,
        save_path: str | Path,
        variable_mat: np.ndarray,
    ) -> None:
        """Queue a matrix write for later batch execution."""
        self._pending_writes.append(("matrix", var_type, save_path, variable_mat))

    def _queue_v_abs_write(
        self,
        var_type: DTypeLike,
        save_path: str | Path,
        variable: NDArray[np.float64 | np.int32] | float,
    ) -> None:
        """Queue a scalar write for later batch execution."""
        self._pending_writes.append(("v_abs", var_type, save_path, variable))

    def _queue_ic_write(self, fname: str | Path, icmat: np.ndarray) -> None:
        """Queue an initial condition write for later batch execution."""
        self._pending_writes.append(("ic", fname, icmat))

    def _queue_coords_write(self, fname: str | Path, coords: np.ndarray) -> None:
        """Queue a coordinates write for later batch execution."""
        self._pending_writes.append(("coords", fname, coords))

    def _flush_writes(self) -> None:
        """Submit all pending writes to a thread pool and wait for completion."""
        pending = self._pending_writes
        if not pending:
            return

        max_workers = min(len(pending), os.cpu_count() or 4)
        logger.debug("Flushing %d pending writes with %d workers", len(pending), max_workers)

        t0 = time.perf_counter()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for item in pending:
                kind = item[0]
                if kind == "matrix":
                    _, var_type, save_path, variable_mat = item
                    futures.append(
                        executor.submit(
                            self._write_matrix,
                            var_type,
                            save_path,
                            variable_mat,
                            self.release_after_write,
                        ),
                    )
                elif kind == "v_abs":
                    _, var_type, save_path, variable = item
                    futures.append(
                        executor.submit(
                            self._write_v_abs,
                            var_type,
                            save_path,
                            variable,
                            self.release_after_write,
                        ),
                    )
                elif kind == "ic":
                    _, fname, icmat = item
                    futures.append(
                        executor.submit(
                            self._write_ic,
                            fname,
                            icmat,
                            self.release_after_write,
                        ),
                    )
                elif kind == "coords":
                    _, fname, coords = item
                    futures.append(
                        executor.submit(
                            self._write_coords,
                            fname,
                            coords,
                            self.release_after_write,
                        ),
                    )

            # Raise any exceptions from worker threads
            for future in concurrent.futures.as_completed(futures):
                future.result()

        t1 = time.perf_counter()
        logger.debug("All %d writes flushed in %.2e seconds", len(pending), t1 - t0)
        self._pending_writes.clear()

    # --- saving utils ---

    def _save_variables_into_dat_file(
        self,
        simulation_dir: Path,
        relaxation_param_map_dict_for_fw2: dict[str, NDArray[np.float64]],
        dim: int,
    ) -> None:
        k_map = (
            self._precomputed_bulk_modulus
            if self._precomputed_bulk_modulus is not None
            else self.medium.bulk_modulus
        )
        self._save_maps(
            simulation_dir,
            c_map=self.medium.sound_speed,
            k_map=k_map,
            rho_map=self.medium.density,
            beta_map=self.medium.beta,
        )
        self._save_coords(simulation_dir=simulation_dir)
        self._save_step_params(simulation_dir)
        self._save_coords_params(simulation_dir)
        self._save_d_params(simulation_dir, dim)
        self._save_sparse_grid_params(simulation_dir)

        if self.use_isotropic_relaxation:
            rename_dict = {
                "kappa_u": "kappau",
            }
            if self.write_maps_the_solver_does_not_read:
                rename_dict["kappa_x"] = "kappax"

            for nu in range(1, self.medium.n_relaxation_mechanisms + 1):
                rename_dict[f"a_pml_u{nu}"] = f"apmlu{nu}"
                rename_dict[f"b_pml_u{nu}"] = f"bpmlu{nu}"
                if nu == 1 or self.write_maps_the_solver_does_not_read:
                    rename_dict[f"a_pml_x{nu}"] = f"apmlx{nu}"
                    rename_dict[f"b_pml_x{nu}"] = f"bpmlx{nu}"
        else:
            rename_dict = {
                "kappa_x": "kappax",
                "kappa_y": "kappay",
                "kappa_u": "kappau",
                "kappa_w": "kappaw",
            }
            if self.is_3d:
                rename_dict.update(
                    {
                        "kappa_z": "kappaz",
                        "kappa_v": "kappav",
                    },
                )

            for nu in range(1, self.medium.n_relaxation_mechanisms + 1):
                rename_dict[f"a_pml_u{nu}"] = f"apmlu{nu}"
                rename_dict[f"b_pml_u{nu}"] = f"bpmlu{nu}"
                rename_dict[f"a_pml_w{nu}"] = f"apmlw{nu}"
                rename_dict[f"b_pml_w{nu}"] = f"bpmlw{nu}"
                rename_dict[f"a_pml_x{nu}"] = f"apmlx{nu}"
                rename_dict[f"b_pml_x{nu}"] = f"bpmlx{nu}"
                rename_dict[f"a_pml_y{nu}"] = f"apmly{nu}"
                rename_dict[f"b_pml_y{nu}"] = f"bpmly{nu}"
                if self.is_3d:
                    rename_dict[f"a_pml_z{nu}"] = f"apmlz{nu}"
                    rename_dict[f"b_pml_z{nu}"] = f"bpmlz{nu}"
                    rename_dict[f"a_pml_v{nu}"] = f"apmlv{nu}"
                    rename_dict[f"b_pml_v{nu}"] = f"bpmlv{nu}"

        # the n_relax kernel reads the mechanism count from this file
        self._queue_v_abs_write(
            np.int32,
            simulation_dir / "n_relax.dat",
            self.medium.n_relaxation_mechanisms,
        )

        if self.use_isotropic_relaxation:
            self._save_impedance_constraint_record(
                simulation_dir,
                relaxation_param_map_dict_for_fw2,
            )

        # save relaxation params
        for var_name, var in relaxation_param_map_dict_for_fw2.items():
            if var_name in rename_dict:
                var_name_fw2 = rename_dict[var_name]
                save_path = simulation_dir / f"{var_name_fw2}.dat"
                self._queue_matrix_write(var_type=np.float32, save_path=save_path, variable_mat=var)

    def _save_impedance_constraint_record(
        self,
        simulation_dir: Path,
        relaxation_param_map_dict_for_fw2: dict[str, NDArray[np.float64]],
    ) -> None:
        """Write the three values the n-relaxation solver reads instead of two maps.

        The impedance constraint holds the momentum stretching function at the
        identity, so kappa_x is 1 at every cell and a_pml_x above the first
        mechanism is 0 at every cell. The solver refuses a medium that does not
        carry both. Each extreme is measured on the float32 the solver reads, so
        the reading is exactly the comparison the solver makes cell by cell.

        Parameters
        ----------
        simulation_dir : Path
            Where the record is written.
        relaxation_param_map_dict_for_fw2 : dict[str, NDArray[np.float64]]
            Every relaxation coefficient map the medium carries.

        """
        kappa_x = _as_written_float32(relaxation_param_map_dict_for_fw2["kappa_x"])
        largest_kappa_x_error = float(np.abs(kappa_x - np.float32(1.0)).max())

        largest_a_pml_x_above_first = 0.0
        for nu in range(2, self.medium.n_relaxation_mechanisms + 1):
            a_pml_x = _as_written_float32(relaxation_param_map_dict_for_fw2[f"a_pml_x{nu}"])
            largest_a_pml_x_above_first = max(
                largest_a_pml_x_above_first,
                float(np.abs(a_pml_x).max()),
            )

        record = np.array(
            [
                self.medium.n_relaxation_mechanisms,
                largest_kappa_x_error,
                largest_a_pml_x_above_first,
            ],
            dtype=np.float32,
        )
        self._queue_matrix_write(
            var_type=np.float32,
            save_path=simulation_dir / "impedance_constraint.dat",
            variable_mat=record,
        )

    def _save_variables_into_dat_file_exponential_attenuation(
        self,
        simulation_dir: Path,
        dim: int,
    ) -> None:
        k_map = (
            self._precomputed_bulk_modulus
            if self._precomputed_bulk_modulus is not None
            else self.medium.bulk_modulus
        )
        self._save_maps(
            simulation_dir,
            c_map=self.medium.sound_speed,
            k_map=k_map,
            rho_map=self.medium.density,
            beta_map=self.medium.beta,
            alpha_exp_map=self.medium.alpha_exp,
        )
        self._save_coords(simulation_dir=simulation_dir)
        self._save_step_params(simulation_dir)
        self._save_coords_params(simulation_dir)
        self._save_d_params(simulation_dir, dim)
        self._save_sparse_grid_params(simulation_dir)

    def _build_symbolic_links_for_dat_files(self, src_dir: Path, dst_dir: Path) -> None:
        var_name_list = [
            "c",
            "K",
            "rho",
            "beta",
            "dX",
            "dY",
            "dT",
            "c0",
            # "icc",
            "icczero",
            "outc",
            "nY",
            "nX",
            "nT",
            "ncoords",
            "ncoords_add",
            "ncoords_u",
            "ncoords_v",
            "ncoords_w",
            "ncoordsout",
            "ncoordszero",
            "nTic",
            "modT",
            "modX",
            "modY",
            "pml_thickness",
            "exponential_attenuation_pml_thickness",
            "exponential_attenuation_pml_interior_offset",
            "n_relax",
            "d",
            "dmap",
            "ndmap",
            "dcmap",
            "kappax",
            "kappau",
            "impedance_constraint",
        ]
        mechanisms = range(1, self.medium.n_relaxation_mechanisms + 1)
        for nu in mechanisms:
            var_name_list.extend([f"apmlu{nu}", f"bpmlu{nu}", f"apmlx{nu}", f"bpmlx{nu}"])
        if not self.use_isotropic_relaxation:
            var_name_list.extend(["kappay", "kappaw"])
            for nu in mechanisms:
                var_name_list.extend([f"apmlw{nu}", f"apmly{nu}", f"bpmlw{nu}", f"bpmly{nu}"])
        if self.is_3d:
            var_name_list.append("modZ")
        if self.is_3d and not self.use_isotropic_relaxation:
            var_name_list.extend(["nZ", "dZ", "kappaz", "kappav"])
            for nu in mechanisms:
                var_name_list.extend([f"apmlz{nu}", f"apmlv{nu}", f"bpmlz{nu}", f"bpmlv{nu}"])
        for var_name in var_name_list:
            src_data = src_dir / f"{var_name}.dat"
            dst_data = dst_dir / f"{var_name}.dat"
            if src_data.exists() is False:
                continue
            # generate the symlink even if the file already exists
            if dst_data.exists():
                dst_data.unlink()
            Path(dst_data).symlink_to(src_data)

    def _save_maps(
        self,
        simulation_dir: Path,
        c_map: NDArray[np.float64],
        k_map: NDArray[np.float64],
        rho_map: NDArray[np.float64],
        beta_map: NDArray[np.float64],
        *,
        alpha_exp_map: NDArray[np.float64] | None = None,
    ) -> None:
        self._queue_matrix_write(
            var_type=np.float32,
            save_path=simulation_dir / "c.dat",
            variable_mat=c_map,
        )
        self._queue_matrix_write(
            var_type=np.float32,
            save_path=simulation_dir / "K.dat",
            variable_mat=k_map,
        )
        self._queue_matrix_write(
            var_type=np.float32,
            save_path=simulation_dir / "rho.dat",
            variable_mat=rho_map,
        )
        self._queue_matrix_write(
            var_type=np.float32,
            save_path=simulation_dir / "beta.dat",
            variable_mat=beta_map,
        )
        if alpha_exp_map is not None:
            self._queue_matrix_write(
                var_type=np.float32,
                save_path=simulation_dir / "a_exp.dat",
                variable_mat=alpha_exp_map,
            )

    def _save_coords(self, simulation_dir: Path) -> None:
        # self._write_coords(simulation_dir / "icc.dat", self.source.incoords)
        self._queue_coords_write(
            simulation_dir / "outc.dat",
            self.sensor.outcoords,
        )
        self._queue_coords_write(
            simulation_dir / "icczero.dat",
            self.medium.air_coords,
        )

        # self._write_ic(simulation_dir / "icmat.dat", np.transpose(initial_condition_mat))

    def _save_step_params(self, simulation_dir: Path) -> None:
        var_list = [
            ("dX", self.grid.dx),
            ("dY", self.grid.dy),
            ("dT", self.grid.dt),
            ("c0", self.grid.c0),
        ]
        if self.is_3d:
            var_list.extend(
                [
                    ("dZ", self.grid.dz),
                ],
            )
        for var_name, var in var_list:
            save_path = simulation_dir / f"{var_name}.dat"
            self._queue_v_abs_write(np.float32, save_path, var)

    def _save_coords_params(self, simulation_dir: Path) -> None:
        nt_ic = self.source.icmat.shape[1]
        var_list = [
            ("nX", self.grid.nx),
            ("nY", self.grid.ny),
            ("nT", self.grid.nt),
            ("ncoords", self.source.n_sources),
            ("ncoordsout", self.sensor.n_sensors),
            ("ncoordszero", self.medium.n_air),
            ("nTic", nt_ic),
            ("modT", self.sensor.sampling_modulus_time),
        ]
        n_sources_add = getattr(self.source, "n_sources_add", 0)
        if n_sources_add > 0:
            var_list.append(("ncoords_add", n_sources_add))
        for _vel_suffix in ("u", "v", "w"):
            _n_vel = getattr(self.source, f"n_sources_{_vel_suffix}", 0)
            if _n_vel > 0:
                var_list.append((f"ncoords_{_vel_suffix}", _n_vel))
        if self.is_3d:
            var_list.extend(
                [
                    ("nZ", self.grid.nz),
                ],
            )
        for var_name, var in var_list:
            save_path = simulation_dir / f"{var_name}.dat"
            self._queue_v_abs_write(np.int32, save_path, var)

    def _save_sparse_grid_params(self, simulation_dir: Path) -> None:
        """Write the sparse-grid strides, and where the interior starts.

        modX / modY / modZ are written only when the sensor was constructed with
        mod_x/mod_y, that is when ``sensor.is_sparse_grid`` is True. In standard
        coordinate or mask mode those files are not produced, which keeps older
        binaries working.

        ``pml_thickness.dat`` is written either way. It says where the interior
        starts, which is true of every run and not only of a sparse one, and a
        solver that places an absorbing layer needs it. Every shipped binary
        reads it behind a file-exists guard, and outside the sparse path no
        binary uses the value, so writing it changes no existing result.

        The C-PML absorbing layer carries its own two files, and both are
        written only when the layer is on. A binary that finds no thickness
        file places no layer, so an existing run is unchanged.
        """
        self._queue_v_abs_write(np.int32, simulation_dir / "pml_thickness.dat", self.pml_thickness)
        if self.exponential_attenuation_pml_thickness_px > 0:
            self._queue_v_abs_write(
                np.int32,
                simulation_dir / "exponential_attenuation_pml_thickness.dat",
                self.exponential_attenuation_pml_thickness_px,
            )
            self._queue_v_abs_write(
                np.int32,
                simulation_dir / "exponential_attenuation_pml_interior_offset.dat",
                self.exponential_attenuation_pml_interior_offset_px,
            )
        if not self.sensor.is_sparse_grid:
            return
        self._queue_v_abs_write(np.int32, simulation_dir / "modX.dat", self.sensor.mod_x)
        self._queue_v_abs_write(np.int32, simulation_dir / "modY.dat", self.sensor.mod_y)
        if self.is_3d:
            self._queue_v_abs_write(np.int32, simulation_dir / "modZ.dat", self.sensor.mod_z)

    def _save_d_params(
        self,
        simulation_dir: Path,
        dim: int,
    ) -> None:
        # save d and dmap
        self._queue_matrix_write(np.float32, simulation_dir / "d.dat", self._d)
        self._queue_matrix_write(np.float32, simulation_dir / "dmap.dat", self._d_map)

        # save ndmap
        ndmap = 1 if dim == 0 else self._d_map.shape[2]

        self._queue_v_abs_write(np.int32, simulation_dir / "ndmap.dat", ndmap)

        # save dcmap
        self._queue_matrix_write(
            var_type=np.int32,
            save_path=simulation_dir / "dcmap.dat",
            variable_mat=(self._dc_map - 1),
            # variable_mat=(self._dc_map),
        )

    def _copy_simulation_bin_file(self, simulation_dir: Path) -> None:
        dst = simulation_dir / self.path_fullwave_simulation_bin.name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(self.path_fullwave_simulation_bin.resolve())

    @staticmethod
    def _write_ic(
        fname: str | Path,
        icmat: np.ndarray,
        release_after_write: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        logger.debug("Writing initial condition matrix to %s", fname, stacklevel=2)

        t0 = time.perf_counter()
        out = np.ascontiguousarray(
            icmat.T,
            dtype=np.float32,
        )  # does transpose+cast into one new buffer
        out.tofile(fname)
        t1 = time.perf_counter()

        logger.debug("Initial condition matrix written in %.2e seconds", t1 - t0, stacklevel=2)
        if release_after_write:
            del icmat

    @staticmethod
    def _write_coords(
        fname: str | Path,
        coords: np.ndarray,
        release_after_write: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        logger.debug("Writing coordinates to %s", fname, stacklevel=2)

        t0 = time.perf_counter()

        # ravel() avoids an unconditional copy; ascontiguousarray(dtype=...)
        # makes one int32 C-contig buffer.
        out = np.ascontiguousarray(
            coords.ravel(),
            dtype=np.int32,
        )  # ravel is view-if-possible, ascontiguousarray makes C-order
        out.tofile(fname)  # binary write of the raw buffer

        t1 = time.perf_counter()
        logger.debug("Coordinates written in %.2e seconds", t1 - t0, stacklevel=2)
        if release_after_write:
            del coords

    @staticmethod
    def _write_v_abs(
        var_type: DTypeLike,
        save_path: str | Path,
        variable: NDArray[np.float64 | np.int32] | float,
        release_after_write: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        np.array(variable).astype(var_type).tofile(save_path)
        if release_after_write:
            del variable

    @staticmethod
    def _write_matrix(
        var_type: DTypeLike,
        save_path: str | Path,
        variable_mat: np.ndarray,
        release_after_write: bool = True,  # noqa: FBT001, FBT002
    ) -> None:
        logger.debug("Writing matrix to %s", save_path, stacklevel=2)
        t0 = time.perf_counter()

        dtype = np.dtype(var_type)

        # Transfer CuPy arrays to CPU for file writing
        if not isinstance(variable_mat, np.ndarray):
            try:
                variable_mat = variable_mat.get()  # CuPy → numpy
            except AttributeError:
                variable_mat = np.asarray(variable_mat)

        # Fast path: no conversion, no reorder.
        if variable_mat.dtype == dtype and variable_mat.flags.c_contiguous:
            variable_mat.tofile(save_path)  # writes in C order
        else:
            out = np.ascontiguousarray(variable_mat, dtype=dtype)  # at most one new buffer
            out.tofile(save_path)

        t1 = time.perf_counter()
        logger.debug("Matrix written in %.2e seconds", t1 - t0, stacklevel=2)
        # release memory by deleting the original variable_mat reference
        if release_after_write:
            del variable_mat
