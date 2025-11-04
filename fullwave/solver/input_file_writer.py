"""input generator modules."""

import logging
import shutil
import time
from pathlib import Path

import numpy as np
from numpy.typing import DTypeLike, NDArray

import fullwave
from fullwave.utils import check_functions
from fullwave.utils.coordinates import map_to_coords
from fullwave.utils.numerical import matlab_round

logger = logging.getLogger("__main__." + __name__)


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

        """
        self._work_dir = Path(work_dir)
        self.path_fullwave_simulation_bin = path_fullwave_simulation_bin
        self.use_isotropic_relaxation = use_isotropic_relaxation

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

        self._dim = int(
            matlab_round(self.medium.sound_speed.max())
            - matlab_round(self.medium.sound_speed.min()),
        )

        self._set_d_mat()
        self._set_d_map(self._dim, self.medium.sound_speed)
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

        self._write_ic(
            simulation_dir / "icmat.dat",
            np.transpose(self.source.icmat),
        )
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
                self._build_symbolic_links_for_dat_files(
                    src_dir=self._work_dir,
                    dst_dir=simulation_dir,
                )
        else:
            dat_output_dir = self._work_dir if is_static_map else simulation_dir
            self._save_variables_into_dat_file_exponential_attenuation(
                simulation_dir=dat_output_dir,
                dim=self._dim,
            )
        end_file_writer_time = time.time()
        message = f"Input files generated in {end_file_writer_time - time_start:.2e} seconds."
        logger.info(message)
        return simulation_dir

    # --- constructor utils ---

    def _set_d_mat(self) -> None:
        # For 2D modeling:
        self._d = np.zeros((9, 2))
        if self.is_3d:
            self._d[1, 0] = (
                3.26627215252963e-3 * self.grid.cfl**7
                - 7.91679373564790e-4 * self.grid.cfl**6
                + 1.08663532410570e-3 * self.grid.cfl**5
                + 2.54974226454794e-2 * self.grid.cfl**4
                + 3.23083288193913e-5 * self.grid.cfl**3
                - 3.97704676886853e-1 * self.grid.cfl**2
                + 7.95584310128586e-8 * self.grid.cfl
                + 1.25425295688331
            )
            self._d[2, 0] = (
                -2.83291379048757e-3 * self.grid.cfl**7
                + 8.52796449228369e-4 * self.grid.cfl**6
                - 9.45353822586534e-4 * self.grid.cfl**5
                - 8.82015372858580e-3 * self.grid.cfl**4
                - 2.81364895458027e-5 * self.grid.cfl**3
                + 6.73021045987599e-2 * self.grid.cfl**2
                - 6.93180036837075e-8 * self.grid.cfl
                - 1.23448809066664e-1
            )
            self._d[3, 0] = (
                2.32775473203342e-3 * self.grid.cfl**7
                - 5.56793042789852e-4 * self.grid.cfl**6
                + 7.77649035879584e-4 * self.grid.cfl**5
                + 2.45547234243566e-3 * self.grid.cfl**4
                + 2.31537892801923e-5 * self.grid.cfl**3
                + 1.61900960524164e-2 * self.grid.cfl**2
                + 5.70523152308121e-8 * self.grid.cfl
                + 3.46683979649506e-2
            )
            self._d[4, 0] = (
                -1.68883462553539e-3 * self.grid.cfl**7
                + 3.03535823592644e-4 * self.grid.cfl**6
                - 5.64777117315819e-4 * self.grid.cfl**5
                + 2.44582905523866e-4 * self.grid.cfl**4
                - 1.68215579314751e-5 * self.grid.cfl**3
                - 2.62344345204941e-2 * self.grid.cfl**2
                - 4.14559953526389e-8 * self.grid.cfl
                - 1.19918511290930e-2
            )
            self._d[5, 0] = (
                1.08994931098070e-3 * self.grid.cfl**7
                - 1.41445142143525e-4 * self.grid.cfl**6
                + 3.64794490139160e-4 * self.grid.cfl**5
                - 8.86057426195227e-4 * self.grid.cfl**4
                + 1.08681882832738e-5 * self.grid.cfl**3
                + 2.07238558666603e-2 * self.grid.cfl**2
                + 2.67876079477806e-8 * self.grid.cfl
                + 4.17058420250698e-3
            )
            self._d[6, 0] = (
                -6.39950124405340e-4 * self.grid.cfl**7
                + 6.06079815415080e-5 * self.grid.cfl**6
                - 2.14633466007892e-4 * self.grid.cfl**5
                + 6.84580412267934e-4 * self.grid.cfl**4
                - 6.39907927898092e-6 * self.grid.cfl**3
                - 1.29825288653404e-2 * self.grid.cfl**2
                - 1.57775422151124e-8 * self.grid.cfl
                - 1.29998325971518e-3
            )
            self._d[7, 0] = (
                2.92716539609611e-4 * self.grid.cfl**7
                - 1.87446062803024e-5 * self.grid.cfl**6
                + 9.85389372183761e-5 * self.grid.cfl**5
                - 2.40360290348543e-4 * self.grid.cfl**4
                + 2.94166215515130e-6 * self.grid.cfl**3
                + 5.57066438452790e-3 * self.grid.cfl**2
                + 7.25741366376659e-9 * self.grid.cfl
                + 3.18698432679400e-4
            )
            self._d[8, 0] = (
                -6.42183857909518e-5 * self.grid.cfl**7
                + 3.38552867751042e-6 * self.grid.cfl**6
                - 2.17377151411164e-5 * self.grid.cfl**5
                + 4.98269067389945e-5 * self.grid.cfl**4
                - 6.50197868987757e-7 * self.grid.cfl**3
                - 1.19096089679178e-3 * self.grid.cfl**2
                - 1.60559948991172e-9 * self.grid.cfl
                - 4.57795411807702e-5
            )
            self._d[1, 1] = (
                -4.47723278782936e-5 * self.grid.cfl**7
                - 7.69502473399932e-5 * self.grid.cfl**6
                - 1.41765498250133e-5 * self.grid.cfl**5
                - 2.54672045901272e-3 * self.grid.cfl**4
                - 4.14343385915353e-7 * self.grid.cfl**3
                + 5.00210047924752e-2 * self.grid.cfl**2
                - 1.01220354410507e-9 * self.grid.cfl
                - 8.07139347787336e-8
            )
        else:
            self._d[1, 0] = (
                -0.000874634088067635 * self.grid.cfl**7
                - 0.00180530560296097 * self.grid.cfl**6
                - 0.000440512972481673 * self.grid.cfl**5
                + 0.00474018847663366 * self.grid.cfl**4
                - 1.93097802254349e-05 * self.grid.cfl**3
                - 0.292328221171893 * self.grid.cfl**2
                - 6.58101498708345e-08 * self.grid.cfl
                + 1.25420636437969
            )
            self._d[2, 0] = (
                0.000793317828964018 * self.grid.cfl**7
                + 0.00161433256585486 * self.grid.cfl**6
                + 0.000397244786277123 * self.grid.cfl**5
                + 0.00546057645976549 * self.grid.cfl**4
                + 1.73781972873916e-05 * self.grid.cfl**3
                + 0.0588754971188371 * self.grid.cfl**2
                + 5.91706982879834e-08 * self.grid.cfl
                - 0.123406473759703
            )
            self._d[3, 0] = (
                -0.000650217700538851 * self.grid.cfl**7
                - 0.00116449260340413 * self.grid.cfl**6
                - 0.000324403734066325 * self.grid.cfl**5
                - 0.00911483710059994 * self.grid.cfl**4
                - 1.417399823126e-05 * self.grid.cfl**3
                + 0.0233184077551615 * self.grid.cfl**2
                - 4.82326094707544e-08 * self.grid.cfl
                + 0.0346342451534453
            )
            self._d[4, 0] = (
                0.000467529510541428 * self.grid.cfl**7
                + 0.000732736676632388 * self.grid.cfl**6
                + 0.000232444388955328 * self.grid.cfl**5
                + 0.00846419766685254 * self.grid.cfl**4
                + 1.01438593426278e-05 * self.grid.cfl**3
                - 0.0317586249260511 * self.grid.cfl**2
                + 3.44988852042879e-08 * self.grid.cfl
                - 0.0119674942518101
            )
            self._d[5, 0] = (
                -0.000298416281187033 * self.grid.cfl**7
                - 0.000399380750669364 * self.grid.cfl**6
                - 0.000148203388388213 * self.grid.cfl**5
                - 0.00601788793192501 * self.grid.cfl**4
                - 6.46543538517443e-06 * self.grid.cfl**3
                + 0.0241912754935119 * self.grid.cfl**2
                - 2.19855171569984e-08 * self.grid.cfl
                + 0.00415554391204146
            )
            self._d[6, 0] = (
                0.000167882669698981 * self.grid.cfl**7
                + 0.000188195874702691 * self.grid.cfl**6
                + 8.3057921860396e-05 * self.grid.cfl**5
                + 0.00348461963201376 * self.grid.cfl**4
                + 3.61873162287129e-06 * self.grid.cfl**3
                - 0.0149875789940005 * self.grid.cfl**2
                + 1.22979142197165e-08 * self.grid.cfl
                - 0.00129213888778954
            )
            self._d[7, 0] = (
                -6.22209937489143e-05 * self.grid.cfl**7
                - 6.44890425871692e-05 * self.grid.cfl**6
                - 3.02936928954918e-05 * self.grid.cfl**5
                - 0.00133386143898282 * self.grid.cfl**4
                - 1.31215186728213e-06 * self.grid.cfl**3
                + 0.00670228205200379 * self.grid.cfl**2
                - 4.44653967516776e-09 * self.grid.cfl
                + 0.000315659916047599
            )
            self._d[8, 0] = (
                6.8474088109024e-06 * self.grid.cfl**7
                + 1.14082245705934e-05 * self.grid.cfl**6
                + 3.0372759370575e-06 * self.grid.cfl**5
                + 0.000236122782444105 * self.grid.cfl**4
                + 1.26768491232397e-07 * self.grid.cfl**3
                - 0.00153347270556276 * self.grid.cfl**2
                + 4.21617557752767e-10 * self.grid.cfl
                - 4.51948990428065e-05
            )
            self._d[1, 1] = (
                2.13188763071246e-06 * self.grid.cfl**7
                - 7.41025068776257e-05 * self.grid.cfl**6
                + 2.31652037371554e-06 * self.grid.cfl**5
                - 0.00259495924602038 * self.grid.cfl**4
                + 1.20637183170338e-07 * self.grid.cfl**3
                + 0.0521123771632193 * self.grid.cfl**2
                + 4.42258843694177e-10 * self.grid.cfl
                - 4.20967682664542e-07
            )

    def _set_d_map(self, dim: int, c_map: NDArray[np.float64]) -> None:
        self._d_map = np.zeros((9, 2, dim + 1))
        if self.is_3d:
            for i in range(dim + 1):
                r_d_map = (i + c_map.min()) * self.grid.dt / self.grid.dx
                self._d_map[1, 0, i] = (
                    3.26627215252963e-3 * r_d_map**7
                    - 7.91679373564790e-4 * r_d_map**6
                    + 1.08663532410570e-3 * r_d_map**5
                    + 2.54974226454794e-2 * r_d_map**4
                    + 3.23083288193913e-5 * r_d_map**3
                    - 3.97704676886853e-1 * r_d_map**2
                    + 7.95584310128586e-8 * r_d_map
                    + 1.25425295688331
                )
                self._d_map[2, 0, i] = (
                    -2.83291379048757e-3 * r_d_map**7
                    + 8.52796449228369e-4 * r_d_map**6
                    - 9.45353822586534e-4 * r_d_map**5
                    - 8.82015372858580e-3 * r_d_map**4
                    - 2.81364895458027e-5 * r_d_map**3
                    + 6.73021045987599e-2 * r_d_map**2
                    - 6.93180036837075e-8 * r_d_map
                    - 1.23448809066664e-1
                )
                self._d_map[3, 0, i] = (
                    2.32775473203342e-3 * r_d_map**7
                    - 5.56793042789852e-4 * r_d_map**6
                    + 7.77649035879584e-4 * r_d_map**5
                    + 2.45547234243566e-3 * r_d_map**4
                    + 2.31537892801923e-5 * r_d_map**3
                    + 1.61900960524164e-2 * r_d_map**2
                    + 5.70523152308121e-8 * r_d_map
                    + 3.46683979649506e-2
                )
                self._d_map[4, 0, i] = (
                    -1.68883462553539e-3 * r_d_map**7
                    + 3.03535823592644e-4 * r_d_map**6
                    - 5.64777117315819e-4 * r_d_map**5
                    + 2.44582905523866e-4 * r_d_map**4
                    - 1.68215579314751e-5 * r_d_map**3
                    - 2.62344345204941e-2 * r_d_map**2
                    - 4.14559953526389e-8 * r_d_map
                    - 1.19918511290930e-2
                )
                self._d_map[5, 0, i] = (
                    1.08994931098070e-3 * r_d_map**7
                    - 1.41445142143525e-4 * r_d_map**6
                    + 3.64794490139160e-4 * r_d_map**5
                    - 8.86057426195227e-4 * r_d_map**4
                    + 1.08681882832738e-5 * r_d_map**3
                    + 2.07238558666603e-2 * r_d_map**2
                    + 2.67876079477806e-8 * r_d_map
                    + 4.17058420250698e-3
                )
                self._d_map[6, 0, i] = (
                    -6.39950124405340e-4 * r_d_map**7
                    + 6.06079815415080e-5 * r_d_map**6
                    - 2.14633466007892e-4 * r_d_map**5
                    + 6.84580412267934e-4 * r_d_map**4
                    - 6.39907927898092e-6 * r_d_map**3
                    - 1.29825288653404e-2 * r_d_map**2
                    - 1.57775422151124e-8 * r_d_map
                    - 1.29998325971518e-3
                )
                self._d_map[7, 0, i] = (
                    2.92716539609611e-4 * r_d_map**7
                    - 1.87446062803024e-5 * r_d_map**6
                    + 9.85389372183761e-5 * r_d_map**5
                    - 2.40360290348543e-4 * r_d_map**4
                    + 2.94166215515130e-6 * r_d_map**3
                    + 5.57066438452790e-3 * r_d_map**2
                    + 7.25741366376659e-9 * r_d_map
                    + 3.18698432679400e-4
                )
                self._d_map[8, 0, i] = (
                    -6.42183857909518e-5 * r_d_map**7
                    + 3.38552867751042e-6 * r_d_map**6
                    - 2.17377151411164e-5 * r_d_map**5
                    + 4.98269067389945e-5 * r_d_map**4
                    - 6.50197868987757e-7 * r_d_map**3
                    - 1.19096089679178e-3 * r_d_map**2
                    - 1.60559948991172e-9 * r_d_map
                    - 4.57795411807702e-5
                )
                self._d_map[1, 1, i] = (
                    -4.47723278782936e-5 * r_d_map**7
                    - 7.69502473399932e-5 * r_d_map**6
                    - 1.41765498250133e-5 * r_d_map**5
                    - 2.54672045901272e-3 * r_d_map**4
                    - 4.14343385915353e-7 * r_d_map**3
                    + 5.00210047924752e-2 * r_d_map**2
                    - 1.01220354410507e-9 * r_d_map
                    - 8.07139347787336e-8
                )
        else:
            for i in range(dim + 1):
                r_d_map = (i + c_map.min()) * self.grid.dt / self.grid.dx
                self._d_map[1, 0, i] = (
                    -0.000874634088067635 * r_d_map**7
                    - 0.00180530560296097 * r_d_map**6
                    - 0.000440512972481673 * r_d_map**5
                    + 0.00474018847663366 * r_d_map**4
                    - 1.93097802254349e-05 * r_d_map**3
                    - 0.292328221171893 * r_d_map**2
                    - 6.58101498708345e-08 * r_d_map
                    + 1.25420636437969
                )
                self._d_map[2, 0, i] = (
                    0.000793317828964018 * r_d_map**7
                    + 0.00161433256585486 * r_d_map**6
                    + 0.000397244786277123 * r_d_map**5
                    + 0.00546057645976549 * r_d_map**4
                    + 1.73781972873916e-05 * r_d_map**3
                    + 0.0588754971188371 * r_d_map**2
                    + 5.91706982879834e-08 * r_d_map
                    - 0.123406473759703
                )
                self._d_map[3, 0, i] = (
                    -0.000650217700538851 * r_d_map**7
                    - 0.00116449260340413 * r_d_map**6
                    - 0.000324403734066325 * r_d_map**5
                    - 0.00911483710059994 * r_d_map**4
                    - 1.417399823126e-05 * r_d_map**3
                    + 0.0233184077551615 * r_d_map**2
                    - 4.82326094707544e-08 * r_d_map
                    + 0.0346342451534453
                )
                self._d_map[4, 0, i] = (
                    0.000467529510541428 * r_d_map**7
                    + 0.000732736676632388 * r_d_map**6
                    + 0.000232444388955328 * r_d_map**5
                    + 0.00846419766685254 * r_d_map**4
                    + 1.01438593426278e-05 * r_d_map**3
                    - 0.0317586249260511 * r_d_map**2
                    + 3.44988852042879e-08 * r_d_map
                    - 0.0119674942518101
                )
                self._d_map[5, 0, i] = (
                    -0.000298416281187033 * r_d_map**7
                    - 0.000399380750669364 * r_d_map**6
                    - 0.000148203388388213 * r_d_map**5
                    - 0.00601788793192501 * r_d_map**4
                    - 6.46543538517443e-06 * r_d_map**3
                    + 0.0241912754935119 * r_d_map**2
                    - 2.19855171569984e-08 * r_d_map
                    + 0.00415554391204146
                )
                self._d_map[6, 0, i] = (
                    0.000167882669698981 * r_d_map**7
                    + 0.000188195874702691 * r_d_map**6
                    + 8.3057921860396e-05 * r_d_map**5
                    + 0.00348461963201376 * r_d_map**4
                    + 3.61873162287129e-06 * r_d_map**3
                    - 0.0149875789940005 * r_d_map**2
                    + 1.22979142197165e-08 * r_d_map
                    - 0.00129213888778954
                )
                self._d_map[7, 0, i] = (
                    -6.22209937489143e-05 * r_d_map**7
                    - 6.44890425871692e-05 * r_d_map**6
                    - 3.02936928954918e-05 * r_d_map**5
                    - 0.00133386143898282 * r_d_map**4
                    - 1.31215186728213e-06 * r_d_map**3
                    + 0.00670228205200379 * r_d_map**2
                    - 4.44653967516776e-09 * r_d_map
                    + 0.000315659916047599
                )
                self._d_map[8, 0, i] = (
                    6.8474088109024e-06 * r_d_map**7
                    + 1.14082245705934e-05 * r_d_map**6
                    + 3.0372759370575e-06 * r_d_map**5
                    + 0.000236122782444105 * r_d_map**4
                    + 1.26768491232397e-07 * r_d_map**3
                    - 0.00153347270556276 * r_d_map**2
                    + 4.21617557752767e-10 * r_d_map
                    - 4.51948990428065e-05
                )
                self._d_map[1, 1, i] = (
                    2.13188763071246e-06 * r_d_map**7
                    - 7.41025068776257e-05 * r_d_map**6
                    + 2.31652037371554e-06 * r_d_map**5
                    - 0.00259495924602038 * r_d_map**4
                    + 1.20637183170338e-07 * r_d_map**3
                    + 0.0521123771632193 * r_d_map**2
                    + 4.42258843694177e-10 * r_d_map
                    - 4.20967682664542e-07
                )

    def _set_dc_map(self, c_map: NDArray[np.float64]) -> None:
        self._dc_map = matlab_round(c_map) - matlab_round(c_map.min()) + 1

    # --- saving utils ---

    def _save_variables_into_dat_file(
        self,
        simulation_dir: Path,
        relaxation_param_map_dict_for_fw2: dict[str, NDArray[np.float64]],
        dim: int,
    ) -> None:
        self._save_maps(
            simulation_dir,
            c_map=self.medium.sound_speed,
            k_map=self.medium.bulk_modulus,
            rho_map=self.medium.density,
            beta_map=self.medium.beta,
        )
        self._save_coords(simulation_dir=simulation_dir)
        self._save_step_params(simulation_dir)
        self._save_coords_params(simulation_dir)
        self._save_d_params(simulation_dir, dim)

        if self.use_isotropic_relaxation:
            rename_dict = {
                "kappa_x": "kappax",
                "kappa_u": "kappau",
            }

            for nu in range(1, self.medium.n_relaxation_mechanisms + 1):
                rename_dict[f"a_pml_u{nu}"] = f"apmlu{nu}"
                rename_dict[f"b_pml_u{nu}"] = f"bpmlu{nu}"
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

        # save relaxation params
        for var_name, var in relaxation_param_map_dict_for_fw2.items():
            if var_name in rename_dict:
                var_name_fw2 = rename_dict[var_name]
                save_path = simulation_dir / f"{var_name_fw2}.dat"
                self._write_matrix(var_type=np.float32, save_path=save_path, variable_mat=var)

    def _save_variables_into_dat_file_exponential_attenuation(
        self,
        simulation_dir: Path,
        dim: int,
    ) -> None:
        self._save_maps(
            simulation_dir,
            c_map=self.medium.sound_speed,
            k_map=self.medium.bulk_modulus,
            rho_map=self.medium.density,
            beta_map=self.medium.beta,
            alpha_exp_map=self.medium.alpha_exp,
        )
        self._save_coords(simulation_dir=simulation_dir)
        self._save_step_params(simulation_dir)
        self._save_coords_params(simulation_dir)
        self._save_d_params(simulation_dir, dim)

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
            "icc",
            "outc",
            "nY",
            "nX",
            "nT",
            "ncoords",
            "ncoordsout",
            "nTic",
            "modT",
            "d",
            "dmap",
            "ndmap",
            "dcmap",
            "kappax",
            "kappay",
            "kappau",
            "kappaw",
            "apmlu1",
            "bpmlu1",
            "apmlw1",
            "bpmlw1",
            "apmlx1",
            "bpmlx1",
            "apmly1",
            "bpmly1",
            "apmlu2",
            "bpmlu2",
            "apmlw2",
            "bpmlw2",
            "apmlx2",
            "bpmlx2",
            "apmly2",
            "bpmly2",
        ]
        if self.is_3d:
            var_name_list.extend(
                [
                    "nZ",
                    "dZ",
                    "kappaz",
                    "kappav",
                    "apmlz1",
                    "bpmlz1",
                    "apmlv1",
                    "bpmlv1",
                    "apmlz2",
                    "bpmlz2",
                    "apmlv2",
                    "bpmlv2",
                ],
            )
        for var_name in var_name_list:
            src_data = src_dir / f"{var_name}.dat"
            dst_data = dst_dir / f"{var_name}.dat"
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
        self._write_matrix(
            var_type=np.float32,
            save_path=simulation_dir / "c.dat",
            variable_mat=c_map,
        )
        self._write_matrix(
            var_type=np.float32,
            save_path=simulation_dir / "K.dat",
            variable_mat=k_map,
        )
        self._write_matrix(
            var_type=np.float32,
            save_path=simulation_dir / "rho.dat",
            variable_mat=rho_map,
        )
        self._write_matrix(
            var_type=np.float32,
            save_path=simulation_dir / "beta.dat",
            variable_mat=beta_map,
        )
        if alpha_exp_map is not None:
            self._write_matrix(
                var_type=np.float32,
                save_path=simulation_dir / "a_exp.dat",
                variable_mat=alpha_exp_map,
            )

    def _save_coords(self, simulation_dir: Path) -> None:
        self._write_coords(simulation_dir / "icc.dat", self.source.incoords)
        self._write_coords(
            simulation_dir / "outc.dat",
            self.sensor.outcoords,
        )
        self._write_coords(
            simulation_dir / "icczero.dat",
            map_to_coords(self.medium.air_map),
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
            self._write_v_abs(np.float32, save_path, var)

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
        if self.is_3d:
            var_list.extend(
                [
                    ("nZ", self.grid.nz),
                ],
            )
        for var_name, var in var_list:
            save_path = simulation_dir / f"{var_name}.dat"
            self._write_v_abs(np.int32, save_path, var)

    def _save_d_params(self, simulation_dir: Path, dim: int) -> None:
        # save d and dmap
        self._write_matrix(np.float32, simulation_dir / "d.dat", self._d)
        self._write_matrix(np.float32, simulation_dir / "dmap.dat", self._d_map)

        # save ndmap
        ndmap = 1 if dim == 0 else self._d_map.shape[2]

        self._write_v_abs(np.int32, simulation_dir / "ndmap.dat", ndmap)

        # save dcmap
        self._write_matrix(
            var_type=np.int32,
            save_path=simulation_dir / "dcmap.dat",
            variable_mat=(self._dc_map - 1),
            # variable_mat=(self._dc_map),
        )

    def _copy_simulation_bin_file(self, simulation_dir: Path) -> None:
        shutil.copy(
            src=self.path_fullwave_simulation_bin,
            dst=simulation_dir / self.path_fullwave_simulation_bin.name,
        )

    @staticmethod
    def _write_ic(fname: str | Path, icmat: NDArray[np.float64]) -> None:
        message = f"Writing initial condition matrix to {fname}"
        logger.debug(message, stacklevel=2)

        start_time = time.time()
        icmat.T.flatten().astype(np.float32).tofile(fname)
        end_time = time.time()

        message = f"Initial condition matrix written in {end_time - start_time:.2e} seconds"
        logger.debug(message, stacklevel=2)

    @staticmethod
    def _write_coords(
        fname: str | Path,
        coords: NDArray[np.float64 | np.int64],
        # *,
        # swap_ij: bool = False,
    ) -> None:
        # if swap_ij:
        #     np.array([coords[:, 1], coords[:, 0]]).T.flatten().astype(np.int32).tofile(fname)
        # else:
        #     coords.T.flatten().astype(np.int32).tofile(fname)

        message = f"Writing coordinates to {fname}"
        logger.debug(message, stacklevel=2)
        time_start = time.time()
        coords.flatten().astype(np.int32).tofile(fname)
        time_end = time.time()
        message = f"Coordinates written in {time_end - time_start:.2e} seconds"
        logger.debug(message, stacklevel=2)

    @staticmethod
    def _write_v_abs(
        var_type: DTypeLike,
        save_path: str | Path,
        variable: NDArray[np.float64 | np.int32] | float,
    ) -> None:
        np.array(variable).astype(var_type).tofile(save_path)

    @staticmethod
    def _write_matrix(
        var_type: DTypeLike,
        save_path: str | Path,
        variable_mat: NDArray[np.float64],
    ) -> None:
        message = f"Writing matrix to {save_path}"
        logger.debug(message, stacklevel=2)
        start_time = time.time()
        variable_mat.astype(var_type).tofile(save_path)
        end_time = time.time()
        message = f"Matrix written in {end_time - start_time:.2e} seconds"
        logger.debug(message, stacklevel=2)
