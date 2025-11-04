"""Module for launching Fullwave simulation."""

import logging
import os
import subprocess
from pathlib import Path
from time import time

import numpy as np
from numpy.typing import NDArray

from .utils import load_dat_data

logger = logging.getLogger("__main__." + __name__)


class SimulationError(Exception):
    """Exception raised for errors in the simulation."""


class Launcher:
    """Launcher class for Fullwave simulation."""

    def __init__(
        self,
        path_fullwave_simulation_bin: Path = Path(__file__).parent / "bins" / "fullwave_solver_gpu",
        *,
        is_3d: bool = False,
        use_gpu: bool = True,
    ) -> None:
        """Initialize a FullwaveLauncher instance.

        Parameters
        ----------
        path_fullwave_simulation_bin : Path, optional
            The fullwave simulation binary path.
            Defaults to Path(__file__).parent / "bins" / "fullwave_solver_gpu".
        is_3d : bool, optional
            Whether the simulation is 3D or not.
            Defaults to False. If True, the simulation will be run in 3D mode.
        use_gpu : bool, optional
            Whether to use GPU for the simulation.
            Defaults to True. If False, the simulation will be run on multi-core CPU version.

        """
        self._path_fullwave_simulation_bin = path_fullwave_simulation_bin
        error_msg = f"Fullwave simulation binary not found at {self._path_fullwave_simulation_bin}"
        assert self._path_fullwave_simulation_bin.exists(), error_msg
        self.is_3d = is_3d
        self.use_gpu = use_gpu
        logger.debug("Launcher instance created.")

    def run(
        self,
        simulation_dir: Path,
        *,
        load_results: bool = True,
    ) -> NDArray[np.float64] | Path:
        """Run the simulation and return the results loaded from genout.dat.

        Parameters
        ----------
        simulation_dir : Path
            The directory where the simulation will be run.
            The directory should contain the necessary input files for the simulation.
        load_results : bool
            Whether to load the results from genout.dat after the simulation.
            Default is True. If set to False, it returns the genout.dat file path instead.

        Returns
        -------
        NDArray[np.float64]
            The array containing simulation results loaded from 'genout.dat'.

        Raises
        ------
        SimulationError
            If the simulation fails and an error occurs during execution.

        """
        home_dir = Path.cwd()
        simulation_dir = simulation_dir.absolute()

        if not self.use_gpu:
            message = "Currently, only GPU version is supported."
            logger.error(message)
            raise NotImplementedError(message)

        os.chdir(simulation_dir)
        try:
            command = [
                "stdbuf",
                "-oL",
                str(self._path_fullwave_simulation_bin.resolve()),
            ]
            logger.info("Running simulation...")
            with (simulation_dir / "fw2_execution.log").open("w", encoding="utf-8") as file:
                time_start = time()
                subprocess.run(  # noqa: S603
                    command,
                    check=True,
                    shell=False,
                    stdout=file,
                    stderr=file,
                    text=True,
                    # check=False,
                )
                time_passed = time() - time_start
                message = f"Simulation completed in {time_passed:.2e} seconds."
                logger.info(message)

            os.chdir(home_dir)
        except Exception as e:
            os.chdir(home_dir)
            logger.exception("Simulation failed")
            # load error message from log file

            with (simulation_dir / "fw2_execution.log").open("r", encoding="utf-8") as file:
                error_message_fw2 = file.read()
                logger.exception(
                    "--- Simulation: fw2_execution log start ---\n"
                    "%s\n"
                    "--- Simulation: fw2_execution log end ---\n",
                    error_message_fw2,
                )

            error_message = (
                "Simulation failed. please check the simulation log file for more information.\n"
                "The log file is located at:\n"
                f"{simulation_dir / 'fw2_execution.log'}"
            )
            logger.exception(error_message)
            raise SimulationError(error_message) from e

        if load_results:
            time_load_start = time()
            logger.info("Loading simulation results from genout.dat...")

            result = load_dat_data(simulation_dir.absolute() / "genout.dat")

            time_load_passed = time() - time_load_start
            logger.info("Loading completed in %.2e seconds.", time_load_passed)
            return result

        logger.info("Returning genout.dat file path.")
        return simulation_dir.absolute() / "genout.dat"
