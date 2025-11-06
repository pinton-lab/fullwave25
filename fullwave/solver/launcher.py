"""Module for launching Fullwave simulation."""

import logging
import os
import re
import shutil
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
        cuda_device_id: str | int | list | None = None,
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
        cuda_device_id : str | int | list | None, optional
            The CUDA device ID(s) to use for the simulation.
            Defaults to None. If None, the default device ID "0" will be used.
            for multiple GPUs, provide a list of device IDs.
            example 1: [0, 1] for using GPU 0 and GPU 1. or "0,1" as a string.
            example 2: 2 for using GPU 2 or "2" as a string.

        """
        self._path_fullwave_simulation_bin = path_fullwave_simulation_bin
        error_msg = f"Fullwave simulation binary not found at {self._path_fullwave_simulation_bin}"
        assert self._path_fullwave_simulation_bin.exists(), error_msg
        self.is_3d = is_3d
        self.use_gpu = use_gpu
        self.cuda_device_id = self._configure_cuda_device_id(cuda_device_id)
        logger.debug("Launcher instance created.")

    @staticmethod
    def _parse_cuda_device_id(cuda_device_id: str | int | list | None) -> str:
        """Parse the CUDA device ID input into a string format.

        Parameters
        ----------
        cuda_device_id : str | int | list | None
            The CUDA device ID to parse.

        Returns
        -------
        str
            The parsed CUDA device ID in string format.

        Raises
        ------
        ValueError
            If the input type is invalid or contains invalid values.

        """
        if cuda_device_id is None:
            return "0"

        if isinstance(cuda_device_id, int):
            if cuda_device_id < 0:
                message = "CUDA device ID must be a non-negative integer."
                raise ValueError(message)
            return str(cuda_device_id)

        if isinstance(cuda_device_id, str):
            if not cuda_device_id.isdigit() or int(cuda_device_id) < 0:
                message = "CUDA device ID string must represent a non-negative integer."
                raise ValueError(message)
            return cuda_device_id

        if isinstance(cuda_device_id, list):
            if not all(isinstance(i, int) and i >= 0 for i in cuda_device_id):
                message = "All CUDA device IDs in the list must be non-negative integers."
                raise ValueError(message)
            return ",".join(str(i) for i in cuda_device_id)

        message = "CUDA device ID must be an integer, string, list, or None."
        raise ValueError(message)

    @staticmethod
    def _verify_cuda_devices_exist(device_id_str: str) -> None:
        """Verify that the specified CUDA devices exist on the system.

        Parameters
        ----------
        device_id_str : str
            The CUDA device ID(s) in string format.

        Raises
        ------
        ValueError
            If any of the specified CUDA device IDs do not exist.

        """
        nvidia_smi_path = shutil.which("nvidia-smi")
        if nvidia_smi_path is None:
            message = "nvidia-smi command not found. Please ensure NVIDIA drivers are installed."
            raise ValueError(message)

        result = subprocess.run(  # noqa: S603
            [nvidia_smi_path, "-L"],
            check=False,
            stdout=subprocess.PIPE,
            encoding="utf-8",
            shell=False,
        )

        gpu_ids = re.findall(r"GPU (\d+):", result.stdout)
        for device_id in device_id_str.split(","):
            if device_id not in gpu_ids:
                message = f"CUDA device ID {device_id} does not exist."
                raise ValueError(message)

    @staticmethod
    def _configure_cuda_device_id(cuda_device_id: str | int | list | None) -> str:
        """Verify and assign the CUDA device ID.

        Parameters
        ----------
        cuda_device_id : str | int | None
            The CUDA device ID to verify and assign.

        Returns
        -------
        str
            The verified and assigned CUDA device ID.

        """
        output = Launcher._parse_cuda_device_id(cuda_device_id)
        Launcher._verify_cuda_devices_exist(output)
        return output

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
                os.environ["CUDA_VISIBLE_DEVICES"] = self.cuda_device_id
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
