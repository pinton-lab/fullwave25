"""fullwave module."""

import logging
import os
import platform
import time
from importlib.metadata import PackageNotFoundError, version

import numexpr

from .grid import Grid
from .medium import Medium, MediumExponentialAttenuation, MediumRelaxationMaps
from .sensor import Sensor
from .source import Source
from .transducer import Transducer, TransducerGeometry, TransducerStack
from .transmit import Pulse

from .medium_builder import presets  # isort:skip

from .solver.solver import Solver  # isort:skip
from .medium_builder.domain import Domain  # isort:skip
from .medium_builder import MediumBuilder  # isort:skip
from . import utils  # isort:skip

logging.Formatter.converter = time.gmtime
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s | %(funcName)s | %(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S GMT",
    level=logging.INFO,
)

logger = logging.getLogger("__main__." + __name__)

# "FullwaveSolver",
__all__ = [
    "Domain",
    "Grid",
    "Medium",
    "MediumBuilder",
    "MediumExponentialAttenuation",
    "MediumRelaxationMaps",
    "Pulse",
    "Sensor",
    "Solver",
    "Source",
    "Transducer",
    "TransducerGeometry",
    "TransducerStack",
    "presets",
    "utils",
]

PLATFORM = platform.system().lower()
# check linux environment
if PLATFORM != "linux":
    message = (
        "Warning: fullwave is primarily developed for Linux environment.\n"
        "Using it on other operating systems may lead to unexpected issues.\n"
        "Please consider using WSL2 (Windows Subsystem for Linux 2) if you are on Windows."
    )
    logger.warning(
        message,
    )

try:
    __version__ = version("fullwave")
except PackageNotFoundError:
    # Update via bump-my-version, not manually
    __version__ = "1.3.2-dev4"

VERSION = __version__  # for convenience
logger.info("Fullwave version: %s", __version__)

# numexpr keeps 16 threads unless told otherwise, and the medium, the absorbing
# layer and the coefficients run through it on grids of hundreds of millions of
# cells. A thread count set in the environment is left alone.
if not any(
    os.environ.get(name)
    for name in ("NUMEXPR_MAX_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS")
):
    numexpr.set_num_threads(min(numexpr.detect_number_of_cores(), numexpr.MAX_THREADS))
logger.info("numexpr threads: %d", numexpr.get_num_threads())
