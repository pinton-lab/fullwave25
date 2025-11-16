"""fullwave module."""

import logging
import platform
import time

from . import utils
from .grid import Grid
from .medium import Medium, MediumExponentialAttenuation, MediumRelaxationMaps
from .sensor import Sensor
from .source import Source
from .transducer import Transducer, TransducerGeometry

from .medium_builder import presets  # isort:skip

from .solver.solver import Solver  # isort:skip
from .medium_builder.domain import Domain  # isort:skip
from .medium_builder import MediumBuilder  # isort:skip

logging.Formatter.converter = time.gmtime
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s | %(funcName)s | %(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S GMT",
    level=logging.INFO,
)

logger = logging.getLogger("__main__." + __name__)

try:
    from importlib.metadata import version

    __version__ = version("fullwave")
except ImportError:
    # Fallback for development/testing when package is not installed
    from pathlib import Path

    import tomli

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    with Path(pyproject_path).open("rb") as f:
        pyproject_data = tomli.load(f)
    __version__ = pyproject_data["project"]["version"]

VERSION = __version__


# "FullwaveSolver",
__all__ = [
    "Domain",
    "Grid",
    "Medium",
    "MediumBuilder",
    "MediumExponentialAttenuation",
    "MediumRelaxationMaps",
    "Sensor",
    "Solver",
    "Source",
    "Transducer",
    "TransducerGeometry",
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
