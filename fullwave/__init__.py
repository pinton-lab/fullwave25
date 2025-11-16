"""fullwave module."""

import importlib.metadata
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

# Versioning only after package is fully installed

__version__ = "unknown"

try:
    # First, try getting version from installed package metadata
    __version__ = importlib.metadata.version(__package__ or __name__)
except importlib.metadata.PackageNotFoundError:
    # Fallback: read from pyproject.toml for local development
    try:
        import tomllib  # Python 3.11+
    except ModuleNotFoundError:
        import tomli as tomllib  # For Python < 3.11, requires install tomli

    from pathlib import Path

    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    if pyproject_path.exists():
        with Path(pyproject_path).open("rb") as f:
            pyproject_data = tomllib.load(f)
            __version__ = pyproject_data["project"]["version"]
