"""fullwave module."""

import logging

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

# check linux environment
import platform

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

if platform.system() != "Linux":
    message = (
        "Warning: fullwave is primarily developed for Linux environment.\n"
        "Using it on other operating systems may lead to unexpected issues.\n"
        "Please consider using WSL2 (Windows Subsystem for Linux 2) if you are on Windows."
    )
    logger.warning(
        message,
    )
del platform

VERSION = "1.0.8"
__version__ = VERSION
