"""fullwave module."""

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
import logging
import time

logging.Formatter.converter = time.gmtime
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(filename)s | %(funcName)s | %(lineno)d - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S GMT",
    level=logging.WARNING,
)


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

VERSION = "1.0.8"
__version__ = VERSION
