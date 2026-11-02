"""domain presets."""

from .domain_abdominal_wall import AbdominalWallDomain
from .domain_background import BackgroundDomain
from .domain_combined_phantom import CombinedPhantomDomain
from .domain_scatterer import ScattererDomain
from .domain_simple import SimpleDomain

__all__ = [
    "AbdominalWallDomain",
    "BackgroundDomain",
    "CombinedPhantomDomain",
    "ScattererDomain",
    "SimpleDomain",
]
