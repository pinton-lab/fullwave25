"""Material properties."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass
class MaterialProperties:
    """Material properties for different tissues."""

    fat: ClassVar[dict] = {
        "b_on_a": 9.6,
        "alpha_coeff": 0.4,
        "alpha_power": 1.1,
        "sound_speed": 1412,
        "density": 937,
    }
    fat["beta"] = 1 + fat["b_on_a"] / 2

    liver: ClassVar[dict] = {
        "b_on_a": 7.6,
        "alpha_coeff": 0.5,
        "alpha_power": 1.1,
        "sound_speed": 1566.03,
        "density": 1064,
    }
    liver["beta"] = 1 + liver["b_on_a"] / 2

    muscle: ClassVar[dict] = {
        "b_on_a": 9,
        "alpha_coeff": 0.15,
        "alpha_power": 1.0,
        "sound_speed": 1527,
        "density": 1070,
    }
    muscle["beta"] = 1 + muscle["b_on_a"] / 2

    water: ClassVar[dict] = {
        "b_on_a": 5,
        "alpha_coeff": 0.005,
        "alpha_power": 2.0,
        "sound_speed": 1523,
        "density": 1000,
    }
    water["beta"] = 1 + water["b_on_a"] / 2

    skin: ClassVar[dict] = {
        "b_on_a": 8,
        "alpha_coeff": 2.1,
        "alpha_power": 1,
        "sound_speed": 1772,
        "density": 1090,
    }
    skin["beta"] = 1 + skin["b_on_a"] / 2

    tissue: ClassVar[dict] = {
        "b_on_a": 9,
        "alpha_coeff": 0.5,
        "alpha_power": 1,
        "sound_speed": 1540,
        "density": 1000,
    }
    tissue["beta"] = 1 + tissue["b_on_a"] / 2

    connective: ClassVar[dict] = {
        "b_on_a": 8,
        "alpha_coeff": 0.5,
        "alpha_power": 1,
        "sound_speed": 1613,
        "density": 1120,
    }
    connective["beta"] = 1 + connective["b_on_a"] / 2

    blood: ClassVar[dict] = {
        "b_on_a": 5,
        "alpha_coeff": 0.005,
        "alpha_power": 2.0,
        "sound_speed": 1583,
        "density": 1000,
    }
    blood["beta"] = 1 + blood["b_on_a"] / 2

    lung_fluid: ClassVar[dict] = {
        "b_on_a": 5,
        "alpha_coeff": 0.005,
        "alpha_power": 2.0,
        "sound_speed": 1440,
        "density": 1000,
    }
    lung_fluid["beta"] = 1 + lung_fluid["b_on_a"] / 2

    sound_speed: float = 1540.0
    density: float = 1000.0
    alpha_coeff: float = 0.5
    alpha_power: float = 1.0
    beta: float = 0.0

    def __init__(
        self,
        sound_speed: float = 1540.0,
        density: float = 1000.0,
        alpha_coeff: float = 0.5,
        alpha_power: float = 1.0,
        beta: float = 0.0,
    ) -> None:
        """Initialize base material properties."""
        super().__init__()
        self.sound_speed = sound_speed
        self.density = density
        self.alpha_coeff = alpha_coeff
        self.alpha_power = alpha_power
        self.beta = beta
