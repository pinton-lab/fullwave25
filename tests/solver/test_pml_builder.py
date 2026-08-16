"""Tests for how the two-stage PML is built.

The regression test is the important one. `medium_matched` is what every
published Fullwave 2 and Fullwave 2.5 result was produced with, and once
`decoupled` became the default nothing exercises the old path by accident, so
it is pinned here against a stored fixture with exact equality.

Regenerate the fixture only when the old path is deliberately changed, which
should be never:

    uv run python tests/solver/test_pml_builder.py --write-fixture
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pytest

import fullwave
from fullwave.solver.pml_builder import PMLBuilder

DATA = Path(__file__).parent / "data"
# Both relaxation modes are pinned. The anisotropic one is the important half,
# because there several PML axis letters read one source map, `d_u` and
# `d_w` both reading `d_x1`, which is where the old in-place ramp could let the
# second letter be built from the first letter's output.
FIXTURES = {
    True: DATA / "pml_builder_medium_matched_isotropic.npz",
    False: DATA / "pml_builder_medium_matched_anisotropic.npz",
}

F0 = 1e6
C0 = 1540.0
ALPHA_COEFF = 0.5
ALPHA_POWER = 1.0001
N_PML_LAYER = 12
N_TRANSITION_LAYER = 12


def _pieces(
    alpha_coeff: float = ALPHA_COEFF,
    alpha_power: float = ALPHA_POWER,
    *,
    isotropic: bool = True,
):
    """Return a small grid, medium, source and sensor."""
    domain_size = (3.2e-3, 3.2e-3)
    grid = fullwave.Grid(
        domain_size=domain_size,
        f0=F0,
        duration=domain_size[0] / C0 * 2,
        c0=C0,
    )
    shape = (grid.nx, grid.ny)
    medium = fullwave.Medium(
        grid=grid,
        sound_speed=C0 * np.ones(shape),
        density=1000 * np.ones(shape),
        alpha_coeff=alpha_coeff * np.ones(shape),
        alpha_power=alpha_power * np.ones(shape),
        beta=np.zeros(shape),
        use_isotropic_relaxation=isotropic,
    )
    p_mask = np.zeros(shape, dtype=bool)
    p_mask[grid.nx // 2, :] = True
    source = fullwave.Source(np.ones((int(p_mask.sum()), grid.nt)), p_mask)
    sensor = fullwave.Sensor(mask=np.ones(shape, dtype=bool))
    return grid, medium, source, sensor


def _build(
    design: str = "medium_matched",
    alpha_coeff: float = ALPHA_COEFF,
    *,
    isotropic: bool = True,
) -> dict:
    """Return the PML arrays for one design, on the small grid."""
    grid, medium, source, sensor = _pieces(alpha_coeff=alpha_coeff, isotropic=isotropic)
    builder = PMLBuilder(
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        n_pml_layer=N_PML_LAYER,
        n_transition_layer=N_TRANSITION_LAYER,
        use_isotropic_relaxation=isotropic,
        pml_design=design,
    )
    extended = builder.run(use_pml=True)
    return {k: np.asarray(v) for k, v in extended.relaxation_param_dict_for_fw2.items()}


@pytest.mark.parametrize("isotropic", [True, False])
def test_medium_matched_is_unchanged(isotropic: bool) -> None:  # noqa: FBT001
    """The previously shipped PML must reproduce the stored fixture exactly."""
    fixture = FIXTURES[isotropic]
    if not fixture.exists():
        pytest.skip(f"fixture missing, regenerate with --write-fixture: {fixture}")
    stored = np.load(fixture)
    built = _build("medium_matched", isotropic=isotropic)
    assert set(stored.files) == set(built), "the set of PML arrays changed"
    for key in stored.files:
        np.testing.assert_array_equal(
            built[key], stored[key], err_msg=f"{key} changed under medium_matched"
        )


def test_apply_transition_and_pml_does_not_modify_its_argument() -> None:
    """The ramp helper must not write through the array it is given."""
    grid, medium, source, sensor = _pieces()
    builder = PMLBuilder(
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        n_pml_layer=N_PML_LAYER,
        n_transition_layer=N_TRANSITION_LAYER,
        use_isotropic_relaxation=True,
    )
    field = np.arange(64 * 48, dtype=np.float64).reshape(64, 48)
    before = field.copy()
    builder._apply_transition_and_pml(
        field,
        value_target=0.0,
        array_shape=field.shape,
        axis=0,
        transition_type="linear",
    )
    np.testing.assert_array_equal(field, before, err_msg="the input array was modified")


@pytest.mark.parametrize("design", ["medium_matched", "decoupled"])
def test_the_medium_survives_the_build(design: str) -> None:
    """Applying the PML must not overwrite the medium maps it read.

    `_apply_pml_2d` writes into `relaxation_param_dict_for_fw2` and reads from
    `relaxation_param_dict`. It used to leave the second holding the PML
    rather than the medium, because the ramp helper wrote through the view it
    was given.
    """
    grid, medium, source, sensor = _pieces()
    builder = PMLBuilder(
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        n_pml_layer=N_PML_LAYER,
        n_transition_layer=N_TRANSITION_LAYER,
        use_isotropic_relaxation=True,
        pml_design=design,
    )
    extended = builder.extended_medium.build()
    before = {k: np.asarray(v).copy() for k, v in extended.relaxation_param_dict.items()}
    builder._apply_pml_2d(
        extended_medium=extended,
        theoritical_reflection_coefficient=builder.theoritical_reflection_coefficient,
        n_polynomial=builder.n_polynomial,
    )
    for key, value in extended.relaxation_param_dict.items():
        np.testing.assert_array_equal(
            np.asarray(value), before[key], err_msg=f"the medium's {key} was overwritten"
        )


def test_unknown_pml_design_raises() -> None:
    """An unknown design must be refused rather than silently falling through."""
    grid, medium, source, sensor = _pieces()
    with pytest.raises(ValueError, match="pml_design"):
        PMLBuilder(
            grid=grid,
            medium=medium,
            source=source,
            sensor=sensor,
            n_pml_layer=N_PML_LAYER,
            n_transition_layer=N_TRANSITION_LAYER,
            pml_design="nonsense",
        )


def test_decoupled_pml_does_not_depend_on_the_medium() -> None:
    """Two very different media must give the same PML inside the PML layer.

    This is the property the redesign exists for, and it is checkable without
    running the solver at all.

    **The relaxation frequency is exactly independent of the medium**, because
    it is spliced in rather than ramped from the interior value. The damping
    strength is independent to about one part in a million rather than exactly,
    because its ramp starts from whatever the transition layer has left at the
    join, which is a thousandth of the interior value rather than zero. That
    residue is nine orders below the PML damping target and is not worth removing.
    """
    edge = 8 + N_PML_LAYER
    water = _build("decoupled", alpha_coeff=0.0022)
    tissue = _build("decoupled", alpha_coeff=0.9)
    compared = 0
    for key in water:
        if not (key.startswith(("d_", "alpha_")) and "nu1" in key):
            continue
        here, there = water[key][:edge, :edge], tissue[key][:edge, :edge]
        if key.startswith("alpha_"):
            np.testing.assert_array_equal(
                here, there, err_msg=f"{key} is not exactly medium independent"
            )
        else:
            residue = np.abs(here - there).max() / np.abs(here).max()
            assert residue < 1e-5, f"{key} carries a {residue:.2e} relative medium dependence"
        compared += 1
    assert compared > 0, "no first-mechanism arrays were compared"


def test_medium_matched_pml_does_depend_on_the_medium() -> None:
    """The control for the test above, so that it cannot pass vacuously."""
    edge = 8 + N_PML_LAYER
    water = _build("medium_matched", alpha_coeff=0.0022)
    tissue = _build("medium_matched", alpha_coeff=0.9)
    differs = any(
        not np.array_equal(water[key][:edge, :edge], tissue[key][:edge, :edge])
        for key in water
        if key.startswith(("d_", "alpha_")) and "nu1" in key
    )
    assert differs, "medium_matched no longer depends on the medium, so the pair is vacuous"


@pytest.mark.parametrize("isotropic", [True, False])
def test_every_named_design_builds(isotropic: bool) -> None:  # noqa: FBT001
    """Every design in the registry must produce a full set of arrays."""
    expected = set(_build("medium_matched", isotropic=isotropic))
    for design in ("medium_matched", "decoupled"):
        built = set(_build(design, isotropic=isotropic))
        assert built == expected, f"{design} produced a different set of arrays"


def test_decoupled_is_the_default() -> None:
    """The default must be the redesigned PML, and it must be named."""
    grid, medium, source, sensor = _pieces()
    builder = PMLBuilder(
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        n_pml_layer=N_PML_LAYER,
        n_transition_layer=N_TRANSITION_LAYER,
        use_isotropic_relaxation=True,
    )
    assert builder.pml_design == "decoupled"


def _write_fixture() -> None:
    """Store the medium_matched arrays as the regression fixtures."""
    DATA.mkdir(parents=True, exist_ok=True)
    for isotropic, path in FIXTURES.items():
        np.savez_compressed(path, **_build("medium_matched", isotropic=isotropic))
        print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-fixture", action="store_true")
    if ap.parse_args().write_fixture:
        _write_fixture()
    else:
        print("nothing to do, pass --write-fixture")
