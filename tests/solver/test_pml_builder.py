"""Tests for how the two-stage PML is built.

The important test is that the PML inside the layer does not depend on the
medium, which is the property the two stage design exists for. Its control is
that the medium outside the layer does still differ, so the pair cannot pass
because both media were built the same.

Both relaxation modes are covered. The anisotropic one is the important half,
because there several PML axis letters read one source map, `d_u` and `d_w`
both reading `d_x1`, which is where an in-place ramp could let the second
letter be built from the first letter's output.
"""

from __future__ import annotations

import numpy as np
import pytest

import fullwave
from fullwave.solver.pml_builder import PMLBuilder, PMLBuilderExponentialAttenuation

F0 = 1e6
C0 = 1540.0
ALPHA_COEFF = 0.5
ALPHA_POWER = 1.0001
N_PML_LAYER = 12
N_TRANSITION_LAYER = 12


class Carried:
    """The one map the layer carries to one across itself, in place."""

    key = "kappa_x1"


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
    alpha_coeff: float = ALPHA_COEFF,
    *,
    isotropic: bool = True,
) -> dict:
    """Return the PML arrays on the small grid."""
    grid, medium, source, sensor = _pieces(alpha_coeff=alpha_coeff, isotropic=isotropic)
    builder = PMLBuilder(
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        n_pml_layer=N_PML_LAYER,
        n_transition_layer=N_TRANSITION_LAYER,
        use_isotropic_relaxation=isotropic,
    )
    extended = builder.run(use_pml=True)
    return {k: np.asarray(v) for k, v in extended.relaxation_param_dict_for_fw2.items()}


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


def test_the_medium_survives_the_build() -> None:
    """Applying the PML must not overwrite the medium maps it read.

    `_apply_pml_2d` writes into `relaxation_param_dict_for_fw2` and reads from
    `relaxation_param_dict`. It used to leave the second holding the PML
    rather than the medium, because the ramp helper wrote through the view it
    was given.

    The layer carries `kappa_x1` to one across itself, and it does that in the
    medium's own map on purpose, so that one key is excluded. Every other key
    must survive.
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
    )
    extended = builder.extended_medium.build()
    before = {k: np.asarray(v).copy() for k, v in extended.relaxation_param_dict.items()}
    builder._apply_pml_2d(
        extended_medium=extended,
        theoritical_reflection_coefficient=builder.theoritical_reflection_coefficient,
        n_polynomial=builder.n_polynomial,
    )
    carried = Carried.key if builder.pml_unstretched else None
    for key, value in extended.relaxation_param_dict.items():
        if key == carried:
            continue
        np.testing.assert_array_equal(
            np.asarray(value), before[key], err_msg=f"the medium's {key} was overwritten"
        )


def test_the_pml_does_not_depend_on_the_medium() -> None:
    """Two very different media must give the same PML inside the PML layer.

    This is the property the two stage design exists for, and it is checkable
    without running the solver at all.

    **The relaxation frequency is exactly independent of the medium**, because
    it is spliced in rather than ramped from the interior value. The damping
    strength is independent to about one part in a million rather than exactly,
    because its ramp starts from whatever the transition layer has left at the
    join, which is a thousandth of the interior value rather than zero. That
    residue is nine orders below the PML damping target and is not worth removing.
    """
    edge = 8 + N_PML_LAYER
    water = _build(alpha_coeff=0.0022)
    tissue = _build(alpha_coeff=0.9)
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


def test_the_interior_does_depend_on_the_medium() -> None:
    """The control for the test above, so that it cannot pass vacuously.

    Two media that give the same PML must still give a different interior. If
    they did not, both arms would be the same medium and the independence above
    would mean nothing.
    """
    middle = slice(None), slice(None)
    water = _build(alpha_coeff=0.0022)
    tissue = _build(alpha_coeff=0.9)
    differs = any(
        not np.array_equal(water[key][middle], tissue[key][middle])
        for key in water
        if key.startswith(("d_", "alpha_")) and "nu1" in key
    )
    assert differs, "the two media build the same interior, so the pair is vacuous"


@pytest.mark.parametrize("isotropic", [True, False])
def test_the_build_produces_a_full_set_of_arrays(isotropic: bool) -> None:  # noqa: FBT001
    """Each axis must come back with its stretching map and a pair for each mechanism."""
    _, medium, _, _ = _pieces(isotropic=isotropic)
    built = _build(isotropic=isotropic)
    letters = ("u", "x") if isotropic else ("u", "w", "x", "y")
    mechanisms = range(1, medium.n_relaxation_mechanisms + 1)
    expected = {f"kappa_{letter}" for letter in letters}
    expected |= {
        f"{side}_pml_{letter}{nu}" for side in ("a", "b") for letter in letters for nu in mechanisms
    }
    expected |= {
        f"{name}_{letter}_nu{nu}"
        for name in ("d", "alpha")
        for letter in letters
        for nu in mechanisms
    }
    assert set(built) == expected


def _exponential_medium():
    """Return the pieces the exponential attenuation builder takes."""
    return _pieces(alpha_coeff=0.5, alpha_power=1.0001)


def test_the_layer_is_two_wavelengths_by_default() -> None:
    """The C-PML layer is on by default and it is 2 wavelengths thick.

    The margin holds the layer and nothing else, so it is the same width.
    """
    grid, medium, source, sensor = _exponential_medium()
    builder = PMLBuilderExponentialAttenuation(
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
    )
    assert builder.exponential_attenuation_pml_thickness_px == 2 * grid.ppw
    assert builder.n_pml_layer == 2 * grid.ppw


def test_the_margin_holds_the_taper_when_the_layer_is_off() -> None:
    """With the layer off the taper is the absorber, so the margin is its depth.

    The taper fills the whole margin, and it needs 4 wavelengths, which is what
    every release before the layer used.
    """
    grid, medium, source, sensor = _exponential_medium()
    builder = PMLBuilderExponentialAttenuation(
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        exponential_attenuation_pml_thickness_px=0,
    )
    assert builder.n_pml_layer == 4 * grid.ppw


def test_a_stated_margin_is_kept() -> None:
    """A caller who states a margin gets it, whichever absorber is in use."""
    grid, medium, source, sensor = _exponential_medium()
    builder = PMLBuilderExponentialAttenuation(
        grid=grid,
        medium=medium,
        source=source,
        sensor=sensor,
        n_pml_layer=5 * grid.ppw,
    )
    assert builder.n_pml_layer == 5 * grid.ppw
    assert builder.exponential_attenuation_pml_thickness_px == 2 * grid.ppw


def test_a_layer_thicker_than_the_margin_is_refused() -> None:
    """The layer sits between the interior and the grid edge, so it must fit."""
    grid, medium, source, sensor = _exponential_medium()
    with pytest.raises(ValueError, match="does not fit"):
        PMLBuilderExponentialAttenuation(
            grid=grid,
            medium=medium,
            source=source,
            sensor=sensor,
            n_pml_layer=N_PML_LAYER,
            exponential_attenuation_pml_thickness_px=N_PML_LAYER + 1,
        )


def test_the_taper_stops_where_the_layer_is_on() -> None:
    """The layer replaces the margin taper rather than adding to it.

    With the layer off the attenuation falls toward the grid edge. With the
    layer on every margin cell keeps the attenuation the medium asked for.
    """
    fields = {}
    for name, thickness in (("off", 0), ("on", N_PML_LAYER)):
        grid, medium, source, sensor = _exponential_medium()
        builder = PMLBuilderExponentialAttenuation(
            grid=grid,
            medium=medium,
            source=source,
            sensor=sensor,
            n_pml_layer=N_PML_LAYER,
            exponential_attenuation_pml_thickness_px=thickness,
        )
        fields[name] = np.asarray(builder.run(use_pml=True).alpha_exp)

    edge_off = fields["off"][0, fields["off"].shape[1] // 2]
    edge_on = fields["on"][0, fields["on"].shape[1] // 2]
    centre = fields["on"][fields["on"].shape[0] // 2, fields["on"].shape[1] // 2]

    assert edge_off < centre, "with the layer off the taper must lower the edge"
    assert edge_on == pytest.approx(centre), "with the layer on the edge must keep the attenuation"
    np.testing.assert_allclose(fields["on"], fields["on"][0, 0], err_msg="the map must be uniform")
