"""Tests for the zero-attenuation gate.

The gate is isolated by building one medium twice, once as shipped and once with
the gate replaced by a no-op, so the two arms differ only in the gate. Half the
tests check that it leaves something alone, which is where the risk is, because
the `a` maps carry the absorbing layer as well as the interior relaxation.
"""

from __future__ import annotations

import numpy as np
import pytest

import fullwave
from fullwave.solver.pml_builder import PMLBuilder

F0 = 1e6
C0 = 1540.0
ALPHA_COEFF = 0.5
ALPHA_POWER = 1.0001
N_PML_LAYER = 12
N_TRANSITION_LAYER = 12
DRIVE_KEYS = ("a_pml_x1", "a_pml_u1", "a_pml_x2", "a_pml_u2")
DECAY_KEYS = ("b_pml_x1", "b_pml_u1", "b_pml_x2", "b_pml_u2")


def _no_gate(_builder: PMLBuilder, _maps: dict) -> int:
    """Stand in for the gate, so the control arm runs the same code path otherwise."""
    return 0


def _build(*, gate_enabled: bool) -> tuple[dict, PMLBuilder]:
    """Return the PML arrays for one medium, half of it lossless.

    The lossless half is the far half in x, so the gated region spans the
    interior and is clipped by the boundary guard on three sides.
    """
    domain_size = (3.2e-3, 3.2e-3)
    grid = fullwave.Grid(domain_size=domain_size, f0=F0, duration=domain_size[0] / C0 * 2, c0=C0)
    shape = (grid.nx, grid.ny)
    alpha_coeff = np.full(shape, ALPHA_COEFF)
    alpha_coeff[grid.nx // 2 :, :] = 0.0
    medium = fullwave.Medium(
        grid=grid,
        sound_speed=C0 * np.ones(shape),
        density=1000 * np.ones(shape),
        alpha_coeff=alpha_coeff,
        alpha_power=np.full(shape, ALPHA_POWER),
        beta=np.zeros(shape),
        use_isotropic_relaxation=True,
    )
    p_mask = np.zeros(shape, dtype=bool)
    p_mask[grid.nx // 2, :] = True
    builder = PMLBuilder(
        grid=grid,
        medium=medium,
        source=fullwave.Source(np.ones((int(p_mask.sum()), grid.nt)), p_mask),
        sensor=fullwave.Sensor(mask=np.ones(shape, dtype=bool)),
        n_pml_layer=N_PML_LAYER,
        n_transition_layer=N_TRANSITION_LAYER,
        use_isotropic_relaxation=True,
    )
    original = PMLBuilder._apply_zero_attenuation_gate
    if not gate_enabled:
        PMLBuilder._apply_zero_attenuation_gate = _no_gate
    try:
        extended = builder.run(use_pml=True)
    finally:
        PMLBuilder._apply_zero_attenuation_gate = original
    arrays = {k: np.asarray(v) for k, v in extended.relaxation_param_dict_for_fw2.items()}
    return arrays, builder


@pytest.fixture(scope="module")
def gated_and_plain() -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """Both builds, plus the interior mask and the gated-voxel mask."""
    gated, builder = _build(gate_enabled=True)
    plain, _ = _build(gate_enabled=False)
    shape = gated[DRIVE_KEYS[0]].shape
    guard = builder.num_boundary_points
    interior = np.zeros(shape, dtype=bool)
    interior[tuple(slice(guard, size - guard) for size in shape)] = True
    lossless = np.zeros(shape, dtype=bool)
    lossless[shape[0] // 2 :, :] = True
    return gated, plain, interior, lossless & interior


@pytest.mark.parametrize("key", DRIVE_KEYS)
def test_drive_is_exactly_zero_where_gated(key: str, gated_and_plain) -> None:
    gated, _, _, gated_voxels = gated_and_plain
    assert gated_voxels.any(), "the fixture gated nothing, so it tests nothing"
    assert (gated[key][gated_voxels] == 0).all()


@pytest.mark.parametrize("key", DRIVE_KEYS)
def test_drive_was_nonzero_there_before_the_gate(key: str, gated_and_plain) -> None:
    """A gate that did nothing at all would pass the test above."""
    _, plain, _, gated_voxels = gated_and_plain
    assert (plain[key][gated_voxels] != 0).any()


@pytest.mark.parametrize("key", DRIVE_KEYS)
def test_boundary_region_beside_a_lossy_region_is_untouched(key: str, gated_and_plain) -> None:
    """The absorbing layer inherits from the interior, so only its lossy side is fixed.

    **This replaces a test that required the whole boundary region to be
    untouched.** The gate now runs before the PML is built, so the region beside
    a lossless region inherits lossless values on purpose. The region beside the
    lossy half must still be identical.
    """
    gated, plain, interior, _ = gated_and_plain
    lossy_side = np.zeros(interior.shape, dtype=bool)
    lossy_side[: interior.shape[0] // 2, :] = True
    outside = ~interior & lossy_side
    np.testing.assert_array_equal(gated[key][outside], plain[key][outside])


@pytest.mark.parametrize("key", ("d_x_nu1", "d_u_nu1"))
def test_the_damping_ramp_survives_beside_a_lossless_region(key: str, gated_and_plain) -> None:
    """The PML raises its own damping, so the absorbing layer keeps working.

    The ramp comes from the first mechanism. The second mechanism carries the
    medium's own attenuation and correctly goes to zero for a lossless medium.
    """
    gated, _, interior, _ = gated_and_plain
    assert (gated[key][~interior] != 0).any(), f"{key} lost its damping ramp"


@pytest.mark.parametrize("key", DRIVE_KEYS)
def test_attenuating_interior_is_untouched(key: str, gated_and_plain) -> None:
    gated, plain, interior, gated_voxels = gated_and_plain
    keep = interior & ~gated_voxels
    np.testing.assert_array_equal(gated[key][keep], plain[key][keep])


@pytest.mark.parametrize("key", DECAY_KEYS)
def test_decay_is_one_where_gated(key: str, gated_and_plain) -> None:
    """`b` is 1 at a gated voxel, which is what a neutral medium carries.

    **This overturns an earlier decision to leave `b` alone.** That decision was
    correct that `a = 0` already holds the memory variable at 0, so `b` is inert.
    It is written anyway, so a gated voxel matches a neutral medium exactly. The
    clipped water cell carries `b_pml_x2 = -0.999404`, a sign-alternating mode
    that decays by 0.06 percent per step, and no lossless voxel should hold it.
    """
    gated, _, _, gated_voxels = gated_and_plain
    np.testing.assert_allclose(gated[key][gated_voxels], 1.0)


@pytest.mark.parametrize("key", DECAY_KEYS)
def test_decay_is_untouched_on_the_lossy_side(key: str, gated_and_plain) -> None:
    """The lossy half keeps its own decay, interior and absorbing layer alike."""
    gated, plain, interior, _ = gated_and_plain
    lossy_side = np.zeros(interior.shape, dtype=bool)
    lossy_side[: interior.shape[0] // 2, :] = True
    np.testing.assert_array_equal(gated[key][lossy_side], plain[key][lossy_side])


def test_gate_introduces_no_new_array(gated_and_plain) -> None:
    gated, plain, _, _ = gated_and_plain
    assert set(gated) == set(plain)


def test_medium_without_an_attenuation_coefficient_is_left_alone() -> None:
    """A MediumRelaxationMaps carries no coefficient, so there is nothing to gate on."""
    builder = PMLBuilder.__new__(PMLBuilder)
    builder.extended_medium = object()
    maps = {"a_pml_x1": np.ones((4, 4))}
    assert PMLBuilder._apply_zero_attenuation_gate(builder, maps) == 0
    np.testing.assert_array_equal(maps["a_pml_x1"], np.ones((4, 4)))


def _grid_and_shape():
    domain_size = (3.2e-3, 3.2e-3)
    grid = fullwave.Grid(domain_size=domain_size, f0=F0, duration=domain_size[0] / C0 * 2, c0=C0)
    return grid, (grid.nx, grid.ny)


def _half_lossless(shape, nx):
    alpha_coeff = np.full(shape, ALPHA_COEFF)
    alpha_coeff[nx // 2 :, :] = 0.0
    return alpha_coeff


def _build_from_maps(*, carry_alpha_coeff: bool) -> tuple[dict, PMLBuilder]:
    """Build the PML from a MediumRelaxationMaps, half of it lossless."""
    grid, shape = _grid_and_shape()
    alpha_coeff = _half_lossless(shape, grid.nx)
    source_medium = fullwave.Medium(
        grid=grid,
        sound_speed=C0 * np.ones(shape),
        density=1000 * np.ones(shape),
        alpha_coeff=alpha_coeff,
        alpha_power=np.full(shape, ALPHA_POWER),
        beta=np.zeros(shape),
        use_isotropic_relaxation=True,
    )
    maps = source_medium.build().relaxation_param_dict
    medium = fullwave.MediumRelaxationMaps(
        grid=grid,
        sound_speed=C0 * np.ones(shape),
        density=1000 * np.ones(shape),
        beta=np.zeros(shape),
        relaxation_param_dict=maps,
        alpha_coeff=alpha_coeff if carry_alpha_coeff else None,
    )
    p_mask = np.zeros(shape, dtype=bool)
    p_mask[grid.nx // 2, :] = True
    builder = PMLBuilder(
        grid=grid,
        medium=medium,
        source=fullwave.Source(np.ones((int(p_mask.sum()), grid.nt)), p_mask),
        sensor=fullwave.Sensor(mask=np.ones(shape, dtype=bool)),
        n_pml_layer=N_PML_LAYER,
        n_transition_layer=N_TRANSITION_LAYER,
        use_isotropic_relaxation=True,
    )
    extended = builder.run(use_pml=True)
    return {k: np.asarray(v) for k, v in extended.relaxation_param_dict_for_fw2.items()}, builder


def _gated_voxels(shape, guard):
    interior = np.zeros(shape, dtype=bool)
    interior[tuple(slice(guard, size - guard) for size in shape)] = True
    lossless = np.zeros(shape, dtype=bool)
    lossless[shape[0] // 2 :, :] = True
    return lossless & interior


def test_maps_route_reaches_the_gate_when_it_carries_the_coefficient() -> None:
    """The gate fires for a MediumRelaxationMaps that was given a coefficient."""
    gated, builder = _build_from_maps(carry_alpha_coeff=True)
    voxels = _gated_voxels(gated[DRIVE_KEYS[0]].shape, builder.num_boundary_points)
    assert voxels.any(), "the fixture gated nothing, so it tests nothing"
    for key in DRIVE_KEYS:
        assert (gated[key][voxels] == 0).all(), key


def test_maps_route_without_a_coefficient_keeps_the_old_behaviour() -> None:
    """Omitting the coefficient leaves the drive nonzero, as before 2026-08-17."""
    plain, builder = _build_from_maps(carry_alpha_coeff=False)
    voxels = _gated_voxels(plain[DRIVE_KEYS[0]].shape, builder.num_boundary_points)
    assert any((plain[key][voxels] != 0).any() for key in DRIVE_KEYS)


def _build_uniform_from_maps(alpha_coeff_value):
    """Return a uniform MediumRelaxationMaps carrying a SCALAR coefficient."""
    grid, shape = _grid_and_shape()
    source_medium = fullwave.Medium(
        grid=grid,
        sound_speed=C0 * np.ones(shape),
        density=1000 * np.ones(shape),
        alpha_coeff=np.full(shape, ALPHA_COEFF),
        alpha_power=np.full(shape, ALPHA_POWER),
        beta=np.zeros(shape),
        use_isotropic_relaxation=True,
    )
    medium = fullwave.MediumRelaxationMaps(
        grid=grid,
        sound_speed=C0 * np.ones(shape),
        density=1000 * np.ones(shape),
        beta=np.zeros(shape),
        relaxation_param_dict=source_medium.build().relaxation_param_dict,
        alpha_coeff=alpha_coeff_value,
    )
    p_mask = np.zeros(shape, dtype=bool)
    p_mask[grid.nx // 2, :] = True
    builder = PMLBuilder(
        grid=grid,
        medium=medium,
        source=fullwave.Source(np.ones((int(p_mask.sum()), grid.nt)), p_mask),
        sensor=fullwave.Sensor(mask=np.ones(shape, dtype=bool)),
        n_pml_layer=N_PML_LAYER,
        n_transition_layer=N_TRANSITION_LAYER,
        use_isotropic_relaxation=True,
    )
    extended = builder.run(use_pml=True)
    arrays = {k: np.asarray(v) for k, v in extended.relaxation_param_dict_for_fw2.items()}
    return arrays, builder, medium


def test_a_scalar_coefficient_is_stored_as_a_scalar() -> None:
    """A uniform medium must not materialise a grid to carry one number."""
    _, _, medium = _build_uniform_from_maps(0.0)
    assert np.ndim(medium.alpha_coeff) == 0


def test_a_scalar_zero_gates_the_whole_interior() -> None:
    gated, builder, _ = _build_uniform_from_maps(0.0)
    voxels = _gated_voxels(gated[DRIVE_KEYS[0]].shape, builder.num_boundary_points)
    interior = np.zeros(gated[DRIVE_KEYS[0]].shape, dtype=bool)
    guard = builder.num_boundary_points
    interior[tuple(slice(guard, size - guard) for size in interior.shape)] = True
    assert voxels.any()
    for key in DRIVE_KEYS:
        assert (gated[key][interior] == 0).all(), key
    for key in ("a_pml_x1", "a_pml_u1"):
        assert (gated[key][~interior] != 0).any(), f"{key} lost its damping ramp"


def test_a_scalar_nonzero_gates_nothing() -> None:
    plain, builder, _ = _build_uniform_from_maps(ALPHA_COEFF)
    guard = builder.num_boundary_points
    shape = plain[DRIVE_KEYS[0]].shape
    interior = np.zeros(shape, dtype=bool)
    interior[tuple(slice(guard, size - guard) for size in shape)] = True
    assert any((plain[key][interior] != 0).any() for key in DRIVE_KEYS)


def _build_with_coords(lossless_coords) -> tuple[dict, PMLBuilder]:
    """Build the PML from maps marked lossless by COORDINATES rather than a grid."""
    grid, shape = _grid_and_shape()
    source_medium = fullwave.Medium(
        grid=grid,
        sound_speed=C0 * np.ones(shape),
        density=1000 * np.ones(shape),
        alpha_coeff=np.full(shape, ALPHA_COEFF),
        alpha_power=np.full(shape, ALPHA_POWER),
        beta=np.zeros(shape),
        use_isotropic_relaxation=True,
    )
    medium = fullwave.MediumRelaxationMaps(
        grid=grid,
        sound_speed=C0 * np.ones(shape),
        density=1000 * np.ones(shape),
        beta=np.zeros(shape),
        relaxation_param_dict=source_medium.build().relaxation_param_dict,
        lossless_coords=lossless_coords,
    )
    p_mask = np.zeros(shape, dtype=bool)
    p_mask[grid.nx // 2, :] = True
    builder = PMLBuilder(
        grid=grid,
        medium=medium,
        source=fullwave.Source(np.ones((int(p_mask.sum()), grid.nt)), p_mask),
        sensor=fullwave.Sensor(mask=np.ones(shape, dtype=bool)),
        n_pml_layer=N_PML_LAYER,
        n_transition_layer=N_TRANSITION_LAYER,
        use_isotropic_relaxation=True,
    )
    extended = builder.run(use_pml=True)
    return {k: np.asarray(v) for k, v in extended.relaxation_param_dict_for_fw2.items()}, builder


def _build_marked(
    *,
    alpha_coeff=None,
    lossless_coords=None,
    sound_speed_transfer: bool = False,
    band_scale: float = 1.0,
) -> dict:
    """Build from one fixed set of relaxation maps, varying only the marking.

    The maps come from a uniform medium, so the two markings see identical
    inputs. A half-lossless medium would change the maps themselves.
    """
    grid, shape = _grid_and_shape()
    uniform = fullwave.Medium(
        grid=grid,
        sound_speed=C0 * np.ones(shape),
        density=1000 * np.ones(shape),
        alpha_coeff=np.full(shape, ALPHA_COEFF),
        alpha_power=np.full(shape, ALPHA_POWER),
        beta=np.zeros(shape),
        use_isotropic_relaxation=True,
        sound_speed_transfer=sound_speed_transfer,
        band_scale=band_scale,
    )
    medium = fullwave.MediumRelaxationMaps(
        grid=grid,
        sound_speed=C0 * np.ones(shape),
        density=1000 * np.ones(shape),
        beta=np.zeros(shape),
        relaxation_param_dict=uniform.build().relaxation_param_dict,
        alpha_coeff=alpha_coeff,
        lossless_coords=lossless_coords,
    )
    p_mask = np.zeros(shape, dtype=bool)
    p_mask[grid.nx // 2, :] = True
    builder = PMLBuilder(
        grid=grid,
        medium=medium,
        source=fullwave.Source(np.ones((int(p_mask.sum()), grid.nt)), p_mask),
        sensor=fullwave.Sensor(mask=np.ones(shape, dtype=bool)),
        n_pml_layer=N_PML_LAYER,
        n_transition_layer=N_TRANSITION_LAYER,
        use_isotropic_relaxation=True,
    )
    extended = builder.run(use_pml=True)
    return {k: np.asarray(v) for k, v in extended.relaxation_param_dict_for_fw2.items()}


def test_coordinates_gate_the_same_voxels_as_an_array() -> None:
    """Coordinates and an array coefficient must gate identically."""
    grid, shape = _grid_and_shape()
    marking = _half_lossless(shape, grid.nx)
    from_array = _build_marked(alpha_coeff=marking)
    from_coords = _build_marked(lossless_coords=np.argwhere(marking == 0))
    for key in (*DRIVE_KEYS, *DECAY_KEYS):
        np.testing.assert_array_equal(from_coords[key], from_array[key], err_msg=key)


def test_the_marking_actually_gates_something() -> None:
    """Guards the test above against passing vacuously."""
    grid, shape = _grid_and_shape()
    marking = _half_lossless(shape, grid.nx)
    unmarked = _build_marked()
    marked = _build_marked(lossless_coords=np.argwhere(marking == 0))
    assert any((unmarked[key] != marked[key]).any() for key in DRIVE_KEYS)


def test_coordinates_and_an_array_coefficient_are_mutually_exclusive() -> None:
    """Two markings of the same thing must not be given at once."""
    grid, shape = _grid_and_shape()
    with pytest.raises(ValueError, match="lossless_coords"):
        fullwave.MediumRelaxationMaps(
            grid=grid,
            sound_speed=C0 * np.ones(shape),
            density=1000 * np.ones(shape),
            beta=np.zeros(shape),
            relaxation_param_dict={},
            alpha_coeff=np.full(shape, ALPHA_COEFF),
            lossless_coords=np.zeros((1, 2), dtype=np.int64),
        )


def test_kappa_is_one_where_gated() -> None:
    """A gated voxel must carry no stretching, or its sound speed is wrong.

    The pressure update and the velocity update read different members of kappa,
    so an ungated kappa moves the speed to c / sqrt(kappa_x kappa_u) while
    nothing is absorbed. Measured at 1554.05 m/s against 1540 before the fix.
    """
    gated, builder = _build(gate_enabled=True)
    guard = builder.num_boundary_points
    shape = gated["a_pml_x1"].shape
    interior = np.zeros(shape, dtype=bool)
    interior[tuple(slice(guard, size - guard) for size in shape)] = True
    lossless = np.zeros(shape, dtype=bool)
    lossless[shape[0] // 2 :, :] = True
    gated_voxels = lossless & interior
    kappa_keys = [key for key in gated if key.startswith("kappa_")]
    assert kappa_keys, "the medium carries no kappa map"
    for key in kappa_keys:
        np.testing.assert_allclose(gated[key][gated_voxels], 1.0, err_msg=key)


def test_kappa_is_untouched_on_the_lossy_side() -> None:
    """The lossy half keeps its own stretching, interior and absorbing layer alike."""
    gated, _ = _build(gate_enabled=True)
    plain, _ = _build(gate_enabled=False)
    shape = gated["a_pml_x1"].shape
    lossy_side = np.zeros(shape, dtype=bool)
    lossy_side[: shape[0] // 2, :] = True
    for key in (k for k in gated if k.startswith("kappa_")):
        np.testing.assert_array_equal(gated[key][lossy_side], plain[key][lossy_side], err_msg=key)


@pytest.mark.parametrize("sound_speed_transfer", [False, True])
@pytest.mark.parametrize("band_scale", [1.0, 10.0])
def test_a_lossless_voxel_ignores_the_relaxation_corrections(
    *, sound_speed_transfer: bool, band_scale: float
) -> None:
    """A lossless voxel holds the same parameters whatever correction was applied.

    Both corrections exist to repair the relaxation formulation. A lossless voxel
    has no relaxation to repair, and the gate runs after they are applied, so it
    overwrites whatever they left. This pins that ordering.
    """
    grid, shape = _grid_and_shape()
    marking = _half_lossless(shape, grid.nx)
    reference = _build_marked(alpha_coeff=marking)
    corrected = _build_marked(
        alpha_coeff=marking,
        sound_speed_transfer=sound_speed_transfer,
        band_scale=band_scale,
    )
    edge = N_PML_LAYER + N_TRANSITION_LAYER + 4
    lossless = np.zeros(reference[DRIVE_KEYS[0]].shape, dtype=bool)
    lossless[reference[DRIVE_KEYS[0]].shape[0] // 2 :, :] = True
    interior = np.zeros(lossless.shape, dtype=bool)
    interior[tuple(slice(edge, size - edge) for size in lossless.shape)] = True
    voxels = lossless & interior
    assert voxels.any()
    for key in (*DRIVE_KEYS, *DECAY_KEYS, *(k for k in reference if k.startswith("kappa"))):
        np.testing.assert_array_equal(corrected[key][voxels], reference[key][voxels], err_msg=key)
