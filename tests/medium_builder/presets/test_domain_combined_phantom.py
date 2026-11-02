import numpy as np
import pytest

import fullwave
from fullwave.medium_builder.presets import CombinedPhantomDomain
from fullwave.medium_builder.presets.combined_phantom_map import (
    HYPERECHOIC_TARGET,
    HYPOECHOIC_TARGET,
    POINT_TARGET,
    generate_combined_phantom_map,
)

BASE_DENSITY = 1000.0
POINT_TARGET_DEPTH_M = 10e-3
CONTRAST_TARGET_DEPTH_M = 20e-3


@pytest.fixture(scope="module")
def grid():
    return fullwave.Grid((0.03, 0.05), 1e6, 20e-6, c0=1540, ppw=12, cfl=0.4)


@pytest.fixture(scope="module")
def phantom(grid):
    return CombinedPhantomDomain(grid=grid, use_center=True)


def depth_row(grid, depth_m):
    return int(depth_m / grid.dx)


def grid_cell(grid, *, lateral_m, depth_m):
    return int(depth_m / grid.dx), int(lateral_m / grid.dx + grid.ny / 2)


def test_point_target_centers_are_exactly_the_point_target_cells(grid, phantom):
    centers = phantom.point_target_centers_in_grid_m()

    cells = {grid_cell(grid, lateral_m=lateral, depth_m=depth) for lateral, depth in centers}

    point_cells = set(zip(*np.nonzero(phantom.phantom_map == POINT_TARGET), strict=True))
    assert cells == point_cells
    assert len(centers) > 0


def test_contrast_target_circles_follow_the_label_map(grid, phantom):
    circles = phantom.contrast_target_circles_in_grid_m()

    assert len(circles) == 2
    for lateral, depth, radius in circles:
        row, col = grid_cell(grid, lateral_m=lateral, depth_m=depth)
        label = phantom.phantom_map[row, col]
        assert label in (HYPOECHOIC_TARGET, HYPERECHOIC_TARGET)
        assert phantom.phantom_map[row, col + int(0.9 * radius / grid.dx)] == label
        assert phantom.phantom_map[row, col + int(1.2 * radius / grid.dx)] != label


def test_target_geometry_moves_with_the_shift(grid):
    shift_rows = 20
    domain = CombinedPhantomDomain(grid=grid, use_center=True, shift_px=[shift_rows, 0])

    unshifted = CombinedPhantomDomain(grid=grid, use_center=True)

    first_shifted = domain.point_target_centers_in_grid_m()[0]
    first_unshifted = unshifted.point_target_centers_in_grid_m()[0]
    assert first_shifted[0] == pytest.approx(first_unshifted[0])
    assert first_shifted[1] - first_unshifted[1] == pytest.approx(shift_rows * grid.dx)


def test_without_targets_the_geometry_is_empty(grid):
    domain = CombinedPhantomDomain(
        grid=grid,
        use_center=True,
        use_point_target=False,
        use_contrast_target=False,
    )

    assert domain.point_target_centers_in_grid_m() == []
    assert domain.contrast_target_circles_in_grid_m() == []


def test_maps_have_the_grid_shape(grid, phantom):
    for domain_map in (
        phantom.sound_speed,
        phantom.density,
        phantom.alpha_coeff,
        phantom.alpha_power,
        phantom.beta,
        phantom.air,
        phantom.phantom_map,
    ):
        assert domain_map.shape == grid.shape


def test_point_target_cell_differs_from_background(grid, phantom):
    row = depth_row(grid, POINT_TARGET_DEPTH_M)
    column = grid.ny // 2
    assert phantom.phantom_map[row, column] == POINT_TARGET
    assert phantom.density[row, column] == pytest.approx(1.2 * BASE_DENSITY)
    assert phantom.sound_speed[row, column] == pytest.approx(1.2 * 1540)
    background_density = np.median(phantom.density[phantom.phantom_map == 0])
    assert phantom.density[row, column] != pytest.approx(background_density)


def test_hyperechoic_target_scatters_more_than_hypoechoic_target(grid, phantom):
    scatterer_std = {
        label: phantom.scatterer[phantom.phantom_map == label].std()
        for label in (0, HYPOECHOIC_TARGET, HYPERECHOIC_TARGET)
    }
    assert scatterer_std[HYPOECHOIC_TARGET] < scatterer_std[0] < scatterer_std[HYPERECHOIC_TARGET]


def test_contrast_targets_sit_on_both_sides_of_the_center(grid, phantom):
    row = depth_row(grid, CONTRAST_TARGET_DEPTH_M)
    offset = int(10e-3 / grid.dx)
    assert phantom.phantom_map[row, grid.ny // 2 - offset] == HYPOECHOIC_TARGET
    assert phantom.phantom_map[row, grid.ny // 2 + offset] == HYPERECHOIC_TARGET


def test_without_point_targets_the_map_has_none(grid):
    domain = CombinedPhantomDomain(grid=grid, use_center=True, use_point_target=False)
    assert not np.any(domain.phantom_map == POINT_TARGET)


def test_contrast_targets_do_not_overlap_point_targets():
    contrast_only = generate_combined_phantom_map(dx_m=0.25e-3, use_point_target=False)
    with_point_targets = generate_combined_phantom_map(dx_m=0.25e-3, point_target_radius_px=3)
    for label in (HYPOECHOIC_TARGET, HYPERECHOIC_TARGET):
        assert np.sum(with_point_targets == label) == np.sum(contrast_only == label)


def test_contrast_target_offset_moves_the_cylinders():
    dx_m = 1e-3
    near = generate_combined_phantom_map(dx_m=dx_m, contrast_target_offset_m=10e-3)
    far = generate_combined_phantom_map(dx_m=dx_m, contrast_target_offset_m=15e-3)
    center = near.shape[1] // 2
    row = int(20e-3 / dx_m)
    assert near[row, center + 10] == HYPERECHOIC_TARGET
    assert far[row, center + 10] != HYPERECHOIC_TARGET
    assert far[row, center + 15] == HYPERECHOIC_TARGET


def test_point_target_disk_covers_more_than_one_cell():
    single_cell = generate_combined_phantom_map(dx_m=1e-3)
    disk = generate_combined_phantom_map(dx_m=1e-3, point_target_radius_px=2)
    assert np.sum(disk == POINT_TARGET) > np.sum(single_cell == POINT_TARGET)


def test_shift_moves_the_phantom_down_and_fills_the_top_with_background(grid):
    shift_rows = 20
    domain = CombinedPhantomDomain(grid=grid, use_center=True, shift_px=[shift_rows, 0])
    row = depth_row(grid, POINT_TARGET_DEPTH_M) + shift_rows
    assert domain.phantom_map[row, grid.ny // 2] == POINT_TARGET
    assert not np.any(domain.phantom_map[:shift_rows] != 0)


def test_crop_end_before_crop_start_raises(grid):
    with pytest.raises(ValueError, match="crop_end"):
        CombinedPhantomDomain(grid=grid, crop_start=[10, 10], crop_end=[5, 20])


def test_negative_crop_start_raises(grid):
    with pytest.raises(ValueError, match="crop_start"):
        CombinedPhantomDomain(grid=grid, crop_start=[-1, 0])
