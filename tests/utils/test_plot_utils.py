import matplotlib as mpl
import numpy as np
import pytest

from fullwave.grid import Grid
from fullwave.utils import plot_utils

mpl.use("Agg")


def _draw_snapshot(tmp_path, **circle_kwargs):
    rng = np.random.default_rng(seed=0)
    shape = (6, 8)
    plot_utils.plot_wave_propagation_snapshot(
        propagation_map=np.zeros(shape),
        c_map=1500 + 100 * rng.random(shape),
        rho_map=1000 + 100 * rng.random(shape),
        export_name=tmp_path / "snapshot.png",
        vmin=-1.0,
        vmax=1.0,
        **circle_kwargs,
    )
    return mpl.pyplot.gcf().axes[0], shape


def test_snapshot_without_circles_draws_the_normalized_impedance_and_no_patches(tmp_path):
    ax, shape = _draw_snapshot(tmp_path)

    rng = np.random.default_rng(seed=0)
    impedance = (1500 + 100 * rng.random(shape)) * (1000 + 100 * rng.random(shape))
    expected = (impedance - impedance.min()) / (impedance.max() - impedance.min() + 1e-9)
    np.testing.assert_array_equal(np.asarray(ax.images[0].get_array()), 0.8 * expected)
    assert len(ax.patches) == 0


def test_snapshot_draws_each_given_circle_as_an_unfilled_outline(tmp_path):
    circles = [(1.0, 2.0, 0.5), (-1.5, 3.0, 1.0)]

    ax, _ = _draw_snapshot(tmp_path, circles=circles)

    drawn = [(*patch.center, patch.radius) for patch in ax.patches]
    assert drawn == circles
    assert not any(patch.get_fill() for patch in ax.patches)


def test_snapshot_uses_the_default_colormap_without_cmap(tmp_path):
    ax, _ = _draw_snapshot(tmp_path)

    assert ax.images[0].get_cmap().name == mpl.rcParams["image.cmap"]


def test_snapshot_uses_the_given_colormap(tmp_path):
    ax, _ = _draw_snapshot(tmp_path, cmap="berlin")

    assert ax.images[0].get_cmap().name == "berlin"


def test_snapshot_without_the_medium_draws_the_pressure_alone(tmp_path):
    ax, shape = _draw_snapshot(tmp_path, show_medium=False)

    np.testing.assert_array_equal(np.asarray(ax.images[0].get_array()), np.zeros(shape))


def test_snapshot_without_points_draws_no_lines(tmp_path):
    ax, _ = _draw_snapshot(tmp_path)

    assert len(ax.lines) == 0


def test_snapshot_draws_the_given_points_as_filled_dots_without_moving_the_axes(tmp_path):
    points = [(1.0, 2.0), (-1.5, 3.0)]
    extent = (-4.0, 4.0, 6.0, 0.0)

    ax, _ = _draw_snapshot(tmp_path, points=points, extent=extent)

    (line,) = ax.lines
    assert list(zip(line.get_xdata(), line.get_ydata(), strict=True)) == points
    assert line.get_linestyle() == "None"
    assert line.get_marker() == "o"
    assert ax.get_xlim() == (-4.0, 4.0)
    assert ax.get_ylim() == (6.0, 0.0)


def test_snapshot_marks_are_red_by_default(tmp_path):
    ax, _ = _draw_snapshot(tmp_path, circles=[(1.0, 2.0, 0.5)], points=[(1.0, 2.0)])

    assert ax.patches[0].get_edgecolor() == mpl.colors.to_rgba("red")
    assert mpl.colors.to_rgba(ax.lines[0].get_color()) == mpl.colors.to_rgba("red")


def test_snapshot_draws_circles_and_points_in_the_given_mark_color(tmp_path):
    ax, _ = _draw_snapshot(
        tmp_path,
        circles=[(1.0, 2.0, 0.5)],
        points=[(1.0, 2.0)],
        mark_color="black",
    )

    assert ax.patches[0].get_edgecolor() == mpl.colors.to_rgba("black")
    assert mpl.colors.to_rgba(ax.lines[0].get_color()) == mpl.colors.to_rgba("black")


def test_animation_draws_circles_and_points_in_the_given_mark_color(tmp_path, monkeypatch):
    shape = (6, 8)
    monkeypatch.setattr(plot_utils.animation.ArtistAnimation, "save", lambda *_, **__: None)
    monkeypatch.setattr(plot_utils.plt, "close", lambda *_: None)

    plot_utils.plot_wave_propagation_with_map(
        propagation_map=np.zeros((2, *shape)),
        c_map=np.full(shape, 1500.0),
        rho_map=np.full(shape, 1000.0),
        export_name=tmp_path / "animation.mp4",
        num_plot_image=2,
        circles=[(1.0, 2.0, 0.5)],
        points=[(1.0, 2.0)],
        mark_color="black",
    )

    ax = mpl.pyplot.gcf().axes[0]
    assert ax.patches[0].get_edgecolor() == mpl.colors.to_rgba("black")
    assert mpl.colors.to_rgba(ax.lines[0].get_color()) == mpl.colors.to_rgba("black")
    monkeypatch.undo()
    mpl.pyplot.close("all")


def test_mm_axes_kwargs_sets_extent_and_labels_on_axes():
    grid = Grid((6e-3, 4e-3), 1e6, 1e-6, c0=1540)
    fig, ax = mpl.pyplot.subplots()

    plot_utils.plot_array_on_ax(
        ax,
        np.zeros((grid.nx, grid.ny)),
        **plot_utils.mm_axes_kwargs(grid),
    )

    assert ax.images[0].get_extent() == pytest.approx(grid.extent_mm())
    assert ax.get_xlabel() == "Lateral position (mm)"
    assert ax.get_ylabel() == "Depth (mm)"
    mpl.pyplot.close(fig)


def test_wave_propagation_snapshot_sets_extent_and_labels(tmp_path):
    grid = Grid((6e-3, 4e-3), 1e6, 1e-6, c0=1540)
    field = np.zeros((grid.nx, grid.ny))

    plot_utils.plot_wave_propagation_snapshot(
        propagation_map=field,
        c_map=field + 1500,
        rho_map=field + 1000,
        export_name=tmp_path / "snapshot.png",
        **plot_utils.mm_axes_kwargs(grid),
    )

    ax = mpl.pyplot.gcf().axes[0]
    assert ax.images[0].get_extent() == pytest.approx(grid.extent_mm())
    assert ax.get_xlabel() == "Lateral position (mm)"
    assert ax.get_ylabel() == "Depth (mm)"
    mpl.pyplot.close("all")


def test_wave_propagation_animation_sets_extent_and_labels(tmp_path, monkeypatch):
    grid = Grid((6e-3, 4e-3), 1e6, 1e-6, c0=1540)
    movie = np.zeros((3, grid.nx, grid.ny))
    monkeypatch.setattr(plot_utils.animation.ArtistAnimation, "save", lambda *_, **__: None)
    monkeypatch.setattr(plot_utils.plt, "close", lambda *_: None)

    plot_utils.plot_wave_propagation_animation(
        propagation_map=movie,
        export_name=tmp_path / "animation.mp4",
        num_plot_image=3,
        **plot_utils.mm_axes_kwargs(grid),
    )

    ax = mpl.pyplot.gcf().axes[0]
    assert ax.images[0].get_extent() == pytest.approx(grid.extent_mm())
    assert ax.get_xlabel() == "Lateral position (mm)"
    assert ax.get_ylabel() == "Depth (mm)"
    monkeypatch.undo()
    mpl.pyplot.close("all")
