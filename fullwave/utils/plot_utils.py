"""plot utils."""

from pathlib import Path

import cv2
import numpy as np
from matplotlib import animation
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from tqdm import tqdm


def plot_array(  # noqa: C901, D417, PLR0912
    x: NDArray[np.float64 | np.int64 | np.bool],
    aspect: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    interpolation: str = "none",
    export_path: str | None | Path = None,
    xlim: tuple[float, ...] | None = None,
    ylim: tuple[float, ...] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    *,
    reverse_y_axis: bool = False,
    clear_all: bool = True,
    save: bool = True,
    fig: Figure | None = None,
    colorbar: bool = False,
    plot_grid: bool = False,
    grid_steps: int = 1,
    labelsize: int = 2,
    dpi: int = 300,
    cmap: str = "turbo",
    alpha_array: NDArray[np.float64] | None = None,
    extent: tuple[float, float, float, float] | None = None,
    show: bool = False,
) -> Figure:
    """Plot a 2D array using matplotlib.

    Parameters
    ----------
    x : NDArray[np.float64 | np.int64 | np.bool]
        The input array.
    aspect : float, optional
        The aspect ratio (default is 1 if None).
    vmin : float, optional
        The minimum data value that corresponds to the colormap's lower limit.
    vmax : float, optional
        The maximum data value that corresponds to the colormap's upper limit.
    interpolation : str, optional
        The interpolation method used in imshow (default "none").
    export_path : str or Path, optional
        The file path to export the generated plot.
    xlim : tuple, optional
        The x-axis limits.
    ylim : tuple, optional
        The y-axis limits.
    reverse_y_axis: bool
        if true, it reverses the y-axis to match the image orientation
    clear_all: bool
        if true, it closes all open plots before plotting
    save: bool
        if true, the generated figure will be saved to ecport_path
    fig: Figure
        The figure to plot on. If None, a new figure will be created.
    colorbar: bool
        if true, a colorbar will be added to the plot
    dpi: int
        The resolution in dots per inch (default is 300).
    show: bool
        if true, it displays the plot after creation

    Returns
    -------
    Figure
        The figure object containing the plot.

    """
    if clear_all:
        plt.close("all")

    if fig is None:
        fig = plt.figure()
    else:
        plt.figure(fig.number)

    plt.imshow(
        x,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation=interpolation,
        alpha=alpha_array,
        extent=extent,
    )
    if xlim is not None:
        plt.xlim(*xlim)
    if ylim is not None:
        plt.ylim(*ylim)
    if xlabel:
        plt.xlabel(xlabel)
    if ylabel:
        plt.ylabel(ylabel)
    if reverse_y_axis:
        # reverse the y-axis to match the image orientation
        plt.gca().invert_yaxis()

    aspect = 1 if aspect is None else aspect

    if colorbar:
        plt.colorbar()
    plt.gca().set_aspect(aspect)
    if plot_grid:
        # adjust the axes steps
        plt.xticks(np.arange(0, x.shape[1], grid_steps))
        plt.yticks(np.arange(0, x.shape[0], grid_steps))
        # font size
        plt.tick_params(axis="both", which="major", labelsize=labelsize)
        # plt.xlim(0, x.shape[0] - 1)
        # plt.ylim(x.shape[1] - 1, 0)

        plt.grid(
            visible=True,
            # color="gray",
            # linestyle="--",
            # linewidth=linewidth,
            which="both",
            alpha=0.5,
        )
    if save:
        if export_path is None:
            export_path = Path("./temp/temp.png")
            export_path.parent.mkdir(exist_ok=True, parents=True)
            plt.savefig(export_path, dpi=dpi)
        else:
            plt.savefig(export_path, dpi=dpi)
    if show:
        plt.show()

    if clear_all:
        plt.close("all")

    return plt.gcf()


def plot_array_on_ax(
    ax: Axes,
    x: NDArray[np.float64 | np.int64 | np.bool],
    aspect: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    interpolation: str = "none",
    xlim: tuple[float, ...] | None = None,
    ylim: tuple[float, ...] | None = None,
    title: str = "",
    *,
    reverse_y_axis: bool = False,
    cmap: str = "vanimo",
) -> Axes:
    """Plot a 2D array using matplotlib.

    Parameters
    ----------
    ax: Axes
        The axes to plot on.
    x : NDArray[np.float64 | np.int64 | np.bool]
        The input array.
    aspect : float, optional
        The aspect ratio (default is 1 if None).
    vmin : float, optional
        The minimum data value that corresponds to the colormap's lower limit.
    vmax : float, optional
        The maximum data value that corresponds to the colormap's upper limit.
    interpolation : str, optional
        The interpolation method used in imshow (default "none").
    xlim : tuple, optional
        The x-axis limits.
    ylim : tuple, optional
        The y-axis limits.
    reverse_y_axis: bool
        if true, it reverses the y-axis to match the image orientation
    title: str
        The title of the plot
    cmap: str
        The colormap to use for the plot.

    Returns
    -------
    Axes
        The axes object containing the plot.

    """
    ax.imshow(
        x,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        interpolation=interpolation,
    )
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    if reverse_y_axis:
        # reverse the y-axis to match the image orientation
        ax.invert_yaxis()

    aspect = 1 if aspect is None else aspect
    ax.set_title(title)

    ax.set_aspect(aspect)
    # set color bar for this ax
    ax.figure.colorbar(ax.get_images()[0], ax=ax, shrink=0.75)

    return ax


def plot_1d_array(
    x: NDArray[np.float64 | np.int64 | np.bool],
    vmin: float | None = None,
    vmax: float | None = None,
    export_path: str | Path | None = None,
    title: str = "",
    linewidth: float = 0.5,
    title_font_size: int = 8,
    marker: str | None = None,
    markersize: int = 1,
    *,
    xlim: tuple[float, ...] | None = None,
    ylim: tuple[float, ...] | None = None,
    show: bool = False,
    dpi: int = 300,
) -> None:
    """Plot a 1D array using matplotlib.

    Parameters
    ----------
    x : NDArray[np.float64 | np.int64 | np.bool]
        The input array.
    vmin : float, optional
        The minimum y-value limit.
    vmax : float, optional
        The maximum y-value limit.
    export_path : str or Path, optional
        The file path to export the generated plot.
    title: str
        The title of the plot
    linewidth: float
        The width of the line
    title_font_size: int
        The font size of the title
    marker: str | None
        The marker style for the plot (default is None).
    markersize: int
        The size of the markers (default is 1).
    show: bool
        if true, it displays the plot after creation
    xlim : tuple, optional
        The x-axis limits.
    ylim : tuple, optional
        The y-axis limits.
    dpi : int
        The resolution in dots per inch (default is 300).

    """
    plt.close("all")
    plt.plot(x, linewidth=linewidth, marker=marker, markersize=markersize)
    plt.ylim(*(ylim if ylim is not None else (vmin, vmax)))
    plt.xlim(*(xlim if xlim is not None else (0, len(x) - 1)))
    plt.title(title, fontsize=title_font_size)

    if export_path is None:
        export_path = Path("./temp/temp.png")
        export_path.parent.mkdir(exist_ok=True, parents=True)
        plt.savefig(export_path, dpi=dpi)
    else:
        plt.savefig(export_path, dpi=dpi)
    if show:
        plt.show()


def plot_wave_propagation_animation(
    propagation_map: NDArray[np.float64],
    dpi: int = 300,
    num_plot_image: int = 50,
    export_name: str | Path = "anim.mp4",
    vmin: float | None = None,
    vmax: float | None = None,
    resize_factor: int = 1,
    figsize: tuple = (12, 6),
    cmap: str = "vanimo",
    *,
    plot_pml_boundary: bool = False,
    pml_thickness_px: int = 40,
    n_transition_layer: int = 20,
    m_spatial_order: int = 8,
    linewidth: float = 0.1,
    # plot_grid: bool = False,
    # grid_steps: int = 2,
) -> None:
    """Plot wave propagation with a map overlay using the specified parameters.

    Parameters
    ----------
    propagation_map : NDArray[np.float64]
        The wave propagation data.
        shape = (nt, nx, ny)
    dpi : int, optional
        Resolution in dots per inch.
    num_plot_image : int, optional
        Number of images to include in the animation.
    export_name : str or Path, optional
        The filename for the output animation.
    vmin : float, optional
        The minimum data value for the plot.
    vmax : float, optional
        The maximum data value for the plot.
    resize_factor : int, optional
        Factor to resize each frame.
    plot_pml_boundary : bool, optional
        If True, plot the PML boundary.
    pml_thickness_px : int, optional
        Thickness of the PML in pixels.
    m_spatial_order : int, optional
        Spatial order for the PML.
    linewidth : float, optional
        Width of the lines in the PML area.
    figsize : tuple, optional
        Size of the figure (width, height).
    n_transition_layer : int, optional
        Thickness of the PML transition layer in pixels.
    cmap : str, optional
        The colormap to use for the plot.

    """
    nt = propagation_map.shape[0]
    skip_every_n_frame = int(nt / num_plot_image)
    plt.close()
    plt.cla()
    plt.clf()
    fig, axes = plt.subplots(1, 1, figsize=figsize)

    start = 0
    end = None
    # z_map = c_map * rho_map
    # z_map = (z_map - np.min(z_map)) / (np.max(z_map) - np.min(z_map) + 1e-9)
    # if vmax is None:
    #     vmax = np.max(propagation_map)
    # if vmin is None:
    #     vmin = np.min(propagation_map)

    # z_map_offset = vmax * 0.8

    animation_list = []
    # propagation_map = propagation_map.transpose(2, 0, 1)
    for i, p_map_i in tqdm(
        enumerate(propagation_map[::skip_every_n_frame, start:end, start:end]),
        total=len(propagation_map[::skip_every_n_frame, start:end, start:end]),
        desc="plotting animation",
    ):
        # processed_p_map = p_map_i + z_map_offset * (z_map)
        processed_p_map = p_map_i
        if resize_factor != 1:
            new_width = int(processed_p_map.shape[1] * resize_factor)
            new_height = int(processed_p_map.shape[0] * resize_factor)
            processed_p_map = cv2.resize(processed_p_map, (new_width, new_height))
        image2 = axes.imshow(
            processed_p_map,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            cmap=cmap,
        )
        text = axes.text(
            0.5,
            1.05,
            f"t = {i * skip_every_n_frame} / {propagation_map.shape[0]}",
            fontsize=4,
            ha="center",
            animated=True,
            transform=axes.transAxes,
        )
        if plot_pml_boundary:
            edge_pml_1 = [
                pml_thickness_px + n_transition_layer + m_spatial_order,
                pml_thickness_px + n_transition_layer + m_spatial_order,
            ]
            edge_pml_2 = [
                pml_thickness_px + n_transition_layer + m_spatial_order,
                processed_p_map.shape[1]
                - pml_thickness_px
                - n_transition_layer
                - m_spatial_order
                - 1,
            ]
            edge_pml_3 = [
                processed_p_map.shape[0]
                - pml_thickness_px
                - n_transition_layer
                - m_spatial_order
                - 1,
                processed_p_map.shape[1]
                - pml_thickness_px
                - n_transition_layer
                - m_spatial_order
                - 1,
            ]
            edge_pml_4 = [
                processed_p_map.shape[0]
                - pml_thickness_px
                - n_transition_layer
                - m_spatial_order
                - 1,
                pml_thickness_px + n_transition_layer + m_spatial_order,
            ]
            # put lines in the pml area
            axes.plot(
                [edge_pml_1[1], edge_pml_2[1]],
                [edge_pml_1[0], edge_pml_2[0]],
                color="red",
                linestyle="--",
                linewidth=linewidth,
            )
            axes.plot(
                [edge_pml_2[1], edge_pml_3[1]],
                [edge_pml_2[0], edge_pml_3[0]],
                color="red",
                linestyle="--",
                linewidth=linewidth,
            )
            axes.plot(
                [edge_pml_3[1], edge_pml_4[1]],
                [edge_pml_3[0], edge_pml_4[0]],
                color="red",
                linestyle="--",
                linewidth=linewidth,
            )
            axes.plot(
                [edge_pml_4[1], edge_pml_1[1]],
                [edge_pml_4[0], edge_pml_1[0]],
                color="red",
                linestyle="--",
                linewidth=linewidth,
            )

            edge_ghost_grid_1 = [
                m_spatial_order,
                m_spatial_order,
            ]
            edge_ghost_grid_2 = [
                m_spatial_order,
                processed_p_map.shape[1] - m_spatial_order - 1,
            ]
            edge_ghost_grid_3 = [
                processed_p_map.shape[0] - m_spatial_order - 1,
                processed_p_map.shape[1] - m_spatial_order - 1,
            ]
            edge_ghost_grid_4 = [
                processed_p_map.shape[0] - m_spatial_order - 1,
                m_spatial_order,
            ]
            # put lines in the ghost grid area
            axes.plot(
                [edge_ghost_grid_1[1], edge_ghost_grid_2[1]],
                [edge_ghost_grid_1[0], edge_ghost_grid_2[0]],
                color="blue",
                linestyle="--",
                linewidth=linewidth,
            )
            axes.plot(
                [edge_ghost_grid_2[1], edge_ghost_grid_3[1]],
                [edge_ghost_grid_2[0], edge_ghost_grid_3[0]],
                color="blue",
                linestyle="--",
                linewidth=linewidth,
            )
            axes.plot(
                [edge_ghost_grid_3[1], edge_ghost_grid_4[1]],
                [edge_ghost_grid_3[0], edge_ghost_grid_4[0]],
                color="blue",
                linestyle="--",
                linewidth=linewidth,
            )
            axes.plot(
                [edge_ghost_grid_4[1], edge_ghost_grid_1[1]],
                [edge_ghost_grid_4[0], edge_ghost_grid_1[0]],
                color="blue",
                linestyle="--",
                linewidth=linewidth,
            )

            if n_transition_layer > 0:
                edge_transition_1 = [
                    pml_thickness_px + m_spatial_order + 1,
                    pml_thickness_px + m_spatial_order + 1,
                ]
                edge_transition_2 = [
                    pml_thickness_px + m_spatial_order + 1,
                    processed_p_map.shape[1] - pml_thickness_px - m_spatial_order - 1,
                ]
                edge_transition_3 = [
                    processed_p_map.shape[0] - pml_thickness_px - m_spatial_order - 1,
                    processed_p_map.shape[1] - pml_thickness_px - m_spatial_order - 1,
                ]
                edge_transition_4 = [
                    processed_p_map.shape[0] - pml_thickness_px - m_spatial_order - 1,
                    pml_thickness_px + m_spatial_order + 1,
                ]
                # put lines in the transition area
                axes.plot(
                    [edge_transition_1[1], edge_transition_2[1]],
                    [edge_transition_1[0], edge_transition_2[0]],
                    color="black",
                    linestyle="--",
                    linewidth=linewidth,
                )
                axes.plot(
                    [edge_transition_2[1], edge_transition_3[1]],
                    [edge_transition_2[0], edge_transition_3[0]],
                    color="black",
                    linestyle="--",
                    linewidth=linewidth,
                )
                axes.plot(
                    [edge_transition_3[1], edge_transition_4[1]],
                    [edge_transition_3[0], edge_transition_4[0]],
                    color="black",
                    linestyle="--",
                    linewidth=linewidth,
                )
                axes.plot(
                    [edge_transition_4[1], edge_transition_1[1]],
                    [edge_transition_4[0], edge_transition_1[0]],
                    color="black",
                    linestyle="--",
                    linewidth=linewidth,
                )
        animation_list.append([image2, text])
    animation_data = animation.ArtistAnimation(
        fig,
        animation_list,
        interval=150,
        blit=True,
        repeat_delay=500,
    )
    animation_data.save(export_name, writer="ffmpeg", dpi=dpi)


def plot_wave_propagation_with_map(  # noqa: PLR0915, C901
    propagation_map: NDArray[np.float64],
    c_map: NDArray[np.float64],
    rho_map: NDArray[np.float64],
    dpi: int = 300,
    num_plot_image: int = 50,
    export_name: str | Path = "anim.mp4",
    vmin: float | None = None,
    vmax: float | None = None,
    resize_factor: int = 1,
    figsize: tuple = (12, 6),
    xlim: tuple[float, ...] | None = None,
    ylim: tuple[float, ...] | None = None,
    xlabel: str = "",
    ylabel: str = "",
    *,
    plot_pml_boundary: bool = False,
    pml_thickness_px: int = 40,
    n_transition_layer: int = 20,
    m_spatial_order: int = 8,
    linewidth: float = 0.1,
    plot_grid: bool = False,
    grid_steps: int = 2,
    extent: tuple[float, float, float, float] | None = None,
) -> None:
    """Plot wave propagation with a map overlay using the specified parameters.

    Parameters
    ----------
    propagation_map : NDArray[np.float64]
        The wave propagation data.
        shape = (nt, nx, ny)
    c_map : NDArray[np.float64]
        The speed of sound map.
    rho_map : NDArray[np.float64]
        The density map.
    dpi : int, optional
        Resolution in dots per inch.
    num_plot_image : int, optional
        Number of images to include in the animation.
    export_name : str or Path, optional
        The filename for the output animation.
    vmin : float, optional
        The minimum data value for the plot.
    vmax : float, optional
        The maximum data value for the plot.
    resize_factor : int, optional
        Factor to resize each frame.
    plot_pml_boundary : bool, optional
        If True, plot the PML boundary.
    pml_thickness_px : int, optional
        Thickness of the PML in pixels.
    m_spatial_order : int, optional
        Spatial order for the PML.
    linewidth : float, optional
        Width of the lines in the PML area.
    figsize : tuple, optional
        Size of the figure (width, height).
    n_transition_layer : int, optional
        Thickness of the PML transition layer in pixels.
    grid_steps : int, optional
        Steps for the grid lines.
    plot_grid : bool, optional
        If True, plot the grid lines.
    extent : tuple, optional
        The extent of the axes (left, right, bottom, top).
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    xlim : tuple, optional
        The x-axis limits.
    ylim : tuple, optional
        The y-axis limits.

    """
    nt = propagation_map.shape[0]
    skip_every_n_frame = int(nt / num_plot_image)
    plt.close()
    plt.cla()
    plt.clf()
    fig, axes = plt.subplots(1, 1, figsize=figsize)

    start = 0
    end = None
    z_map = c_map * rho_map
    z_map = (z_map - np.min(z_map)) / (np.max(z_map) - np.min(z_map) + 1e-9)
    if vmax is None:
        vmax = np.max(propagation_map)
    if vmin is None:
        vmin = np.min(propagation_map)

    z_map_offset = vmax * 0.8

    animation_list = []
    # propagation_map = propagation_map.transpose(2, 0, 1)
    for i, p_map_i in tqdm(
        enumerate(propagation_map[::skip_every_n_frame, start:end, start:end]),
        total=len(propagation_map[::skip_every_n_frame, start:end, start:end]),
        desc="plotting animation",
    ):
        processed_p_map = p_map_i + z_map_offset * (z_map)
        if resize_factor != 1:
            new_width = int(processed_p_map.shape[1] * resize_factor)
            new_height = int(processed_p_map.shape[0] * resize_factor)
            processed_p_map = cv2.resize(processed_p_map, (new_width, new_height))
        image2 = axes.imshow(
            processed_p_map,
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            extent=extent,
        )
        if xlim is not None:
            plt.xlim(*xlim)
        if ylim is not None:
            plt.ylim(*ylim)
        if xlabel:
            plt.xlabel(xlabel)
        if ylabel:
            plt.ylabel(ylabel)
        # set text to show the current time step
        text = axes.text(
            0.5,
            1.05,
            f"t = {i * skip_every_n_frame} / {propagation_map.shape[0]}",
            fontsize=4,
            ha="center",
            animated=True,
            transform=axes.transAxes,
        )
        if plot_pml_boundary:
            edge_pml_1 = [
                pml_thickness_px + n_transition_layer + m_spatial_order,
                pml_thickness_px + n_transition_layer + m_spatial_order,
            ]
            edge_pml_2 = [
                pml_thickness_px + n_transition_layer + m_spatial_order,
                processed_p_map.shape[1]
                - pml_thickness_px
                - n_transition_layer
                - m_spatial_order
                - 1,
            ]
            edge_pml_3 = [
                processed_p_map.shape[0]
                - pml_thickness_px
                - n_transition_layer
                - m_spatial_order
                - 1,
                processed_p_map.shape[1]
                - pml_thickness_px
                - n_transition_layer
                - m_spatial_order
                - 1,
            ]
            edge_pml_4 = [
                processed_p_map.shape[0]
                - pml_thickness_px
                - n_transition_layer
                - m_spatial_order
                - 1,
                pml_thickness_px + n_transition_layer + m_spatial_order,
            ]
            # put lines in the pml area
            axes.plot(
                [edge_pml_1[1], edge_pml_2[1]],
                [edge_pml_1[0], edge_pml_2[0]],
                color="red",
                linestyle="--",
                linewidth=linewidth,
            )
            axes.plot(
                [edge_pml_2[1], edge_pml_3[1]],
                [edge_pml_2[0], edge_pml_3[0]],
                color="red",
                linestyle="--",
                linewidth=linewidth,
            )
            axes.plot(
                [edge_pml_3[1], edge_pml_4[1]],
                [edge_pml_3[0], edge_pml_4[0]],
                color="red",
                linestyle="--",
                linewidth=linewidth,
            )
            axes.plot(
                [edge_pml_4[1], edge_pml_1[1]],
                [edge_pml_4[0], edge_pml_1[0]],
                color="red",
                linestyle="--",
                linewidth=linewidth,
            )

            edge_ghost_grid_1 = [
                m_spatial_order,
                m_spatial_order,
            ]
            edge_ghost_grid_2 = [
                m_spatial_order,
                processed_p_map.shape[1] - m_spatial_order - 1,
            ]
            edge_ghost_grid_3 = [
                processed_p_map.shape[0] - m_spatial_order - 1,
                processed_p_map.shape[1] - m_spatial_order - 1,
            ]
            edge_ghost_grid_4 = [
                processed_p_map.shape[0] - m_spatial_order - 1,
                m_spatial_order,
            ]
            # put lines in the ghost grid area
            axes.plot(
                [edge_ghost_grid_1[1], edge_ghost_grid_2[1]],
                [edge_ghost_grid_1[0], edge_ghost_grid_2[0]],
                color="blue",
                linestyle="--",
                linewidth=linewidth,
            )
            axes.plot(
                [edge_ghost_grid_2[1], edge_ghost_grid_3[1]],
                [edge_ghost_grid_2[0], edge_ghost_grid_3[0]],
                color="blue",
                linestyle="--",
                linewidth=linewidth,
            )
            axes.plot(
                [edge_ghost_grid_3[1], edge_ghost_grid_4[1]],
                [edge_ghost_grid_3[0], edge_ghost_grid_4[0]],
                color="blue",
                linestyle="--",
                linewidth=linewidth,
            )
            axes.plot(
                [edge_ghost_grid_4[1], edge_ghost_grid_1[1]],
                [edge_ghost_grid_4[0], edge_ghost_grid_1[0]],
                color="blue",
                linestyle="--",
                linewidth=linewidth,
            )

            if n_transition_layer > 0:
                edge_transition_1 = [
                    pml_thickness_px + m_spatial_order + 1,
                    pml_thickness_px + m_spatial_order + 1,
                ]
                edge_transition_2 = [
                    pml_thickness_px + m_spatial_order + 1,
                    processed_p_map.shape[1] - pml_thickness_px - m_spatial_order - 1,
                ]
                edge_transition_3 = [
                    processed_p_map.shape[0] - pml_thickness_px - m_spatial_order - 1,
                    processed_p_map.shape[1] - pml_thickness_px - m_spatial_order - 1,
                ]
                edge_transition_4 = [
                    processed_p_map.shape[0] - pml_thickness_px - m_spatial_order - 1,
                    pml_thickness_px + m_spatial_order + 1,
                ]
                # put lines in the transition area
                axes.plot(
                    [edge_transition_1[1], edge_transition_2[1]],
                    [edge_transition_1[0], edge_transition_2[0]],
                    color="black",
                    linestyle="--",
                    linewidth=linewidth,
                )
                axes.plot(
                    [edge_transition_2[1], edge_transition_3[1]],
                    [edge_transition_2[0], edge_transition_3[0]],
                    color="black",
                    linestyle="--",
                    linewidth=linewidth,
                )
                axes.plot(
                    [edge_transition_3[1], edge_transition_4[1]],
                    [edge_transition_3[0], edge_transition_4[0]],
                    color="black",
                    linestyle="--",
                    linewidth=linewidth,
                )
                axes.plot(
                    [edge_transition_4[1], edge_transition_1[1]],
                    [edge_transition_4[0], edge_transition_1[0]],
                    color="black",
                    linestyle="--",
                    linewidth=linewidth,
                )
        if plot_grid:
            # adjust the axes steps
            axes.set_xticks(np.arange(0, processed_p_map.shape[1], grid_steps))
            axes.set_yticks(np.arange(0, processed_p_map.shape[0], grid_steps))
            # font size
            axes.tick_params(axis="both", which="major", labelsize=4)
            axes.set_xlim(0, processed_p_map.shape[1] - 1)
            axes.set_ylim(processed_p_map.shape[0] - 1, 0)

            axes.grid(
                visible=True,
                # color="gray",
                # linestyle="--",
                # linewidth=linewidth,
                which="both",
                alpha=0.5,
            )
        animation_list.append([image2, text])
    animation_data = animation.ArtistAnimation(
        fig,
        animation_list,
        interval=150,
        blit=True,
        repeat_delay=500,
    )
    animation_data.save(export_name, writer="ffmpeg", dpi=dpi)
    plt.close("all")


def plot_wave_propagation_snapshot(  # noqa: PLR0915
    propagation_map: NDArray[np.float64],
    c_map: NDArray[np.float64],
    rho_map: NDArray[np.float64],
    dpi: int = 300,
    export_name: str | Path = "./temp/temp.png",
    vmin: float | None = None,
    vmax: float | None = None,
    resize_factor: int = 1,
    figsize: tuple = (12, 6),
    *,
    plot_pml_boundary: bool = False,
    pml_thickness_px: int = 40,
    n_transition_layer: int = 20,
    m_spatial_order: int = 8,
    linewidth: float = 0.1,
    plot_grid: bool = False,
    grid_steps: int = 2,
    turn_off_axes: bool = False,
) -> None:
    """Plot wave propagation with a map overlay using the specified parameters.

    Parameters
    ----------
    propagation_map : NDArray[np.float64]
        The wave propagation data.
        shape = (nt, nx, ny)
    c_map : NDArray[np.float64]
        The speed of sound map.
    rho_map : NDArray[np.float64]
        The density map.
    dpi : int, optional
        Resolution in dots per inch.
    export_name : str or Path, optional
        The filename for the output animation.
    vmin : float, optional
        The minimum data value for the plot.
    vmax : float, optional
        The maximum data value for the plot.
    resize_factor : int, optional
        Factor to resize each frame.
    plot_pml_boundary : bool, optional
        If True, plot the PML boundary.
    pml_thickness_px : int, optional
        Thickness of the PML in pixels.
    m_spatial_order : int, optional
        Spatial order for the PML.
    linewidth : float, optional
        Width of the lines in the PML area.
    figsize : tuple, optional
        Size of the figure (width, height).
    n_transition_layer : int, optional
        Thickness of the PML transition layer in pixels.
    grid_steps : int, optional
        Steps for the grid lines.
    plot_grid : bool, optional
        If True, plot the grid lines.
    turn_off_axes : bool, optional
        If True, turn off the axes.

    """
    plt.close()
    plt.cla()
    plt.clf()
    _, axes = plt.subplots(1, 1, figsize=figsize)

    z_map = c_map * rho_map
    z_map = (z_map - np.min(z_map)) / (np.max(z_map) - np.min(z_map) + 1e-9)
    if vmax is None:
        vmax = np.max(propagation_map)
    if vmin is None:
        vmin = np.min(propagation_map)

    z_map_offset = vmax * 0.8

    p_map_i = propagation_map
    processed_p_map = p_map_i + z_map_offset * (z_map)
    if resize_factor != 1:
        new_width = int(processed_p_map.shape[1] * resize_factor)
        new_height = int(processed_p_map.shape[0] * resize_factor)
        processed_p_map = cv2.resize(processed_p_map, (new_width, new_height))
    axes.imshow(
        processed_p_map,
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    # set text to show the current time step

    if plot_pml_boundary:
        edge_pml_1 = [
            pml_thickness_px + n_transition_layer + m_spatial_order,
            pml_thickness_px + n_transition_layer + m_spatial_order,
        ]
        edge_pml_2 = [
            pml_thickness_px + n_transition_layer + m_spatial_order,
            processed_p_map.shape[1] - pml_thickness_px - n_transition_layer - m_spatial_order - 1,
        ]
        edge_pml_3 = [
            processed_p_map.shape[0] - pml_thickness_px - n_transition_layer - m_spatial_order - 1,
            processed_p_map.shape[1] - pml_thickness_px - n_transition_layer - m_spatial_order - 1,
        ]
        edge_pml_4 = [
            processed_p_map.shape[0] - pml_thickness_px - n_transition_layer - m_spatial_order - 1,
            pml_thickness_px + n_transition_layer + m_spatial_order,
        ]
        # put lines in the pml area
        axes.plot(
            [edge_pml_1[1], edge_pml_2[1]],
            [edge_pml_1[0], edge_pml_2[0]],
            color="red",
            linestyle="--",
            linewidth=linewidth,
        )
        axes.plot(
            [edge_pml_2[1], edge_pml_3[1]],
            [edge_pml_2[0], edge_pml_3[0]],
            color="red",
            linestyle="--",
            linewidth=linewidth,
        )
        axes.plot(
            [edge_pml_3[1], edge_pml_4[1]],
            [edge_pml_3[0], edge_pml_4[0]],
            color="red",
            linestyle="--",
            linewidth=linewidth,
        )
        axes.plot(
            [edge_pml_4[1], edge_pml_1[1]],
            [edge_pml_4[0], edge_pml_1[0]],
            color="red",
            linestyle="--",
            linewidth=linewidth,
        )

        edge_ghost_grid_1 = [
            m_spatial_order,
            m_spatial_order,
        ]
        edge_ghost_grid_2 = [
            m_spatial_order,
            processed_p_map.shape[1] - m_spatial_order - 1,
        ]
        edge_ghost_grid_3 = [
            processed_p_map.shape[0] - m_spatial_order - 1,
            processed_p_map.shape[1] - m_spatial_order - 1,
        ]
        edge_ghost_grid_4 = [
            processed_p_map.shape[0] - m_spatial_order - 1,
            m_spatial_order,
        ]
        # put lines in the ghost grid area
        axes.plot(
            [edge_ghost_grid_1[1], edge_ghost_grid_2[1]],
            [edge_ghost_grid_1[0], edge_ghost_grid_2[0]],
            color="blue",
            linestyle="--",
            linewidth=linewidth,
        )
        axes.plot(
            [edge_ghost_grid_2[1], edge_ghost_grid_3[1]],
            [edge_ghost_grid_2[0], edge_ghost_grid_3[0]],
            color="blue",
            linestyle="--",
            linewidth=linewidth,
        )
        axes.plot(
            [edge_ghost_grid_3[1], edge_ghost_grid_4[1]],
            [edge_ghost_grid_3[0], edge_ghost_grid_4[0]],
            color="blue",
            linestyle="--",
            linewidth=linewidth,
        )
        axes.plot(
            [edge_ghost_grid_4[1], edge_ghost_grid_1[1]],
            [edge_ghost_grid_4[0], edge_ghost_grid_1[0]],
            color="blue",
            linestyle="--",
            linewidth=linewidth,
        )

        if n_transition_layer > 0:
            edge_transition_1 = [
                pml_thickness_px + m_spatial_order + 1,
                pml_thickness_px + m_spatial_order + 1,
            ]
            edge_transition_2 = [
                pml_thickness_px + m_spatial_order + 1,
                processed_p_map.shape[1] - pml_thickness_px - m_spatial_order - 1,
            ]
            edge_transition_3 = [
                processed_p_map.shape[0] - pml_thickness_px - m_spatial_order - 1,
                processed_p_map.shape[1] - pml_thickness_px - m_spatial_order - 1,
            ]
            edge_transition_4 = [
                processed_p_map.shape[0] - pml_thickness_px - m_spatial_order - 1,
                pml_thickness_px + m_spatial_order + 1,
            ]
            # put lines in the transition area
            axes.plot(
                [edge_transition_1[1], edge_transition_2[1]],
                [edge_transition_1[0], edge_transition_2[0]],
                color="black",
                linestyle="--",
                linewidth=linewidth,
            )
            axes.plot(
                [edge_transition_2[1], edge_transition_3[1]],
                [edge_transition_2[0], edge_transition_3[0]],
                color="black",
                linestyle="--",
                linewidth=linewidth,
            )
            axes.plot(
                [edge_transition_3[1], edge_transition_4[1]],
                [edge_transition_3[0], edge_transition_4[0]],
                color="black",
                linestyle="--",
                linewidth=linewidth,
            )
            axes.plot(
                [edge_transition_4[1], edge_transition_1[1]],
                [edge_transition_4[0], edge_transition_1[0]],
                color="black",
                linestyle="--",
                linewidth=linewidth,
            )
    if plot_grid:
        # adjust the axes steps
        axes.set_xticks(np.arange(0, processed_p_map.shape[1], grid_steps))
        axes.set_yticks(np.arange(0, processed_p_map.shape[0], grid_steps))
        # font size
        axes.tick_params(axis="both", which="major", labelsize=4)
        axes.set_xlim(0, processed_p_map.shape[1] - 1)
        axes.set_ylim(processed_p_map.shape[0] - 1, 0)

        axes.grid(
            visible=True,
            which="both",
            alpha=0.5,
        )
    if turn_off_axes:
        axes.set_xticks([])
        axes.set_yticks([])
        axes.set_xlabel("")
        axes.set_ylabel("")
        axes.set_title("")
    plt.tight_layout()
    plt.savefig(export_name, dpi=dpi)
