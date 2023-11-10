import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Callable, Optional
import matplotlib as mpl
from matplotlib import colormaps


def setColors(
    variable_arr: np.ndarray,
    cmap_name: str = "viridis",
    norm_type: str = "log",
) -> tuple[mpl.colors.Colormap, mpl.colors.LogNorm, mpl.cm.ScalarMappable]:
    """
    Returns a colormap, a normalization instance, and a scalar mappable instance.

    Args:
        variable_arr (np.ndarray):
            Array of values to be plotted.
        cmap_name (str, optional):
            Name of the colormap. Defaults to "viridis".

    Returns:
        tuple[mpl.colors.Colormap, mpl.colors.LogNorm, mpl.cm.ScalarMappable]:
            A tuple containing the colormap, normalization instance, and scalar mappable instance.
    """
    cmap = colormaps.get_cmap(cmap_name)
    if norm_type == "log":
        norm = mpl.colors.LogNorm(vmin=variable_arr[0], vmax=variable_arr[-1])
    elif norm_type == "linear":
        norm = mpl.colors.Normalize(vmin=variable_arr[0], vmax=variable_arr[-1])
    else:
        raise ValueError(f"Invalid norm_type: {norm_type}")

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    return cmap, norm, sm


def PlotPredictionPerVariable(
    x_vals: np.ndarray,
    y_vals: list[np.ndarray],
    variable_arr: np.ndarray,
    target_func: Optional[Callable] = None,
    x_label: str = r"$x$",
    y_label: str = r"$y$",
    variable_label: str = r"$\eta$",
    variable_type: str = "log",
    title: str = "Predicted polynomials",
    colormap: str = "viridis",
    n_epochs: int = 500,
    savePlots: bool = False,
    showPlots: bool = True,
    figsPath: Path = Path(__file__).parent.parent / "figures",
) -> None:
    r"""
    Plots predicted polynomials for different values of a variable parameter.

    Args:
        x_vals (np.ndarray):
            Array of x values to plot.
        y_vals (List[np.ndarray]):
            List of arrays of y values to plot, one for each value of the variable parameter.
        variable_arr (np.ndarray):
            Array of values of the variable parameter.
        target_func (Callable, optional):
            Function representing the true target function to plot. Defaults to None.
        x_label (str, optional):
            Label for the x-axis. Defaults to r"$x$".
        y_label (str, optional):
            Label for the y-axis. Defaults to r"$y$".
        variable_label (str, optional):
            Label for the variable parameter. Defaults to r"$\eta$".
        title (str, optional):
            Title for the plot. Defaults to "Predicted polynomials".
        n_epochs (int, optional):
            Number of epochs used for training the model. Defaults to 500.
        savePlots (bool, optional):
            Whether to save the plot as a PNG file. Defaults to False.
        showPlots (bool, optional):
            Whether to display the plot. Defaults to True.
        figsPath (Path, optional):
            Path to the directory where the plot should be saved. Defaults to Path(__file__).parent.parent / "figures".

    Returns:
        None
    """
    fig, ax = plt.subplots()

    cmap, norm, sm = setColors(
        variable_arr, cmap_name=colormap, norm_type=variable_type
    )

    ax.set_xlim(np.min(x_vals), np.max(x_vals))
    ax.set_ylim(np.min(y_vals), np.max(y_vals))

    for i, ynew in enumerate(y_vals):
        ax.plot(x_vals, ynew, color=cmap(norm(variable_arr[i])))

    if target_func is not None:
        ax.plot(x_vals, target_func(x_vals), ":", color="k", label="True")

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(title + rf" $n_{{epochs}}={n_epochs}$")
    plt.legend()
    cbar = plt.colorbar(sm, ax=ax)
    cbar.ax.set_ylabel(variable_label, rotation=45, fontsize="large")
    plt.tight_layout()

    if savePlots:
        plt.savefig(figsPath / f"{title}.png")
    if showPlots:
        plt.show()
    plt.close(fig)


def PlotErrorPerVariable(
    error_vals: list[np.ndarray],
    variable_arr: np.ndarray,
    x_label: str = "epoch",
    error_label: str = "MSE",
    variable_label: str = r"$\eta$",
    variable_type: str = "log",
    title: str = "Error per epoch",
    colormap: str = "viridis",
    savePlots: bool = False,
    showPlots: bool = True,
    figsPath: Path = Path(__file__).parent.parent / "figures",
) -> None:
    r"""
    Plots training error for different values of a variable parameter.

    Args:
        error_vals (List[np.ndarray]):
            List of arrays of error values to plot, one for each value of the variable parameter.
        variable_arr (np.ndarray):
            Array of values of the variable parameter.
        x_label (str, optional):
            Label for the x-axis. Defaults to "epoch".
        error_label (str, optional):
            Label for the y-axis. Defaults to "MSE".
        variable_label (str, optional):
            Label for the variable parameter. Defaults to r"$\eta$".
        title (str, optional):
            Title for the plot. Defaults to "Error per epoch".
        savePlots (bool, optional):
            Whether to save the plot as a PNG file. Defaults to False.
        showPlots (bool, optional):
            Whether to display the plot. Defaults to True.
        figsPath (Path, optional):
            Path to the directory where the plot should be saved. Defaults to Path(__file__).parent.parent / "figures".

    Returns:
        None
    """
    fig, ax = plt.subplots()

    cmap, norm, sm = setColors(
        variable_arr, cmap_name=colormap, norm_type=variable_type
    )

    ax.set_xlim(0, error_vals.shape[1])
    ax.set_ylim(np.min(error_vals), np.max(error_vals))

    for i, error in enumerate(error_vals):
        ax.plot(error, color=cmap(norm(variable_arr[i])))

    ax.set_xlabel(x_label)
    ax.set_ylabel(error_label)
    plt.title(f"{title} ({variable_label})")
    cbar = plt.colorbar(sm)
    cbar.ax.set_ylabel(variable_label, rotation=45, fontsize="large")
    plt.tight_layout()

    if savePlots:
        plt.savefig(figsPath / f"{title}.png")
    if showPlots:
        plt.show()
    plt.close(fig)


def plotThetas(
    theta_arr: np.ndarray,
    variable_arr: np.ndarray,
    variable_type: str = "log",
    true_theta: np.ndarray = None,
    variable_label: str = r"$\eta$",
    title: str = r"Values for $\theta$ for different values of ",
    colormap: str = "viridis",
    savePlots: bool = False,
    showPlots: bool = True,
    figsPath: Path = Path(__file__).parent.parent / "figures",
) -> None:
    r"""
    Plots the values of theta for different values of eta.

    Args:
        theta_arr (np.ndarray):
            Array of theta values for different values.
        x_vals (np.ndarray):
            Array of x values.
        xscale (str, optional):
            Scale of the x-axis. Defaults to "log".
        true_theta (np.ndarray, optional):
            Array of true theta values. Defaults to None.
        x_label (str, optional):
            Label for the x-axis. Defaults to r"$eta$".
        title (str, optional):
            Title of the plot. Defaults to r"Values for $\theta$ for different values of $\eta$".
        savePlots (bool, optional):
            Whether to save the plot. Defaults to False.
        showPlots (bool, optional):
            Whether to show the plot. Defaults to True.
        figsPath (Path, optional):
            Path to the directory where the plot will be saved. Defaults to Path(__file__).parent.parent / "figures".

    Returns:
        None
    """

    tmp_arr = np.linspace(1, 2, theta_arr.shape[1])

    cmap, norm, sm = setColors(tmp_arr, cmap_name=colormap, norm_type=variable_type)

    for i in range(theta_arr.shape[1]):
        plt.plot(variable_arr, theta_arr[:, i], color=cmap(norm(tmp_arr[i])))

    if true_theta is not None:
        for i in range(theta_arr.shape[1]):
            plt.axhline(
                true_theta[i],
                color=cmap(norm(tmp_arr[i])),
                linestyle=":",
                label=rf"$\theta_{{{i}}}$",
            )
        plt.legend()

    title = title + " " + variable_label
    plt.xscale(variable_type)
    plt.xlabel(variable_label)
    plt.ylabel(r"$\theta$")
    plt.title(title)
    if savePlots:
        plt.savefig(figsPath / f"{title}.png")
    if showPlots:
        plt.show()
    plt.close()
