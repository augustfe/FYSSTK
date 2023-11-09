import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Callable, Optional
import matplotlib as mpl
from matplotlib import colormaps


def plotPredictionPerVariable(
    x_vals: np.ndarray,
    y_vals: list[np.ndarray],
    variable_arr: np.ndarray,
    target_func: Optional[Callable] = None,
    x_label: str = r"$x$",
    y_label: str = r"$y$",
    variable_label: str = r"$\eta$",
    title: str = "Predicted polynomials",
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

    cmap = colormaps.get_cmap("viridis")
    norm = mpl.colors.LogNorm(vmin=variable_arr[0], vmax=variable_arr[-1])

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

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
    cbar.ax.set_ylabel(variable_label, rotation=0, fontsize="large")
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
    title: str = "Error per epoch",
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

    cmap = colormaps.get_cmap("viridis")
    norm = mpl.colors.LogNorm(vmin=variable_arr[0], vmax=variable_arr[-1])

    sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    ax.set_xlim(0, error_vals.shape[1])
    ax.set_ylim(np.min(error_vals), np.max(error_vals))

    for i, error in enumerate(error_vals):
        ax.plot(error, color=cmap(norm(variable_arr[i])))

    ax.set_xlabel(x_label)
    ax.set_ylabel(error_label)
    plt.title(f"{title} ({variable_label})")
    plt.legend()
    cbar = plt.colorbar(sm)
    cbar.ax.set_ylabel(variable_label, rotation=0, fontsize="large")
    plt.tight_layout()

    if savePlots:
        plt.savefig(figsPath / f"{title}.png")
    if showPlots:
        plt.show()
    plt.close(fig)


plotPredictionPerVariable(
    np.linspace(0, 1, 100),
    [np.linspace(0, 1, 100), np.logspace(0, 1, 100)],
    np.linspace(1e-8, 1, 2),
    target_func=lambda x: x**2,
    title="test",
    savePlots=False,
    showPlots=True,
)
