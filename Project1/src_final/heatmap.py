import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def create_heatmap(
    MSE_test,
    lmbds=np.linspace(-3, 5, 10),
    savePlots: bool = False,
    showPlots: bool = True,
    figsPath: Path = Path(".").parent,
    title: str = None,
):
    """Create heatmap of MSE for lambda vs degree.

    Creates a heatmap plot of the test set MSE given the results of the model for
    different polynomial degrees and lambda values.

    Parameters
    ----------
    MSE_test : np.array
        A 2D numpy array of shape (n_poly_degrees, n_lambdas) with test set MSE
        values for different polynomial degrees and lambda values.
    lmbds : np.array
        A 1D numpy array of lambda values for regularization.
    title : str
        A string containing the title for the heatmap plot.

    Returns
    -------
    None
        The function displays a heatmap plot using Seaborn and Matplotlib.
    """

    # Define polynomial degrees and lambda values
    degrees = np.arange(1, len(MSE_test) + 1)

    fig, ax = plt.subplots()

    ax = sns.heatmap(
        MSE_test,
        cmap="coolwarm",
        annot=True,
        fmt=".4f",
        cbar=True,
        annot_kws={"fontsize": 8},
        xticklabels=[f"{lmbda:.1f}" for lmbda in np.log10(lmbds)],
        yticklabels=degrees,
    )

    ax.set_xlabel(r"$log_{10} \lambda$")
    ax.set_ylabel("Polynomial Degree")

    # Set title
    if title is None:
        if savePlots:
            raise ValueError("Attempting to save file without setting title!")
        title = "MSE"

    ax.set_title(title, fontweight="bold", fontsize=20, pad=25)

    fig.tight_layout()

    if savePlots:
        plt.savefig(figsPath / f"Heatmap_{title}.png", dpi=300)
    if showPlots:
        plt.show()
    plt.clf()
