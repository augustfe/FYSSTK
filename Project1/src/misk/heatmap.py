import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from globals import *


def create_heatmap(MSE_test, lambds, title):
    """
    Creates a heatmap plot of the test set MSE given the results of the model for different polynomial degrees and lambda values.

    Parameters
    ----------
    MSE_test : np.array
        A 2D numpy array of shape (n_poly_degrees, n_lambdas) with test set MSE values for different polynomial degrees and lambda values.
    lambds : np.array
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
        # linecolor="black",
        # linewidths=0.8,
        annot=True,
        fmt=".4f",
        cbar=True,
        annot_kws={"fontsize": 8},
        xticklabels=[f"{lmbda:.1f}" for lmbda in np.log10(lambds)],
        yticklabels=degrees,
    )

    ax.set_xlabel(r"$log_{10} \lambda$")
    ax.set_ylabel("Polynomial Degree")  # fontsize=?


    # Set title
    ax.set_title(
        title, fontweight="bold", fontsize=20, pad=25
    )  # fontsize=? fontweihgt='bold'

    fig.tight_layout()

    if savePlots:
        plt.savefig(figsPath / f"Heatmap_{method}.png", dpi=300)
    if showPlots:
        plt.show()
    plt.clf()
