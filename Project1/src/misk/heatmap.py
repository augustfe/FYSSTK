import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def create_heatmap(
    MSE_test, polyDegrees, lambdas, title=None, showPlots=True, savePlots=False, figsPath=None
):
    """
    Creates a heatmap plot of the test set MSE given the results of the model for different polynomial degrees and lambda values.

    Parameters
    ----------
    MSE_test : np.array
        A 2D numpy array of shape (n_poly_degrees, n_lambdas) with test set MSE values for different polynomial degrees and lambda values.
    lambdas : np.array
        A 1D numpy array of lambda values for regularization.
    title : str
        A string containing the title for the heatmap plot.

    Returns
    -------
    None
        The function displays a heatmap plot using Seaborn and Matplotlib.

    """

    # Find the index of the cell with the lowest value
    min_index = np.unravel_index(np.argmin(MSE_test), MSE_test.shape)

    # Create the heatmap using seaborn and highlight the minimum cell
    fig, ax = plt.subplots()
    sns.heatmap(
        MSE_test,
        cmap="coolwarm",
        annot=True,
        fmt=".4f",
        cbar=True,
        annot_kws={"fontsize": 8},
        xticklabels=[f"{lmbda:.1f}" for lmbda in np.log10(lambdas)],
        yticklabels=polyDegrees,
        ax=ax,
    )

    # Add a patch to the minimum cell
    ax.add_patch(
        plt.Rectangle(
            (min_index[1], min_index[0]),
            1,
            1,
            linewidth=3,
            edgecolor="gold",
            facecolor="none",
        )
    )

    ax.set_xlabel(r"$log_{10} \lambda$")
    ax.set_ylabel("Polynomial Degree")
    ax.set_title(title, fontweight="bold", fontsize=20, pad=25)

    fig.tight_layout()

    if savePlots:
        plt.savefig(figsPath / f"Heatmap_{title}.png", dpi=300)
    if showPlots:
        plt.show()
    plt.clf()
