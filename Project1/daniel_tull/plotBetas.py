import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# from globals import *
from pathlib import Path


def plotBeta(
    betas,
    title="beta by polynomial degree",
    methodname="OLS",
    polyDegrees=range(1, 14),
    savePlots: bool = False,
    showPlots: bool = True,
    figsPath: Path = None,
):
    """
    Plots the beta values of a linear regression model for different polynomial degrees.

    Parameters
    ----------
    betas : List[List[float]]
        A list of beta values for different polynomial degrees.
        The list should be of length n_poly_degrees, where each element is a numpy array of shape (n_features, 1)
        containing the estimated beta values.
    title : str
        A string containing the title for the plot.
    methodname : str, optional
        A string containing the name of the method used to calculate beta values, by default "OLS".

    """
    for i, degree in enumerate(polyDegrees):
        for beta in betas[i]:
            plt.scatter(degree, beta, c="r", alpha=0.5)  # , "r", alpha=1)

    plt.title(title)
    plt.xticks(polyDegrees)
    plt.xlabel("Polynomial degree")
    plt.ylabel(r"beta_i value")

    tmp = []
    for beta in betas:
        tmp += list(beta.ravel())

    maxBeta = max(abs(min(tmp)), abs(max(tmp))) * 1.2

    plt.ylim((-maxBeta, maxBeta))

    if savePlots:
        plt.savefig(figsPath / f"{methodname}__betas.png", dpi=300)
    if showPlots:
        plt.show()
    plt.close()
