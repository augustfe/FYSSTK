import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from globals import *


def plotBeta(betas, title, methodname="OLS"):
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
    dims = range(len(betas))
    for dim in dims:
        for beta in betas[dim]:
            plt.scatter(dim + 1, beta, c="r", alpha=0.5)  # , "r", alpha=1)

    plt.title(title)
    plt.xticks([dim + 1 for dim in dims])
    plt.xlabel("Polynomial degree")
    plt.ylabel(r"$\beta_i$ value")

    tmp = []
    for beta in betas:
        tmp += list(beta.ravel())

    maxBeta = max(abs(min(tmp)), abs(max(tmp))) * 1.2

    plt.ylim((-maxBeta, maxBeta))

    if savePlots:
        plt.savefig(figsPath / f"{methodname}_{maxDim}_betas.png", dpi=300)
    if showPlots:
        plt.show()
    plt.close()
