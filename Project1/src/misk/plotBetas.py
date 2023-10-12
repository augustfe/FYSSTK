import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from globals import*

def plotBeta(betas, title, showPlots = True,savePlots = False, methodname = "OLS"):
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
