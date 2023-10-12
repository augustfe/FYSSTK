from Models import OLS, Ridge, Lasso
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from resampling import bootstrap_polydegrees
from plotBetas import plotBeta
from metrics import *
from globals import *


def plotScores(
    data,
    MSEs_train: np.array,
    MSEs_test: np.array,
    R2s: np.array,
    methodname: str = "OLS",
) -> None:
    """
    Plots MSE_train, MSE_test, and R2 values as a function of polynomial degree

    Parameters:
    -----------
        MSE_train: (np.array)
            MSE of traning data and model prediction

        MSE_test: (np.array)
            MSE of test data and model prediction

        R2: (np.array)
            R-squared score of test data and model prediction

        methodname: (str)
            Name of method used to generate values
    """
    xVals = [i + 1 for i in range(data.maxDim)]

    color = "tab:red"
    plt.xlabel("Polynomial degree")
    plt.xticks(xVals)
    plt.ylabel("MSE score")
    plt.plot(xVals, MSEs_train, label="MSE train", color="r")
    plt.plot(xVals, MSEs_test, label="MSE test", color="g")

    minTestMSE = np.argmin(MSEs_test)
    plt.scatter(
        minTestMSE + 1,
        MSEs_test[minTestMSE],
        marker="x",
        label="Minimum test error",
    )
    plt.title(f"Mean squared error for {methodname}")
    plt.legend()

    if savePlots:
        plt.savefig(data.figs / f"{methodname}_{data.maxDim}_MSE.png", dpi=300)
    if showPlots:
        plt.show()
    plt.clf()
    color = "tab:blue"
    plt.ylabel("$R^2$")
    plt.plot(xVals, R2s, label="$R^2$ score", color=color)

    maxR2 = np.argmax(R2s)
    plt.scatter(maxR2 + 1, R2s[maxR2], marker="x", label="Maximum $R^2$")

    plt.legend()
    plt.title(f"$R^2$ Scores by polynomial degree for {methodname}")

    if savePlots:
        plt.savefig(data.figs / f"{methodname}_{data.maxDim}_R2.png", dpi=300)
    if showPlots:
        plt.show()
    plt.clf()


def OLS_train_test(data, savePlots=False, showPlots=True):
    betas = []
    MSETrain = np.zeros(data.maxDim)
    MSETest = np.zeros(data.maxDim)
    R2Scores = np.zeros(data.maxDim)

    for dim in range(data.maxDim):
        X_train = data.create_X(data.x_train, data.y_train, dim)
        X_test = data.create_X(data.x_test, data.y_test, dim)

        model = OLS()

        beta = model.fit(X_train, data.z_train)
        betas.append(beta)

        z_tilde = X_train @ beta
        z_pred = X_test @ beta

        MSETrain[dim] = MSE(data.z_train, z_tilde)
        MSETest[dim] = MSE(data.z_test, z_pred)
        R2Scores[dim] = R2Score(data.z_test, z_pred)

    plotBeta(betas, "OLS")
    plotScores(data, MSETrain, MSETest, R2Scores, "OLS")


def plot_Bias_VS_Varaince(data):
    """
    Plots variance, error(MSE), and bias using bootstrapping as resampling technique.

    Parameters:
    -----------
    data : ndarray
        A one-dimensional numpy array containing the data set to be analyzed.

    num_samples : int
        Optional, default is 1000. Number of random samples to generate using bootstrapping.

    """
    n_boostraps = 400
    polyDegrees = range(maxDim)
    error, bias, variance = bootstrap_polydegrees(data, polyDegrees, n_boostraps, OLS())
    # print(error)
    # print(bias)
    # print(variance)

    plt.plot(polyDegrees, error, label="Error")
    plt.plot(polyDegrees, bias, label="bias")
    plt.plot(polyDegrees, variance, label="Variance")
    plt.legend()
    plt.show()


if __name__ == "__main__":
    a
