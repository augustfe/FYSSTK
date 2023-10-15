from Models import OLS, Ridge, Lasso
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from tqdm import tqdm
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from resampling import bootstrap_degrees, kfold_score_degrees
from plotBetas import plotBeta
from metrics import MSE, R2Score

# from globals import *


def plotScores(
    data,
    MSEs_train: np.array,
    MSEs_test: np.array,
    R2s: np.array,
    methodname: str = "OLS",
    title: str = "",
    polyDegrees=range(1, 14),
    showPlots: bool = True,
    savePlots: bool = False,
    figsPath=None,
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
    xVals = polyDegrees

    color = "tab:red"
    plt.xlabel("Polynomial degree")
    plt.xticks(xVals)
    plt.ylabel("MSE score")
    plt.plot(xVals, MSEs_train, label="MSE train", color="r")
    plt.plot(xVals, MSEs_test, label="MSE test", color="g")

    #HERE
    minTestMSE = np.argmin(MSEs_test)
    plt.scatter(
        xVals[minTestMSE],
        MSEs_test[minTestMSE],
        marker="x",
        label="Minimum test error",
    )
    if title == "":
        title = "MSE OLS (no resampling)"
    plt.title(title)
    plt.legend()

    if savePlots:
        plt.savefig(figsPath / f"{methodname}_MSE_by_degree.png", dpi=300)
    if showPlots:
        plt.show()
    plt.clf()
    color = "tab:blue"
    plt.ylabel("$R^2$")
    plt.plot(xVals, R2s, label="$R^2$ score", color=color)

    maxR2 = np.argmax(R2s)
    plt.scatter(xVals[maxR2], R2s[maxR2], marker="x", label="Maximum $R^2$")

    plt.legend()
    plt.title(f"$R^2$ Scores by polynomial degree for {methodname}")

    if savePlots:
        plt.savefig(figsPath / f"{methodname}_R2_by_degree.png", dpi=300)
    if showPlots:
        plt.show()
    plt.clf()


def OLS_train_test(data, **kwargs):
    betas = []

    polyDegrees = kwargs["polyDegrees"]
    ndegrees = len(polyDegrees)

    MSETrain = np.zeros(ndegrees)
    MSETest = np.zeros(ndegrees)
    R2Scores = np.zeros(ndegrees)

    pbar = tqdm(total=ndegrees, desc="OLS MSEs")
    for i, dim in enumerate(polyDegrees):
        model = OLS()
        X_train = model.create_X(data.x_train, data.y_train, dim)
        X_test = model.create_X(data.x_test, data.y_test, dim)

        beta = model.fit(X_train, data.z_train)
        betas.append(beta)

        z_tilde = X_train @ beta
        z_pred = X_test @ beta

        MSETrain[i] = MSE(data.z_train, z_tilde)
        MSETest[i] = MSE(data.z_test, z_pred)
        R2Scores[i] = R2Score(data.z_test, z_pred)
        pbar.update(1)

    plotBeta(betas, methodname="OLS", **kwargs)
    plotScores(data, MSETrain, MSETest, R2Scores, methodname="OLS", **kwargs)


def plot_Bias_VS_Variance(
    data,
    polyDegrees=range(1, 14),
    showPlots=True,
    savePlots=False,
    figsPath=None,
    title="Bias Variance Tradeoff OLS",
):
    """
    Plots variance, error(MSE), and bias using bootstrapping as resampling technique.

    Parameters:
    -----------
    data : ndarray
        A one-dimensional numpy array containing the data set to be analyzed.

    num_samples : int
        Optional, default is 1000. Number of random samples to generate using bootstrapping.

    """
    n_bootstraps = 400

    error, bias, variance = bootstrap_degrees(
        data=data, polyDegrees=polyDegrees, n_bootstraps=n_bootstraps, model=OLS()
    )

    plt.plot(polyDegrees, error, label="Error")
    plt.plot(polyDegrees, bias, label="bias")
    plt.plot(polyDegrees, variance, label="Variance")
    plt.title(title)
    plt.legend()
    if savePlots:
        plt.savefig(figsPath / f"{title}.png", dpi=300)
    if showPlots:
        plt.show()
    plt.clf()


def bootstrap_vs_cross_val_OLS(
    data,
    polyDegrees=range(1, 14),
    savePlots: bool = False,
    showPlots: bool = True,
    figsPath: Path = None,
):
    """
    Here we wish to compare our bootstrap to cross val.
    guessing that plotting the variance, bias and errors in the same plot
    is fine
    """
    error_kfold, variance_kfold = kfold_score_degrees(
        data, polyDegrees=polyDegrees, kfolds=5, model=OLS()
    )
    error_boot, bias_boot, variance_boot = bootstrap_degrees(
        data, polyDegrees=polyDegrees, n_bootstraps=100
    )
    # error_CV, varaince_CV = sklearn_cross_val_OLS(x, y, z, polyDegrees, kfolds)
    # print(error_kfold)
    # print(error_boot)
    # plt.plot(polyDegrees, error_boot, label="Bootstrap Error")
    # plt.plot(polyDegrees, variance_boot, label="Bootstrap Variance")
    plt.plot(polyDegrees, error_kfold, "b", label="Kfold CV Error")
    # plt.plot(polyDegrees, variance_kfold, label="Kfold variance")
    plt.plot(polyDegrees, error_boot, "r--", label="Bootstrap Error")
    plt.title("Bootstrap vs CV")
    plt.legend()
    if savePlots:
        plt.savefig(figsPath / f"BootstrapVScrossValOLS.png", dpi=300)
    if showPlots:
        plt.show()
    plt.clf()
