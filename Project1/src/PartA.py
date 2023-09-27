import numpy as np
import matplotlib.pyplot as plt

from util import FrankeFunction, create_X
from sklearn.model_selection import train_test_split
from OLS import OLS


def OLSofFranke(N: int = 500) -> tuple[list[float], list[float], list[float]]:
    """
    Fits a linear regression model to the Franke function using Ordinary Least Squares (OLS), for each
    polynomial degree in the given polyDegrees. Returns a tuple containing the lists MSEs_train,
    MSEs_test and R2s (R-aquared scores)

    Parameters:
    N (int): The number of data points to generate. The default value is 100
    """
    maxDim = 9
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    # x, y = np.meshgrid(x, y)
    # x = x.flatten()
    # y = y.flatten()

    z_true = FrankeFunction(x, y)
    # z = z_true + np.random.randn(N * N) * 0.2
    z = z_true + z_true.mean() * np.random.randn(N) * 0.2

    # print(z_true.mean())

    MSE_trains = []
    MSE_tests = []
    R2s = []

    for dim in range(maxDim + 1):
        X = create_X(x, y, dim)
        # X = ScaleandCenterData(X)

        X_train, X_test, z_train, z_test = train_test_split(X, z, random_state=2018)
        scores = OLS(X_train, X_test, z_train, z_test)
        MSE_trains.append(scores[0])
        MSE_tests.append(scores[1])
        R2s.append(scores[2])

    return MSEs_train, MSEs_test, R2s


def plotScores(MSEs_train: list[float], MSEs_test: list[float], R2s: list[float])-> None:
    """
    Plots MSE_train, MSE_test, and R2 values as a function of polynomial degree

    Args:
    MSE_train(list[float]): MSE of traning data and model prediction
    MSE_test(list[float]): MSE of test data and model prediction
    R2(list[float]): R-squared score of test data and model prediction
    """

    fig, ax1 = plt.subplots()

    xVals = [i for i in range(len(MSE_train))]

    color = "tab:red"
    ax1.set_xlabel("Polynomial dimension")
    ax1.set_xticks(xVals)
    ax1.set_ylabel("MSE score", color=color)
    ax1.plot(xVals, MSEs_train, label="MSE train", color="r")
    ax1.plot(xVals, MSEs_test, label="MSE test", color="g")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel("R2", color=color)
    ax2.plot(xVals, R2s, label="R2 score", color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.legend()
    fig.suptitle("Scores by polynomial degree for OLS")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    np.random.seed(2018)
    scores = OLSofFranke()
    plotScores(*scores)
