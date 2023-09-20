import numpy as np
import matplotlib.pyplot as plt

from util import FrankeFunction, create_X, ScaleandCenterData
from sklearn.model_selection import train_test_split
from OLS import OLS


def OLSofFranke() -> tuple[list, list, list]:
    maxDim = 5
    N = 100
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y) + 0.2 * np.random.randn(N, N)

    MSE_trains = []
    MSE_tests = []
    R2s = []

    for dim in range(maxDim + 1):
        X = create_X(x, y, dim)
        X = ScaleandCenterData(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, ScaleandCenterData(np.ravel(z))
        )

        scores = OLS(X_train, X_test, y_train, y_test)
        MSE_trains.append(scores[0])
        MSE_tests.append(scores[1])
        R2s.append(scores[2])

    return MSE_trains, MSE_tests, R2s


def plotScores(MSE_train, MSE_test, R2):
    fig, ax1 = plt.subplots()

    xVals = [i for i in range(len(MSE_train))]

    color = "tab:red"
    ax1.set_xlabel("# of Polynomial dimensions")
    ax1.set_xticks(xVals)
    ax1.set_ylabel("MSE score", color=color)
    ax1.plot(xVals, MSE_train, label="MSE train", color="r")
    ax1.plot(xVals, MSE_test, label="MSE test", color="g")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel("R2", color=color)
    ax2.plot(xVals, R2, label="R2 score", color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    fig.legend()
    fig.suptitle("Scores by polynomial degree for OLS")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    scores = OLSofFranke()
    plotScores(*scores)
