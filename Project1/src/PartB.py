import numpy as np
import matplotlib.pyplot as plt

from Ridge import Ridge
from util import FrankeFunction, create_X, ScaleandCenterData
from sklearn.model_selection import train_test_split

lmbdaVals = [0.0001, 0.001, 0.01, 0.1, 1.0]
maxDim = 5


def RidgeofFranke():
    N = 100
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y) + 0.2 * np.random.randn(N, N)

    MSE_trains = [[] for _ in lmbdaVals]
    MSE_tests = [[] for _ in lmbdaVals]
    R2s = [[] for _ in lmbdaVals]

    for dim in range(maxDim + 1):
        X = create_X(x, y, dim)
        X = ScaleandCenterData(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, ScaleandCenterData(np.ravel(z))
        )
        for i, lmbda in enumerate(lmbdaVals):
            scores = Ridge(X_train, X_test, y_train, y_test, lmbda)
            MSE_trains[i].append(scores[0])
            MSE_tests[i].append(scores[1])
            R2s[i].append(scores[2])

    return MSE_trains, MSE_tests, R2s


def plotScores(MSE_train: list, MSE_test: list, R2: list):
    fig, ax1 = plt.subplots()

    xVals = [i for i in range(maxDim + 1)]

    color = "tab:red"
    ax1.set_xlabel("# of Polynomial dimensions")
    ax1.set_xticks(xVals)
    ax1.set_ylabel("MSE score", color=color)
    for i, lmbda in enumerate(lmbdaVals):
        ax1.plot(xVals, MSE_train[i], label=rf"MSE train $\lambda$={lmbda}")
        ax1.plot(xVals, MSE_test[i], label=rf"MSE test $\lambda$={lmbda}")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel("R2", color=color)
    for i, lmbda in enumerate(lmbdaVals):
        ax2.plot(xVals, R2[i], label=rf"R2 score $\lambda$={lmbda}")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.legend()
    fig.suptitle("Scores by polynomial degree for Ridge")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    scores = RidgeofFranke()
    plotScores(*scores)