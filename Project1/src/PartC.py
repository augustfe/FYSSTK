import numpy as np
import matplotlib.pyplot as plt

from Lasso import pureSKLearnLasso
from util import FrankeFunction, ScaleandCenterData, create_X
from sklearn.model_selection import train_test_split

alphaVals = [0.0001, 0.001, 0.01, 0.1, 1.0]
maxDim = 5


def LassoofFranke():
    N = 100
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y) + 0.2 * np.random.randn(N, N)

    MSE_trains = [[] for _ in alphaVals]
    MSE_tests = [[] for _ in alphaVals]
    R2s = [[] for _ in alphaVals]

    for dim in range(maxDim + 1):
        X = create_X(x, y, dim)
        X = ScaleandCenterData(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, ScaleandCenterData(np.ravel(z))
        )
        for i, alpha in enumerate(alphaVals):
            scores = pureSKLearnLasso(X_train, X_test, y_train, y_test, alpha)
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
    for i, alpha in enumerate(alphaVals):
        ax1.plot(xVals, MSE_train[i], label=rf"MSE train $\alpha$={alpha}")
        ax1.plot(xVals, MSE_test[i], label=rf"MSE test $\alpha$={alpha}")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel("R2", color=color)
    for i, alpha in enumerate(alphaVals):
        ax2.plot(xVals, R2[i], label=rf"R2 score $\alpha$={alpha}")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.legend()
    fig.suptitle("Scores by polynomial degree for Lasso")
    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    scores = LassoofFranke()
    plotScores(*scores)