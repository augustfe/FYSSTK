import numpy as np
import matplotlib.pyplot as plt

from Ridge import Ridge
from util import FrankeFunction, create_X, ScaleandCenterData
from sklearn.model_selection import train_test_split

lmbdaVals = [0.0001, 0.001, 0.01, 0.1, 1.0]
maxDim = 15


def RidgeofFranke(N: int = 100)-> tuple[list[float], list[float], list[float]]:
    """
    Fits a linear regression model to the Franke function using ridge regression, for each
    polynomial degree less than maxDim. Returns a tuple containing the lists MSEs_train,
    MSEs_test and R2s (R-aquared scores)

    Parameters:
    N (int): The number of data points to generate. The default value is 100
    """
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    # x, y = np.meshgrid(x, y)

    z_true = FrankeFunction(x, y)
    # z = z_true + np.random.randn(N * N) * 0.2
    z = z_true + z_true.mean() * np.random.randn(N) * 0.2

    MSE_trains = [[] for _ in lmbdaVals]
    MSE_tests = [[] for _ in lmbdaVals]
    R2s = [[] for _ in lmbdaVals]

    for dim in range(maxDim + 1):
        X = create_X(x, y, dim)
        # X = ScaleandCenterData(X)

        X_train, X_test, y_train, y_test = train_test_split(X, z, random_state=2018)
        X_train_mean = X_train.mean(axis=0)
        X_train_scaled = X_train - X_train_mean
        X_test_scaled = X_test - X_train_mean

        y_train_mean = y_train.mean()
        y_train_scaled = y_train - y_train_mean
        y_test_scaled = y_test - y_train_mean

        for i, lmbda in enumerate(lmbdaVals):
            scores = Ridge(
                X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, lmbda
            )
            MSE_trains[i].append(scores[0])
            MSE_tests[i].append(scores[1])
            R2s[i].append(scores[2])

    return MSE_trains, MSE_tests, R2s


def plotScores(MSE_train: list[list[float]], MSE_test: list[list[float]], R2: list[list[float]]):
    """
    Plots MSE_train, MSE_test, and R2 values as a function of polynomial
    degree for different lambda using ridge regression.
    """
    fig, ax1 = plt.subplots()

    xVals = [i for i in range(maxDim + 1)]

    color = "tab:red"
    ax1.set_xlabel("Polynomial dimension")
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
    np.random.seed(2018)
    scores = RidgeofFranke()
    plotScores(*scores)
