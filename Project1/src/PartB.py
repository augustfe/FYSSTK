import numpy as np

from Ridge import Ridge
from util import FrankeFunction, create_X, ScaleandCenterData
from sklearn.model_selection import train_test_split


def RidgeofFranke():
    lmbdaVals = [0.0001, 0.001, 0.01, 0.1, 1.0]
    maxDim = 5

    N = 100
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))
    x, y = np.meshgrid(x, y)
    z = FrankeFunction(x, y) + 0.2 * np.random.randn(N, N)

    MSE_trains = [[] * len(lmbdaVals)]
    MSE_tests = [[] * len(lmbdaVals)]
    R2s = [[] * len(lmbdaVals)]

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
