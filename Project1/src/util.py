import numpy as np


def MSE(y: np.array, y_pred: np.array) -> float:
    "Calculate mean squared error"
    n = y.shape[0]
    return np.sum((y - y_pred) ** 2) / n


def R2Score(y: np.array, y_pred: np.array) -> float:
    "Calculate R2 score"
    s1 = np.sum((y - y_pred) ** 2)
    m = np.sum(y_pred) / y_pred.shape[0]
    s2 = np.sum((y - m) ** 2)

    return 1 - s1 / s2


def FrankeFunction(x: np.array, y: np.array) -> np.array:
    "Franke function for evaluating methods"
    term1 = 0.75 * np.exp(-(0.25 * (9 * x - 2) ** 2) - 0.25 * ((9 * y - 2) ** 2))
    term2 = 0.75 * np.exp(-((9 * x + 1) ** 2) / 49.0 - 0.1 * (9 * y + 1))
    term3 = 0.5 * np.exp(-((9 * x - 7) ** 2) / 4.0 - 0.25 * ((9 * y - 3) ** 2))
    term4 = -0.2 * np.exp(-((9 * x - 4) ** 2) - (9 * y - 7) ** 2)
    return term1 + term2 + term3 + term4


def create_X(x: np.array, y: np.array, n: int) -> np.ndarray:
    "Create design matrix"
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    lgth = int((n + 1) * (n + 2) / 2)
    X = np.ones((N, lgth))

    for i in range(1, n + 1):
        q = int((i) * (i + 1) / 2)
        for k in range(i + 1):
            X[:, q + k] = (x ** (i - k)) * (y**k)

    return X


def ScaleandCenterData(X: np.ndarray) -> np.ndarray:
    "Scale and center the given data"
    X /= np.max(np.abs(X), axis=0)
    X -= np.mean(X, axis=0)
    return X
