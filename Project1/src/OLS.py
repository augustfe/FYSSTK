import numpy as np
from util import MSE


def create_OLS_beta(X: np.ndarray, z: np.array) -> np.array:
    """Create OLS beta.

    inputs:
        X (n x n matrix): Design matrix
        z (np.array): Solution to optimize for
    returns:
        Optimal solution (np.array)
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ z


def OLS(
    X_train: np.ndarray, X_test: np.ndarray, y_train: np.array, y_test: np.ndarray
) -> tuple[float, float]:
    """Create MSE for dataset using OLS.

    inputs:
        X_train: Training values
        X_test: Testing values
        y_train: Training values
        y_test: Testing values
    returns:
        Mean squared error score for training and predicted.
    """
    beta_OLS = create_OLS_beta(X_train, y_train)
    ztilde = X_train @ beta_OLS

    beta_OLS_test = create_OLS_beta(X_test, y_test)
    z_pred_OLS = X_test @ beta_OLS_test

    train_MSE = MSE(y_train, ztilde)
    test_MSE = MSE(y_test, z_pred_OLS)

    return train_MSE, test_MSE
