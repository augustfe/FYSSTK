import numpy as np
from util import MSE, R2Score


def create_ridge_beta(X: np.ndarray, z: np.array, lmbd: float = 0) -> np.array:
    """Create ridge beta.

    inputs:
        X (n x n matrix): Design matrix
        z (np.array): Solution to optimize for
        lmbd (float): lambda value for ridge
    returns:
        Optimal solution (np.array)
    """
    Identity = np.identity(X.shape[1])
    return np.linalg.pinv(X.T @ X + lmbd * Identity) @ X.T @ z


def Ridge(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.array,
    y_test: np.ndarray,
    lmbda: float,
) -> tuple[float, float, float]:
    """Create MSE for dataset using Ridge.

    inputs:
        X_train: Training values
        X_test: Testing values
        y_train: Training values
        y_test: Testing values
        lmda: Lambda value
    returns:
        Mean squared error score for training and predicted.
    """
    beta_ridge = create_ridge_beta(X_train, y_train, lmbda)
    z_tilde_ridge = X_train @ beta_ridge

    beta_ridge_test = create_ridge_beta(X_test, y_test, lmbda)
    z_pred_Ridge = X_test @ beta_ridge_test

    MSE_train = MSE(y_train, z_tilde_ridge)
    MSE_test = MSE(y_test, z_pred_Ridge)

    R2 = R2Score(y_test, z_pred_Ridge)

    return MSE_train, MSE_test, R2
