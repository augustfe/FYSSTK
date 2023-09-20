import numpy as np


def create_ridge_beta(X: np.ndarray, z: np.array, lmbd: float = 0) -> np.array:
    """Create ridge beta.

    inputs:
        X (n x n matrix): Design matrix
        z (np.array): Solution to optimize for
        lmbd (float): lambda value for ridge
    returns:
        Optimal solution (np.array)
    """
    I = np.identity(X.shape[1])
    return np.linalg.pinv(X.T @ X + lmbd * I) @ X.T @ z
