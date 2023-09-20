import numpy as np


def create_OLS_beta(X: np.ndarray, z: np.array) -> np.array:
    """Create OLS beta.

    inputs:
        X (n x n matrix): Design matrix
        z (np.array): Solution to optimize for
    returns:
        Optimal solution (np.array)
    """
    return np.linalg.pinv(X.T @ X) @ X.T @ z
