import jax.numpy as np
from jax import lax, jit
from typing import Callable


@jit
def CostCrossEntropy_fast(X: np.ndarray, target: np.ndarray) -> float:
    return -lax.mul(
        1.0 / target.size,
        np.sum(
            lax.mul(
                target,
                lax.log(
                    lax.add(X, 1e-7),
                ),
            )
        ),
    )


@jit
def CostOLS_fast(X: np.ndarray, target: np.ndarray) -> float:
    return lax.mul(
        1.0 / target.size,
        np.sum(
            lax.integer_pow(lax.sub(target, X), 2),
        ),
    )


@jit
def CostCrossEntropy_binary(X, target):
    # NOTE: A delta of 1e-8 is too small, is rounded to zero by jit.
    return -lax.mul(
        (1.0 / target.shape[0]),
        np.sum(
            lax.add(
                lax.mul(
                    target,
                    lax.log(lax.add(X, 1e-7)),
                ),
                lax.mul(
                    lax.sub(1.0 + 1e-7, target),
                    lax.log(
                        lax.add(lax.sub(1.0, X), 1e-7),
                    ),
                ),
            ),
        ),
    )


def CostCrossEntropy(target: np.ndarray) -> Callable:
    """
    Computes the cross-entropy cost function for a given target.

    Args:
        target (np.ndarray): The target values.

    Returns:
        Callable: A function that computes the cross-entropy cost function for a given input.
    """

    def func(X: np.ndarray) -> float:
        """
        Computes the cross-entropy cost function for a given input.

        Args:
            X (np.ndarray): The input values.

        Returns:
            float: The cross-entropy cost function value.
        """
        p0 = target * lax.log(X + 1e-7)
        p1 = (1 - target) * lax.log(1 - X + 1e-7)
        return -(1.0 / target.size) * np.sum(p0 + p1)

    return func


@jit
def fast_OLS(
    X: np.ndarray, y: np.ndarray, theta: np.ndarray, lmbda: float = None
) -> np.ndarray:
    return lax.mul(
        1.0 / X.shape[0],
        np.sum(lax.integer_pow(lax.sub(y, np.dot(X, theta)), 2)),
    )


@jit
def fast_OLS_grad(
    X: np.ndarray, y: np.ndarray, theta: np.ndarray, lmbda: float = None
) -> np.ndarray:
    return lax.mul(
        2.0 / X.shape[0],
        lax.dot(
            X.T,
            lax.sub(np.dot(X, theta), y),
        ),
    )


@jit
def fast_ridge(
    X: np.ndarray, y: np.ndarray, theta: np.ndarray, lmbda: float = None
) -> np.ndarray:
    tmp_theta = theta.squeeze()
    return 1.0 / X.shape[0] * np.sum(
        lax.integer_pow(lax.sub(y, np.dot(X, theta)), 2)
    ) + lmbda * lax.dot(tmp_theta.T, tmp_theta)


@jit
def fast_ridge_grad(
    X: np.ndarray, y: np.ndarray, theta: np.ndarray, lmbda: float = None
) -> np.ndarray:
    # tmp_theta = theta.squeeze()
    return 2.0 * (
        lax.add(
            1.0
            / X.shape[0]
            * lax.dot(
                X.T,
                lax.sub(
                    lax.dot(X, theta),
                    y,
                ),
            ),
            lax.mul(lmbda, theta),
        )
    )


def CostOLS(target: np.ndarray) -> Callable:
    """
    Returns a function that calculates the mean squared error between the target and predicted values.

    Args:
        target: A numpy array of shape (n_samples,) containing the target values.

    Returns:
        A function that takes in a numpy array of shape (n_samples,) containing
        the predicted values and returns the mean squared error.
    """

    def func(X: np.ndarray) -> float:
        """
        Calculates the mean squared error between the target and predicted values.

        Args:
            X: A numpy array of shape (n_samples,) containing the predicted values.

        Returns:
            The mean squared error between the target and predicted values.
        """
        return (1.0 / target.shape[0]) * np.sum((target - X) ** 2)

    return func


def CostLogReg(target: np.ndarray) -> Callable:
    """
    Returns a function that calculates the cost function for logistic regression.

    Args:
        target: A numpy array of shape (n_samples,) containing the true labels.

    Returns:
        A function that takes in a numpy array of shape (n_samples,) containing the predicted labels
        and returns the cost function value.
    """

    def func(X: np.ndarray) -> float:
        """
        Calculates the cost function for logistic regression.

        Args:
            X: A numpy array of shape (n_samples,) containing the predicted labels.

        Returns:
            The cost function value.
        """
        return -(1.0 / target.shape[0]) * np.sum(
            (target * np.log(X + 10e-10)) + ((1 - target) * np.log(1 - X + 10e-10))
        )

    return func
