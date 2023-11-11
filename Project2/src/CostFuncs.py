import jax.numpy as jnp
from jax import lax
from typing import Callable


def CostCrossEntropy(target: jnp.ndarray) -> Callable:
    """
    Computes the cross-entropy cost function for a given target.

    Args:
        target (jnp.ndarray): The target values.

    Returns:
        Callable: A function that computes the cross-entropy cost function for a given input.
    """

    def func(X: jnp.ndarray) -> float:
        """
        Computes the cross-entropy cost function for a given input.

        Args:
            X (jnp.ndarray): The input values.

        Returns:
            float: The cross-entropy cost function value.
        """
        p0 = target * lax.log(X + 1e-10)
        p1 = (1 - target) * lax.log(1 - X + 1e-10)
        return -(1.0 / target.size) * jnp.sum(p0 + p1)

    return func


def regressionOLS(X: jnp.ndarray, y: jnp.ndarray, lmbda: float = None) -> Callable:
    def func(theta: jnp.ndarray) -> float:
        return (1.0 / X.shape[0]) * jnp.sum(
            lax.integer_pow(lax.sub(y, jnp.dot(X, theta)), 2)
        )

    return func


def derivativeOLS(X: jnp.ndarray, y: jnp.ndarray, lmbda: float = None) -> Callable:
    def func(theta: jnp.ndarray) -> jnp.ndarray:
        return (2.0 / X.shape[0]) * jnp.matmul(X.T, (lax.sub(jnp.matmul(X, theta), y)))

    return func


def regressionRidge(X: jnp.ndarray, y: jnp.ndarray, lmbda: float) -> Callable:
    def func(theta: jnp.ndarray) -> float:
        return (1.0 / X.shape[0]) * jnp.sum(
            lax.integer_pow(lax.sub(y, jnp.dot(X, theta)), 2)
        ) + lmbda * jnp.dot(theta, theta)

    return func


def derivativeRidge(X: jnp.ndarray, y: jnp.ndarray, lmbda: float) -> Callable:
    def func(theta: jnp.ndarray) -> jnp.ndarray:
        return 2.0 * (1 / X.shape[0] * X.T @ ((X @ theta) - y) + lmbda * theta)

    return func


def CostOLS(target: jnp.ndarray) -> Callable:
    """
    Returns a function that calculates the mean squared error between the target and predicted values.

    Args:
        target: A numpy array of shape (n_samples,) containing the target values.

    Returns:
        A function that takes in a numpy array of shape (n_samples,) containing
        the predicted values and returns the mean squared error.
    """

    def func(X: jnp.ndarray) -> float:
        """
        Calculates the mean squared error between the target and predicted values.

        Args:
            X: A numpy array of shape (n_samples,) containing the predicted values.

        Returns:
            The mean squared error between the target and predicted values.
        """
        return (1.0 / target.shape[0]) * jnp.sum((target - X) ** 2)

    return func


def CostLogReg(target: jnp.ndarray) -> Callable:
    """
    Returns a function that calculates the cost function for logistic regression.

    Args:
        target: A numpy array of shape (n_samples,) containing the true labels.

    Returns:
        A function that takes in a numpy array of shape (n_samples,) containing the predicted labels
        and returns the cost function value.
    """

    def func(X: jnp.ndarray) -> float:
        """
        Calculates the cost function for logistic regression.

        Args:
            X: A numpy array of shape (n_samples,) containing the predicted labels.

        Returns:
            The cost function value.
        """
        return -(1.0 / target.shape[0]) * jnp.sum(
            (target * jnp.log(X + 10e-10)) + ((1 - target) * jnp.log(1 - X + 10e-10))
        )

    return func
