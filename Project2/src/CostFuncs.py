import jax.numpy as jnp
from jax import lax, jit
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


@jit
def fast_OLS(
    X: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray, lmbda: float = None
) -> jnp.ndarray:
    return lax.mul(
        1.0 / X.shape[0],
        jnp.sum(lax.integer_pow(lax.sub(y, jnp.dot(X, theta)), 2)),
    )


@jit
def fast_OLS_grad(
    X: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray, lmbda: float = None
) -> jnp.ndarray:
    return lax.mul(
        2.0 / X.shape[0],
        lax.dot(
            X.T,
            lax.sub(jnp.dot(X, theta), y),
        ),
    )


@jit
def fast_ridge(
    X: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray, lmbda: float = None
) -> jnp.ndarray:
    tmp_theta = theta.squeeze()
    return 1.0 / X.shape[0] * jnp.sum(
        lax.integer_pow(lax.sub(y, jnp.dot(X, theta)), 2)
    ) + lmbda * lax.dot(tmp_theta.T, tmp_theta)


@jit
def fast_ridge_grad(
    X: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray, lmbda: float = None
) -> jnp.ndarray:
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
