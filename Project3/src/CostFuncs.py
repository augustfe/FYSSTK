from functools import partial
from jax import grad, jit, vmap, lax
import jax.numpy as jnp
from typing import Callable
import NeuralNet from NeuralNetwork


@jit
def CostMSE_fast(X: jnp.ndarray, target: jnp.ndarray) -> float:
    """
    Calculates the mean squared error (MSE) for ordinary least squares (OLS) regression.

    Calculates:
        1/n * sum((y - X)^2)

    Args:
        X (jnp.ndarray): The prediction array of shape (n_samples,).
        target (jnp.ndarray): The target array of shape (n_samples,).

    Returns:
        float: The mean squared error (MSE) value.
    """
    return lax.mul(
        1.0 / target.size,
        jnp.sum(
            lax.integer_pow(lax.sub(target, X), 2),
        ),
    )


@jit
def CostCrossEntropy_binary(X, target):
    """
    Calculates the cross-entropy cost function for binary classification.

    Calculates:
        -1/n * sum(target * log(X) + (1 - target) * log(1 - X))

    Args:
        X (ndarray): The predicted values of shape (n_samples,).
        target (ndarray): The target values of shape (n_samples,).

    Returns:
        float: The cross-entropy cost.
    """
    # NOTE: A delta of 1e-8 is too small, is rounded to zero by jit.
    return -lax.mul(
        (1.0 / target.shape[0]),
        jnp.sum(
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
        p0 = target * lax.log(X + 1e-7)
        p1 = (1 - target) * lax.log(1 - X + 1e-7)
        return -(1.0 / target.size) * jnp.sum(p0 + p1)

    return func


@jit
def fast_OLS(
    X: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray, lmbda: float = None
) -> jnp.ndarray:
    """
    Calculates the cost function for Ordinary Least Squares (OLS) regression.

    Calculates:
        1 / n * sum((y - X @ theta)^2)

    Args:
        X (jnp.ndarray): The input feature matrix of shape (n_samples, n_features).
        y (jnp.ndarray): The target values of shape (n_samples,).
        theta (jnp.ndarray): The weight vector of shape (n_features,).
        lmbda (float, optional): Not used in this function.

    Returns:
        jnp.ndarray: The cost function value.
    """
    return lax.mul(
        1.0 / X.shape[0],
        jnp.sum(lax.integer_pow(lax.sub(y, jnp.dot(X, theta)), 2)),
    )


@jit
def fast_OLS_grad(
    X: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray, lmbda: float = None
) -> jnp.ndarray:
    """
    Compute the analytic gradient of the Ordinary Least Squares (OLS) cost function.

    Calculates:
        2 / n * X.T @ (X @ theta - y)

    Args:
        X (jnp.ndarray): The input feature matrix of shape (n_samples, n_features).
        y (jnp.ndarray): The target values of shape (n_samples,).
        theta (jnp.ndarray): The weight vector of shape (n_features,).
        lmbda (float, optional): Not used in this function.

    Returns:
        jnp.ndarray: The gradient of the OLS cost function with respect to theta, of shape (n_features,).
    """
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
    """
    Compute the cost function for ridge regression.

    Calculates:
        1 / n * sum((y - X @ theta)^2) + lmbda * theta.T @ theta

    Args:
        X (jnp.ndarray): The input feature matrix of shape (n_samples, n_features).
        y (jnp.ndarray): The target values of shape (n_samples,).
        theta (jnp.ndarray): The weight vector of shape (n_features,).
        lmbda (float, optional): The regularization parameter. Defaults to None.

    Returns:
        jnp.ndarray: The cost function value.
    """
    tmp_theta = theta.squeeze()
    return 1.0 / X.shape[0] * jnp.sum(
        lax.integer_pow(lax.sub(y, jnp.dot(X, theta)), 2)
    ) + lmbda * lax.dot(tmp_theta.T, tmp_theta)


@jit
def fast_ridge_grad(
    X: jnp.ndarray, y: jnp.ndarray, theta: jnp.ndarray, lmbda: float = None
) -> jnp.ndarray:
    """
    Compute the analytic gradient of the ridge regression cost function.

    Calculates:
        2 / n * X.T @ (X @ theta - y) + 2 * lmbda * theta

    Args:
        X (jnp.ndarray): The input feature matrix of shape (n_samples, n_features).
        y (jnp.ndarray): The target values of shape (n_samples,).
        theta (jnp.ndarray): The parameter vector of shape (n_features,).
        lmbda (float, optional): The regularization parameter. Defaults to None.

    Returns:
        jnp.ndarray: The gradient of the ridge regression cost function, of shape (n_features,).
    """
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


def CostMSE(target: jnp.ndarray) -> Callable:
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
            (target * jnp.log(X + 10e-10)) +
            ((1 - target) * jnp.log(1 - X + 10e-10))
        )

    return func


def Cost_1d_heat(
    NNu: NeuralNet,
    u0: jnp.ndarray,
    tx: jnp.ndarray,
    u_b: float,
    mu1: float,
    mu2: float,
    alpha: float = 1,
) -> callable:
    # Define differentiable versions of the NN
    dNNu_dx = grad(NNu, 1)
    d2NNu_dx2 = grad(dNNu_dx, 1)

    # JAX operations must be batched to process a batch of inputs (t, x)
    d2_dx2 = jit(vmap(d2NNu_dx2, in_axes=(None, 0)))
    d_dt = jit(vmap(grad(NNu, 0), in_axes=(None, 0)))

    def mse_loss(y_true, y_pred):
        return jnp.mean((y_true - y_pred) ** 2)

    @partial(jit, static_argnums=(1,))
    def inner_loss(nnu, t, x):
        "calculate "
        u_pred = nnu(jnp.stack([t, x], axis=1))
        return mse_loss(
            d_dt(NNu, t, x) - alpha * d2_dx2(NNu, t, x), jnp.zeros_like(u_pred)
        )

    @partial(jit, static_argnums=(1,))
    def initial_loss(nnu, x):
        u_pred = nnu(jnp.stack([jnp.zeros_like(x), x], axis=1))
        return mse_loss(u0, u_pred)

    @partial(jit, static_argnums=(1,))
    def boundary_loss(nnu, t, x, ub):
        u_pred = nnu(jnp.stack([t, x], axis=1))
        return mse_loss(u_pred, jnp.full_like(u_pred, ub))

    @jit
    def func(NNu, batch: jnp.ndarray):
        t_inner, x_inner, t_boundary, x_boundary = batch
        pde_loss_val = inner_loss(NNu, t_inner, x_inner)
        init_loss_val = initial_loss(NNu, tx)
        boundary_loss_val = boundary_loss(NNu, t_boundary, x_boundary, u_b)

        total_loss = (
            mu1 * pde_loss_val + mu2 * boundary_loss_val +
            (1 - mu2) * init_loss_val
        )
        return total_loss

    # Partially applying the NNu to the func to make it a static argument
    func_with_NNu = partial(func, NNu)

    return func_with_NNu
