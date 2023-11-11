import numpy as np
from jax import grad, vmap
import jax.numpy as jnp
from typing import Callable


def identity(X: np.ndarray) -> np.ndarray:
    """
    Identity activation function.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: The input array, unchanged.
    """
    return X


def sigmoid(X: np.ndarray) -> np.ndarray:
    """
    Applies the sigmoid activation function to the input array.

    Args:
        X (np.ndarray): The input array.

    Returns:
        np.ndarray: The output array after applying the sigmoid function.
    """
    try:
        return 1.0 / (1 + jnp.exp(-X))
    except FloatingPointError:
        return jnp.where(X > jnp.zeros(X.shape), jnp.ones(X.shape), jnp.zeros(X.shape))


def softmax(X: np.ndarray) -> np.ndarray:
    """
    Computes the softmax activation function for a given input array.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying the softmax function.
    """
    X = X - jnp.max(X, axis=-1, keepdims=True)
    delta = 10e-10
    return jnp.exp(X) / (jnp.sum(jnp.exp(X), axis=-1, keepdims=True) + delta)


def RELU(X: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation function.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array with values equal to X where X > 0, and 0 elsewhere.
    """
    return jnp.where(X > jnp.zeros(X.shape), X, jnp.zeros(X.shape))


def LRELU(X: np.ndarray) -> np.ndarray:
    """
    Leaky Rectified Linear Unit activation function.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array with the same shape as X.
    """
    delta = 10e-4
    return jnp.where(X > jnp.zeros(X.shape), X, delta * X)


def derivate(func: Callable) -> Callable:
    """
    Computes the derivative of the input activation function.

    Args:
        func: The activation function to compute the derivative of.

    Returns:
        The derivative of the input activation function.
    """
    if func.__name__ == "RELU":

        def func(X: np.ndarray) -> np.ndarray:
            """
            Computes the derivative of the ReLU activation function.

            Args:
                X: The input to the ReLU activation function.

            Returns:
                The derivative of the ReLU activation function.
            """
            return jnp.where(X > 0, 1, 0)

        return func

    elif func.__name__ == "LRELU":

        def func(X: np.ndarray) -> np.ndarray:
            """
            Computes the derivative of the Leaky ReLU activation function.

            Args:
                X: The input to the Leaky ReLU activation function.

            Returns:
                The derivative of the Leaky ReLU activation function.
            """
            delta = 10e-4
            return jnp.where(X > 0, 1, delta)

        return func

    else:
        return vmap(grad(func))
