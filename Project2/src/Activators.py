import autograd.numpy as np
from autograd import elementwise_grad


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
        return 1.0 / (1 + np.exp(-X))
    except FloatingPointError:
        return np.where(X > np.zeros(X.shape), np.ones(X.shape), np.zeros(X.shape))


def softmax(X: np.ndarray) -> np.ndarray:
    """
    Computes the softmax activation function for a given input array.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array after applying the softmax function.
    """
    X = X - np.max(X, axis=-1, keepdims=True)
    delta = 10e-10
    return np.exp(X) / (np.sum(np.exp(X), axis=-1, keepdims=True) + delta)


def RELU(X: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit (ReLU) activation function.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array with values equal to X where X > 0, and 0 elsewhere.
    """
    return np.where(X > np.zeros(X.shape), X, np.zeros(X.shape))


def LRELU(X: np.ndarray) -> np.ndarray:
    """
    Leaky Rectified Linear Unit activation function.

    Args:
        X (np.ndarray): Input array.

    Returns:
        np.ndarray: Output array with the same shape as X.
    """
    delta = 10e-4
    return np.where(X > np.zeros(X.shape), X, delta * X)


def derivate(func: callable) -> callable:
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
            return np.where(X > 0, 1, 0)

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
            return np.where(X > 0, 1, delta)

        return func

    else:
        return elementwise_grad(func)
