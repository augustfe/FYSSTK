import autograd.numpy as np


def CostCrossEntropy(target: np.ndarray) -> callable:
    """
    Computes the cross-entropy cost function for a given target.

    Args:
        target (np.ndarray): The target values.

    Returns:
        callable: A function that computes the cross-entropy cost function for a given input.
    """

    def func(X: np.ndarray) -> float:
        """
        Computes the cross-entropy cost function for a given input.

        Args:
            X (np.ndarray): The input values.

        Returns:
            float: The cross-entropy cost function value.
        """
        p0 = target * np.log(X + 1e-10)
        p1 = (1 - target) * np.log(1 - X + 1e-10)
        return -(1.0 / target.size) * np.sum(p0 + p1)

    return func


def CostOLS(target: np.ndarray) -> callable:
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


def CostLogReg(target: np.ndarray) -> callable:
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
