import autograd.numpy as np


class Scheduler:
    """
    Base class for Schedulers
    """

    def __init__(self, eta: float) -> None:
        """
        Initializes the scheduler with a given learning rate.

        Args:
            eta (float): The learning rate for the scheduler.
        """
        raise NotImplementedError

    def update_change(self, gradient: np.ndarray) -> None:
        """
        Updates the scheduler based on the gradient.

        Args:
            gradient (np.ndarray): The gradient used to update the scheduler.
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Resets the scheduler to its initial state.
        """
        pass


class Constant(Scheduler):
    """
    A learning rate scheduler that keeps the learning rate constant throughout training.

    Args:
        eta (float): The learning rate.

    Attributes:
        eta (float): The learning rate.

    Methods:
        update_change: Updates the learning rate by multiplying it with the gradient.

    """

    def __init__(self, eta: float) -> None:
        self.eta = eta

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Updates the learning rate by multiplying it with the gradient.

        Args:
            gradient (np.ndarray): The gradient.

        Returns:
            np.ndarray: The updated learning rate.

        """
        return self.eta * gradient


class Momentum(Scheduler):
    """
    Implements the Momentum optimizer.

    Args:
        eta (float): The learning rate.
        momentum (float): The momentum parameter.

    Attributes:
        eta (float): The learning rate.
        momentum (float): The momentum parameter.
        change (float): The change in the weights.

    Methods:
        update_change: Updates the change in the weights.

    """

    def __init__(self, eta: float, momentum: float) -> None:
        self.eta = eta
        self.momentum = momentum
        self.change = 0.0

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Updates the change in the weights.

        Args:
            gradient (np.ndarray): The gradient of the weights.

        Returns:
            np.ndarray: The updated change in the weights.

        """
        self.change = self.momentum * self.change + self.eta * gradient
        return self.change


class Adagrad(Scheduler):
    """
    Adagrad optimizer.

    Args:
        eta (float): Learning rate.

    Attributes:
        eta (float): Learning rate.
        G_t (ndarray): Matrix of sum of squares of past gradients.

    Methods:
        update_change(gradient: np.ndarray) -> np.ndarray:
            Update the weights of the model based on the gradient.
        reset() -> None:
            Reset the optimizer's state.
    """

    def __init__(self, eta: float) -> None:
        self.eta = eta
        self.G_t = None

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update the weights of the model based on the gradient.

        Args:
            gradient (ndarray): Gradient of the loss function.

        Returns:
            ndarray: Updated weights of the model.
        """
        delta = 1e-8  # avoid division by zero

        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t += gradient @ gradient.T

        G_t_inverse = 1 / (
            delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (self.G_t.shape[0], 1)))
        )
        return self.eta * gradient * G_t_inverse

    def reset(self) -> None:
        """
        Reset the optimizer's state.
        """
        self.G_t = None


class AdagradMomentum(Scheduler):
    """
    AdagradMomentum is a class that implements the Adagrad Momentum optimizer.

    Args:
        eta (float): The learning rate.
        momentum (float): The momentum parameter.

    Attributes:
        eta (float): The learning rate.
        G_t (np.ndarray): The sum of the squares of the gradients.
        momentum (float): The momentum parameter.
        change (np.ndarray): The change in the weights.

    Methods:
        update_change(gradient: np.ndarray) -> np.ndarray:
            Updates the change in the weights based on the gradient.
        reset() -> None:
            Resets the sum of the squares of the gradients to None.
    """

    def __init__(self, eta: float, momentum: float) -> None:
        self.eta = eta
        self.G_t = None
        self.momentum = momentum
        self.change = 0.0

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Updates the change in the weights based on the gradient.

        Args:
            gradient (np.ndarray): The gradient.

        Returns:
            np.ndarray: The change in the weights.
        """
        delta = 1e-8  # avoid division by zero

        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t += gradient @ gradient.T

        G_t_inverse = 1 / (
            delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (self.G_t.shape[0], 1)))
        )
        self.change = self.change * self.momentum + self.eta * gradient * G_t_inverse
        return self.change

    def reset(self) -> None:
        """
        Resets the sum of the squares of the gradients to None.
        """
        self.G_t = None


class RMS_prop(Scheduler):
    """
    Root Mean Square Propagation (RMS_prop) optimizer.

    Args:
        eta (float): Learning rate.
        rho (float): Decay rate.

    Attributes:
        eta (float): Learning rate.
        rho (float): Decay rate.
        second (float): Running average of the square of the gradients.

    Methods:
        update_change(gradient: np.ndarray) -> np.ndarray:
            Update the parameters based on the gradient.
        reset() -> None:
            Reset the running average of the square of the gradients.
    """

    def __init__(self, eta: float, rho: float) -> None:
        self.eta = eta
        self.rho = rho
        self.second = 0.0

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update the parameters based on the gradient.

        Args:
            gradient (np.ndarray): Gradient of the loss function.

        Returns:
            np.ndarray: Updated parameters.
        """
        delta = 1e-8  # avoid division by zero
        self.second = self.rho * self.second + (1 - self.rho) * gradient * gradient
        return self.eta * gradient / (np.sqrt(self.second + delta))

    def reset(self) -> None:
        """
        Reset the running average of the square of the gradients.
        """
        self.second = 0.0


class Adam(Scheduler):
    """
    Adam optimizer.

    Args:
        eta (float): Learning rate.
        rho (float): Decay rate for the first moment estimate.
        rho2 (float): Decay rate for the second moment estimate.

    Attributes:
        moment (float): First moment estimate.
        second (float): Second moment estimate.
        n_epochs (int): Number of epochs.

    Methods:
        update_change: Update the parameters.
        reset: Reset the optimizer.

    """

    def __init__(self, eta: float, rho: float, rho2: float) -> None:
        self.eta = eta
        self.rho = rho
        self.rho2 = rho2
        self.moment: float = 0.0
        self.second: float = 0.0
        self.n_epochs: int = 1

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        """
        Update the parameters.

        Args:
            gradient (np.ndarray): Gradient of the loss function.

        Returns:
            np.ndarray: Updated parameters.
        """
        delta = 1e-8  # avoid division by zero

        self.moment = self.rho * self.moment + (1 - self.rho) * gradient
        self.second = self.rho2 * self.second + (1 - self.rho2) * gradient * gradient

        moment_corrected = self.moment / (1 - self.rho**self.n_epochs)
        second_corrected = self.second / (1 - self.rho2**self.n_epochs)

        return self.eta * moment_corrected / (np.sqrt(second_corrected + delta))

    def reset(self) -> None:
        """
        Reset the optimizer.
        """
        self.n_epochs += 1
        self.moment = 0.0
        self.second = 0.0


class TimeDecay(Scheduler):
    def __init__(self, t0: float, t1: float, minibatch_size: int):
        self.t0 = t0
        self.t1 = t1
        self.epochs = 0
        self.minibatch_size = minibatch_size
        self.minibatch_num = 0

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        t = self.epochs * self.minibatch_size + self.minibatch_num
        eta = self.t0 / (t + self.t1)

        change = eta * gradient
        self.minibatch_num += 1
        return change

    def reset(self) -> None:
        self.epochs += 1
        self.minibatch_num = 0
