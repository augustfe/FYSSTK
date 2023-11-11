import jax.numpy as np
from jax import lax, jit
from functools import partial


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


@jit
def fast_const(eta, gradient):
    return lax.mul(eta, gradient)


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
        return fast_const(self.eta, gradient)


@jit
def fast_mom(momentum, change, eta, gradient):
    return lax.add(
        lax.mul(momentum, change),
        lax.mul(eta, gradient),
    )


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
        self.change = fast_mom(self.momentum, self.change, self.eta, gradient)
        return self.change

    def reset(self) -> None:
        """
        Resets the change in the weights to 0.
        """
        self.change = 0.0


@jit
def fast_gt_add(gradient, G_t):
    G_t = lax.add(
        G_t,
        lax.dot(gradient, gradient.T),
    )
    return G_t


@partial(jit, static_argnums=(1,))
def fast_gt_inv(G_t):
    delta = 1e-8  # avoid division by zero
    G_t_inverse = lax.reciprocal(
        delta
        + lax.sqrt(
            np.reshape(
                np.diagonal(G_t),
                (G_t.shape[0], 1),
            ),
        )
    )
    # G_t_inverse = 1 / (delta + np.sqrt(np.reshape(np.diagonal(G_t), (G_t.shape[0], 1))))
    return G_t_inverse


@jit
def fast_adagrad(eta, gradient, G_t_inverse):
    change = lax.mul(
        lax.mul(
            eta,
            gradient,
        ),
        G_t_inverse,
    )
    return change


@jit
def fast_adamoment(eta, gradient, G_t_inverse, momentum, change):
    change = lax.add(
        lax.mul(
            momentum,
            change,
        ),
        lax.mul(
            lax.mul(eta, gradient),
            G_t_inverse,
        ),
    )
    return change


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
        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t = fast_gt_add(gradient, self.G_t)
        G_t_inverse = fast_gt_inv(self.G_t)

        change = fast_adagrad(self.eta, gradient, G_t_inverse)
        return change

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
        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t = fast_gt_add(gradient, self.G_t)

        G_t_inverse = fast_gt_inv(self.G_t)

        self.change = fast_adamoment(
            self.eta, gradient, G_t_inverse, self.momentum, self.change
        )
        return self.change

    def reset(self) -> None:
        """
        Resets the sum of the squares of the gradients to None.
        """
        self.G_t = None


@jit
def fast_rms_second(rho, second, gradient):
    second = lax.add(
        lax.mul(rho, second),
        lax.mul(
            lax.sub(1.0, rho),
            lax.square(gradient),
        ),
    )
    return second


@jit
def fast_rmsprop(eta, gradient, second):
    delta = 1e-8
    change = lax.div(
        lax.mul(eta, gradient),
        lax.sqrt(
            lax.add(second, delta),
        ),
    )
    return change


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
        self.second = fast_rms_second(self.rho, self.second, gradient)
        # self.second = self.rho * self.second + (1 - self.rho) * gradient * gradient
        # return self.eta * gradient / (np.sqrt(self.second + delta))
        change = fast_rmsprop(self.eta, gradient, self.second)
        return change

    def reset(self) -> None:
        """
        Reset the running average of the square of the gradients.
        """
        self.second = 0.0


@jit
def fast_adam_moment(rho, moment, gradient):
    moment = lax.add(
        lax.mul(rho, moment),
        lax.mul(
            lax.sub(1.0, rho),
            gradient,
        ),
    )
    return moment


@jit
def fast_adam_second(rho2, second, gradient):
    second = lax.add(
        lax.mul(rho2, second),
        lax.mul(
            lax.sub(1.0, rho2),
            lax.square(gradient),
        ),
    )
    return second


@partial(jit, static_argnums=(5,))
def fast_adam(moment, rho, second, rho2, eta, n_epochs):
    delta = 1e-8  # avoid division by zero
    moment_corrected = lax.div(
        moment,
        lax.sub(
            1.0,
            lax.integer_pow(rho, n_epochs),
        ),
    )
    second_corrected = lax.div(
        second,
        lax.sub(
            1.0,
            lax.integer_pow(rho2, n_epochs),
        ),
    )
    change = lax.div(
        lax.mul(eta, moment_corrected),
        lax.sqrt(
            lax.add(second_corrected, delta),
        ),
    )
    return change


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
        self.moment = fast_adam_moment(self.rho, self.moment, gradient)
        # self.moment = self.rho * self.moment + (1 - self.rho) * gradient
        self.second = fast_adam_second(self.rho2, self.second, gradient)
        # self.second = self.rho2 * self.second + (1 - self.rho2) * gradient * gradient

        # moment_corrected = self.moment / (1 - self.rho**self.n_epochs)

        # second_corrected = self.second / (1 - self.rho2**self.n_epochs)

        change = fast_adam(
            self.moment,
            self.rho,
            self.second,
            self.rho2,
            self.eta,
            self.n_epochs,
        )
        return change
        # return self.eta * moment_corrected / (np.sqrt(second_corrected + delta))

    def reset(self) -> None:
        """
        Reset the optimizer.
        """
        self.n_epochs += 1
        self.moment = 0.0
        self.second = 0.0


@jit
def fast_timedecay(epochs, minibatch_size, minibatch_num, t0, t1, gradient):
    t = lax.add(lax.mul(epochs, minibatch_size), minibatch_num)
    eta = lax.div(t0, lax.add(t, t1))
    change = lax.mul(eta, gradient)
    return change


class TimeDecay(Scheduler):
    def __init__(self, t0: float, t1: float, minibatch_size: int):
        self.t0 = t0
        self.t1 = t1
        self.epochs = 0
        self.minibatch_size = minibatch_size
        self.minibatch_num = 0

    def update_change(self, gradient: np.ndarray) -> np.ndarray:
        # t = self.epochs * self.minibatch_size + self.minibatch_num
        # eta = self.t0 / (t + self.t1)

        # change = eta * gradient
        change = fast_timedecay(
            self.epochs,
            self.minibatch_size,
            self.minibatch_num,
            self.t0,
            self.t1,
            gradient,
        )
        self.minibatch_num += 1
        return change

    def reset(self) -> None:
        self.epochs += 1
        self.minibatch_num = 0
