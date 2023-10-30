import numpy as np
from numpy.typing import NDArray


class Scheduler:
    """
    Base class for Schedulers
    """

    def __init__(self, eta: float) -> None:
        raise NotImplementedError

    def update_change(self, gradient: NDArray[np.float64]):
        raise NotImplementedError

    def reset(self) -> None:
        pass


class Constant(Scheduler):
    def __init__(self, eta: float) -> None:
        self.eta = eta

    def update_change(self, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        return self.eta * gradient


class Momentum(Scheduler):
    def __init__(self, eta: float, momentum: float) -> None:
        self.eta = eta
        self.momentum = momentum
        self.change = 0.0

    def update_change(self, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        self.change = self.momentum * self.change + self.eta * gradient
        return self.change


class Adagrad(Scheduler):
    def __init__(self, eta: float) -> None:
        self.eta = eta
        self.G_t = None

    def update_change(self, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        delta = 1e-8  # avoid division ny zero

        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t += gradient @ gradient.T

        G_t_inverse = 1 / (
            delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (self.G_t.shape[0], 1)))
        )
        return self.eta * gradient * G_t_inverse

    def reset(self) -> None:
        self.G_t = None


class AdagradMomentum(Scheduler):
    def __init__(self, eta: float, momentum: float) -> None:
        self.eta = eta
        self.G_t = None
        self.momentum = momentum
        self.change = 0.0

    def update_change(self, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        delta = 1e-8  # avoid division ny zero

        if self.G_t is None:
            self.G_t = np.zeros((gradient.shape[0], gradient.shape[0]))

        self.G_t += gradient @ gradient.T

        G_t_inverse = 1 / (
            delta + np.sqrt(np.reshape(np.diagonal(self.G_t), (self.G_t.shape[0], 1)))
        )
        self.change = self.change * self.momentum + self.eta * gradient * G_t_inverse
        return self.change

    def reset(self) -> None:
        self.G_t = None


class RMS_prop(Scheduler):
    def __init__(self, eta: float, rho: float) -> None:
        self.eta = eta
        self.rho = rho
        self.second = 0.0

    def update_change(self, gradient: NDArray[np.float64]):
        delta = 1e-8  # avoid division ny zero
        self.second = self.rho * self.second + (1 - self.rho) * gradient * gradient
        return self.eta * gradient / (np.sqrt(self.second + delta))

    def reset(self) -> None:
        self.second = 0.0


class Adam(Scheduler):
    def __init__(self, eta: float, rho: float, rho2: float) -> None:
        self.eta = eta
        self.rho = rho
        self.rho2 = rho2
        self.moment: float = 0.0
        self.second: float = 0.0
        self.n_epochs: int = 1

    def update_change(self, gradient: NDArray[np.float64]) -> NDArray[np.float64]:
        delta = 1e-8  # avoid division ny zero

        self.moment = self.rho * self.moment + (1 - self.rho) * gradient
        self.second = self.rho2 * self.second + (1 - self.rho2) * gradient * gradient

        moment_corrected = self.moment / (1 - self.rho**self.n_epochs)
        second_corrected = self.second / (1 - self.rho2**self.n_epochs)

        return self.eta * moment_corrected / (np.sqrt(second_corrected + delta))

    def reset(self) -> None:
        self.n_epochs += 1
        self.moment = 0.0
        self.second = 0.0
