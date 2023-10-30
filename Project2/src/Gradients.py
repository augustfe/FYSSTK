import autograd.numpy as np
from autograd import grad
from typing import Callable


np.random.seed(2018)


class Gradients:
    def __init__(
        self,
        n: int,
        x: np.ndarray,
        y: np.array,
        model: str = "OLS",
        method: str = "analytic",
        dim: int = 2,
        lmbda: float = None,
    ) -> None:
        if not isinstance(n, int):
            raise TypeError(f"n should be and integer, not {n} of type {type(n)}")
        if not isinstance(x, np.ndarray) or not isinstance(y, np.ndarray):
            raise TypeError(f"x and y must be arrays, not {type(x)} and {type(y)}")
        if not (n == len(x) == len(y)):
            raise ValueError("Number of points must correspond to length of arrays")

        self.n = n
        self.x = x
        self.y = y
        self.X = self.design(x, dim=dim)
        self.model = model
        self.method = method
        self.lmbda = lmbda
        self.dim = dim
        self.BaseCost: Callable[[np.array, np.array, np.array], float] = getattr(
            self, f"cost{model}"
        )
        self.gradient: Callable[[np.array, np.array, np.array], np.array] = (
            getattr(self, f"analytic{model}")
            if (method == "analytic")
            else grad(self.BaseCost, 2)
        )

    def pickfunc(self) -> Callable[[np.array, np.array, np.array], float]:
        if self.model == "OLS":
            return self.costOLS
        else:
            return self.costRidge

    def costOLS(self, y: np.array, X: np.array, theta: np.array) -> float:
        return 1 / self.n * np.sum((y - X @ theta) ** 2)

    def analyticOLS(self, y: np.array, X: np.array, theta: np.array) -> np.array:
        return 2.0 / self.n * X.T @ ((X @ theta) - y)

    def costRidge(self, y: np.array, X: np.array, theta: np.array) -> float:
        return self.costOLS(y, X, theta) + self.lmbda * np.dot(theta, theta)

    def analyticRidge(self, y: np.array, X: np.array, theta: np.array) -> np.array:
        return 2.0 * (1 / self.n * X.T @ ((X @ theta) - y) + self.lmbda * theta)

    def design(self, x: np.array, dim: int, n: int = None) -> np.array:
        if n is None:
            n = self.n
        X = np.ones((n, dim + 1))
        for i in range(1, dim + 1):
            X[:, i] = (x**i).flatten()

        return X

    def GradientDescent(
        self, theta: np.array, n_iter: int, step_size: float = 0.1
    ) -> np.array:
        for i in range(n_iter):
            gradients = self.gradient(self.y, self.X, theta)
            if np.linalg.norm(gradients, 2) < 1e-8:
                break
            theta = theta - step_size * gradients

        return theta

    def GradientDescentMomentum(
        self, theta: np.array, n_iter: int, step_size: float, momentum: float
    ):
        change = 0.0

        for i in range(n_iter):
            gradients = self.gradient(self.y, self.X, theta)
            new_change = step_size * gradients + momentum * change

            theta = theta - new_change
            change = new_change

        return theta

    def step_length(self, t: float, t0: float, t1: float) -> float:
        return t0 / (t + t1)

    def StochasticGradientDescent(
        self, theta: np.array, n_epochs: int, M: int, t0: float = None, t1: float = None
    ) -> np.array:
        m = self.n // M

        for epoch in range(1, n_epochs + 1):
            for i in range(m):
                idxs = np.random.choice(self.n, M)
                xi = self.X[idxs]
                yi = self.y[idxs]

                gradients = self.gradient(yi, xi, theta)
                t = epoch * m + i

                theta = theta - gradients * self.step_length(t, t0, t1)

        return theta

    def AdaGrad(
        self, theta: np.array, n_epochs: int, M: int, eta: float = 0.01
    ) -> np.array:
        m = self.n // M
        delta = 1e-8

        for epoch in range(n_epochs):
            Giter = 0.0
            for i in range(m):
                idxs = np.random.choice(self.n, M)
                xi = self.X[idxs]
                yi = self.y[idxs]

                gradients = (1.0 / M) * self.gradient(yi, xi, theta)
                Giter += gradients * gradients
                update = gradients * eta / (delta + np.sqrt(Giter))
                theta = theta - update

        return theta

    def RMSprop(
        self,
        theta: np.array,
        n_epochs: int,
        M: int,
        eta: float = 0.01,
        rho: float = 0.99,
    ) -> np.array:
        m = self.n // M
        delta = 1e-8

        for epoch in range(n_epochs):
            Giter = 0.0
            for i in range(m):
                idxs = np.random.choice(self.n, M)
                xi = self.X[idxs]
                yi = self.y[idxs]

                gradients = (1.0 / M) * self.gradient(yi, xi, theta)
                Giter = rho * Giter + (1 - rho) * gradients * gradients
                update = gradients * eta / (delta + np.sqrt(Giter))
                theta = theta - update

        return theta

    def ADAM(
        self,
        theta: np.array,
        n_epochs: int,
        M: int,
        eta: float = 0.01,
        beta1: float = 0.9,
        beta2: float = 0.999,
    ) -> np.array:
        m = self.n // M
        delta = 1e-7
        iter = 0
        for epoch in range(n_epochs):
            first_moment = 0.0
            second_moment = 0.0
            iter += 1
            for i in range(m):
                idxs = np.random.choice(self.n, M)
                xi = self.X[idxs]
                yi = self.y[idxs]

                gradients = (1.0 / M) * self.gradient(yi, xi, theta)
                first_moment = beta1 * first_moment + (1 - beta1) * gradients
                second_moment = (
                    beta2 * second_moment + (1 - beta2) * gradients * gradients
                )

                first_term = first_moment / (1.0 - beta1**iter)
                second_term = second_moment / (1.0 - beta2**iter)

                update = eta * first_term / (np.sqrt(second_term) + delta)
                theta = theta - update

        return theta

    def predict(self, x: np.array, theta: np.array, dim: int = 2) -> np.array:
        X = self.design(x, dim, len(x))
        return X @ theta
