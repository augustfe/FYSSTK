import autograd.numpy as np
from autograd import grad
from typing import Callable
from Schedules import Scheduler, Constant


np.random.seed(2018)


class Gradients:
    def __init__(
        self,
        n: int,
        x: np.ndarray,
        y: np.array,
        model: str = "OLS",
        method: str = "analytic",
        scheduler: Scheduler = Constant,
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
        self.scheduler = scheduler

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

    def GradientDescent(self, theta: np.array, n_iter: int) -> np.array:
        for i in range(n_iter):
            gradients = self.gradient(self.y, self.X, theta)
            change = self.scheduler.update_change(gradients)
            theta = theta - change

        return theta

    def StochasticGradientDescent(
        self, theta: np.array, n_epochs: int, M: int, t0: float = None, t1: float = None
    ) -> np.array:
        m = self.n // M

        for epoch in range(1, n_epochs + 1):
            self.scheduler.reset()
            for i in range(m):
                idxs = np.random.choice(self.n, M)
                xi = self.X[idxs]
                yi = self.y[idxs]

                gradients = self.gradient(yi, xi, theta)
                change = self.scheduler.update_change(gradients)

                theta = theta - change

        return theta

    def predict(self, x: np.array, theta: np.array, dim: int = 2) -> np.array:
        X = self.design(x, dim, len(x))
        return X @ theta
