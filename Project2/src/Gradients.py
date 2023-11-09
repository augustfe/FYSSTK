import autograd.numpy as np
from autograd import grad
from Schedules import Scheduler, Constant
from typing import Callable


np.random.seed(2018)


class Gradients:
    """
    A class for computing gradients using various methods.

    Attributes:
        n (int): The number of data points.
        x (np.ndarray): The input data.
        y (np.ndarray): The output data.
        model (str): The type of model to use. Default is "OLS".
        method (str): The method to use for computing gradients. Default is "analytic".
        scheduler (Scheduler): The scheduler to use for updating the learning rate. Default is Constant.
        dim (int): The degree of the polynomial to use for the design matrix. Default is 2.
        lmbda (float): The regularization parameter. Default is None.

    Methods:
        __init__(self, n: int, x: np.ndarray, y: np.ndarray, model: str = "OLS", method: str = "analytic",
                 scheduler: Scheduler = Constant, dim: int = 2, lmbda: float = None) -> None:
            Initializes the Gradients object.
        pickfunc(self) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
            Returns the cost function to use based on the model type.
        costOLS(self, y: np.ndarray, X: np.ndarray, theta: np.ndarray) -> float:
            Computes the cost function for OLS regression.
        analyticOLS(self, y: np.ndarray, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
            Computes the gradient analytically for OLS regression.
        costRidge(self, y: np.ndarray, X: np.ndarray, theta: np.ndarray) -> float:
            Computes the cost function for Ridge regression.
        analyticRidge(self, y: np.ndarray, X: np.ndarray, theta: np.ndarray) -> np.ndarray:
            Computes the gradient analytically for Ridge regression.
        design(self, x: np.ndarray, dim: int, n: int = None) -> np.ndarray:
            Computes the design matrix for the given input data.
        GradientDescent(self, theta: np.ndarray, n_iter: int) -> np.ndarray:
            Performs gradient descent to optimize the model parameters.
        StochasticGradientDescent(self, theta: np.ndarray, n_epochs: int, M: int) -> np.ndarray:
            Performs stochastic gradient descent to optimize the model parameters.
        predict(self, x: np.ndarray, theta: np.ndarray, dim: int = 2) -> np.ndarray:
            Predicts the output values for the given input data using the model parameters.
    """

    def __init__(
        self,
        n: int,
        x: np.ndarray,
        y: np.ndarray,
        model: str = "OLS",
        method: str = "analytic",
        scheduler: Scheduler = Constant,
        dim: int = 2,
        lmbda: float = None,
    ) -> None:
        """
        Initializes the Gradients object.

        Args:
            n (int): The number of data points.
            x (np.ndarray): The input data.
            y (np.ndarray): The output data.
            model (str): The type of model to use. Default is "OLS".
            method (str): The method to use for computing gradients. Default is "analytic".
            scheduler (Scheduler): The scheduler to use for updating the learning rate. Default is Constant.
            dim (int): The degree of the polynomial to use for the design matrix. Default is 2.
            lmbda (float): The regularization parameter. Default is None.

        Raises:
            TypeError: If n is not an integer.
            ValueError: If the length of x and y do not match n.
        """
        if not isinstance(n, int):
            raise TypeError(f"n should be an integer, not {n} of type {type(n)}")
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
        self.BaseCost: Callable[[np.ndarray, np.ndarray, np.ndarray], float] = getattr(
            self, f"cost{model}"
        )
        self.gradient: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray] = (
            getattr(self, f"analytic{model}")
            if (method == "analytic")
            else grad(self.BaseCost, 2)
        )
        self.scheduler = scheduler

    def pickfunc(self) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
        """
        Returns the cost function to use based on the model type.

        Returns:
            Callable[[np.ndarray, np.ndarray, np.ndarray], float]: The cost function to use.
        """
        if self.model == "OLS":
            return self.costOLS
        else:
            return self.costRidge

    def costOLS(self, y: np.ndarray, X: np.ndarray, theta: np.ndarray) -> float:
        """
        Computes the cost function for OLS regression.

        Args:
            y (np.ndarray): The output data.
            X (np.ndarray): The design matrix.
            theta (np.ndarray): The model parameters.

        Returns:
            float: The value of the cost function.
        """
        return 1 / self.n * np.sum((y - X @ theta) ** 2)

    def analyticOLS(
        self, y: np.ndarray, X: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """
        Computes the gradient analytically for OLS regression.

        Args:
            y (np.ndarray): The output data.
            X (np.ndarray): The design matrix.
            theta (np.ndarray): The model parameters.

        Returns:
            np.ndarray: The gradient of the cost function.
        """
        return 2.0 / self.n * X.T @ ((X @ theta) - y)

    def costRidge(self, y: np.ndarray, X: np.ndarray, theta: np.ndarray) -> float:
        """
        Computes the cost function for Ridge regression.

        Args:
            y (np.ndarray): The output data.
            X (np.ndarray): The design matrix.
            theta (np.ndarray): The model parameters.

        Returns:
            float: The value of the cost function.
        """
        return self.costOLS(y, X, theta) + self.lmbda * np.dot(theta, theta)

    def analyticRidge(
        self, y: np.ndarray, X: np.ndarray, theta: np.ndarray
    ) -> np.ndarray:
        """
        Computes the gradient analytically for Ridge regression.

        Args:
            y (np.ndarray): The output data.
            X (np.ndarray): The design matrix.
            theta (np.ndarray): The model parameters.

        Returns:
            np.ndarray: The gradient of the cost function.
        """
        return 2.0 * (1 / self.n * X.T @ ((X @ theta) - y) + self.lmbda * theta)

    def design(self, x: np.ndarray, dim: int, n: int = None) -> np.ndarray:
        """
        Computes the design matrix for the given input data.

        Args:
            x (np.ndarray): The input data.
            dim (int): The degree of the polynomial to use for the design matrix.
            n (int): The number of data points. Default is None.

        Returns:
            np.ndarray: The design matrix.
        """
        if n is None:
            n = self.n
        X = np.ones((n, dim + 1))
        for i in range(1, dim + 1):
            X[:, i] = (x**i).flatten()

        return X

    def GradientDescent(self, theta: np.ndarray, n_iter: int) -> np.ndarray:
        """
        Performs gradient descent to optimize the model parameters.

        Args:
            theta (np.ndarray): The initial model parameters.
            n_iter (int): The number of iterations to perform.

        Returns:
            np.ndarray: The optimized model parameters.
        """
        self.errors = np.zeros(n_iter)
        self.errors.fill(np.nan)

        for i in range(n_iter):
            gradients = self.gradient(self.y, self.X, theta)
            change = self.scheduler.update_change(gradients)
            theta = theta - change
            self.errors[i] = self.BaseCost(self.y, self.X, theta)

        return theta

    def StochasticGradientDescent(
        self, theta: np.ndarray, n_epochs: int, M: int
    ) -> np.ndarray:
        """
        Performs stochastic gradient descent to optimize the model parameters.

        Args:
            theta (np.ndarray): The initial model parameters.
            n_epochs (int): The number of epochs to perform.
            M (int): The batch size.

        Returns:
            np.ndarray: The optimized model parameters.
        """
        m = self.n // M

        self.errors = np.zeros(n_epochs)
        self.errors.fill(np.nan)

        for epoch in range(n_epochs):
            self.scheduler.reset()
            for i in range(m):
                idxs = np.random.choice(self.n, M)
                xi = self.X[idxs]
                yi = self.y[idxs]

                gradients = self.gradient(yi, xi, theta)
                change = self.scheduler.update_change(gradients)

                theta = theta - change

            self.errors[epoch] = self.BaseCost(self.y, self.X, theta)

        return theta

    def predict(self, x: np.ndarray, theta: np.ndarray, dim: int = 2) -> np.ndarray:
        """
        Predicts the output values for the given input data using the model parameters.

        Args:
            x (np.ndarray): The input data.
            theta (np.ndarray): The model parameters.
            dim (int): The degree of the polynomial to use for the design matrix. Default is 2.

        Returns:
            np.ndarray: The predicted output values.
        """
        X = self.design(x, dim, len(x))
        return X @ theta
