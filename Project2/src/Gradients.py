import jax.numpy as jnp
import numpy as np
from jax import grad, jit, lax
from Schedules import Scheduler, Constant
from typing import Callable
from CostFuncs import fast_OLS
from utils import assign, design, update_theta
from line_profiler import profile

np.random.seed(2018)


class Gradients:
    """
    A class for computing gradients using various methods.

    Attributes:
        n (int): The number of data points.
        x (jnp.ndarray): The input data.
        y (jnp.ndarray): The output data.
        model (str): The type of model to use. Default is "OLS".
        method (str): The method to use for computing gradients. Default is "analytic".
        scheduler (Scheduler): The scheduler to use for updating the learning rate. Default is Constant.
        dim (int): The degree of the polynomial to use for the design matrix. Default is 2.
        lmbda (float): The regularization parameter. Default is None.

    Methods:
        __init__(self, n: int, x: jnp.ndarray, y: jnp.ndarray, model: str = "OLS", method: str = "analytic",
                 scheduler: Scheduler = Constant, dim: int = 2, lmbda: float = None) -> None:
            Initializes the Gradients object.
        pickfunc(self) -> Callable[[jnp.ndarray, jnp.ndarray, jnp.ndarray], float]:
            Returns the cost function to use based on the model type.
        costOLS(self, y: jnp.ndarray, X: jnp.ndarray, theta: jnp.ndarray) -> float:
            Computes the cost function for OLS regression.
        analyticOLS(self, y: jnp.ndarray, X: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
            Computes the gradient analytically for OLS regression.
        costRidge(self, y: jnp.ndarray, X: jnp.ndarray, theta: jnp.ndarray) -> float:
            Computes the cost function for Ridge regression.
        analyticRidge(self, y: jnp.ndarray, X: jnp.ndarray, theta: jnp.ndarray) -> jnp.ndarray:
            Computes the gradient analytically for Ridge regression.
        design(self, x: jnp.ndarray, dim: int, n: int = None) -> jnp.ndarray:
            Computes the design matrix for the given input data.
        GradientDescent(self, theta: jnp.ndarray, n_iter: int) -> jnp.ndarray:
            Performs gradient descent to optimize the model parameters.
        StochasticGradientDescent(self, theta: jnp.ndarray, n_epochs: int, M: int) -> jnp.ndarray:
            Performs stochastic gradient descent to optimize the model parameters.
        predict(self, x: jnp.ndarray, theta: jnp.ndarray, dim: int = 2) -> jnp.ndarray:
            Predicts the output values for the given input data using the model parameters.
    """

    def __init__(
        self,
        n: int,
        x: jnp.ndarray,
        y: jnp.ndarray,
        cost_func: Callable = fast_OLS,
        analytic_derivative: Callable = None,
        scheduler: Scheduler = Constant,
        dim: int = 2,
        lmbda: float = None,
    ) -> None:
        """
        Initializes the Gradients object.

        Args:
            n (int): The number of data points.
            x (jnp.ndarray): The input data.
            y (jnp.ndarray): The output data.
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
        self.X = design(x, dim=dim, n=n)
        self.lmbda = lmbda
        self.dim = dim
        self.cost_func = cost_func
        if analytic_derivative is None:
            self.gradient = jit(grad(self.cost_func, argnums=2))
        else:
            self.gradient = analytic_derivative

        self.scheduler = scheduler

    @profile
    def GradientDescent(self, theta: jnp.ndarray, n_iter: int) -> jnp.ndarray:
        """
        Performs gradient descent to optimize the model parameters.

        Args:
            theta (jnp.ndarray): The initial model parameters.
            n_iter (int): The number of iterations to perform.

        Returns:
            jnp.ndarray: The optimized model parameters.
        """
        self.errors = jnp.zeros(n_iter)

        for i in range(n_iter):
            gradients = self.gradient(self.X, self.y, theta)
            change = self.scheduler.update_change(gradients)
            theta = update_theta(theta, change)
            tmp = self.cost_func(self.X, self.y, theta)
            self.errors = assign(self.errors, i, tmp)

        return theta

    def StochasticGradientDescent(
        self, theta: jnp.ndarray, n_epochs: int, M: int
    ) -> jnp.ndarray:
        """
        Performs stochastic gradient descent to optimize the model parameters.

        Args:
            theta (jnp.ndarray): The initial model parameters.
            n_epochs (int): The number of epochs to perform.
            M (int): The batch size.

        Returns:
            jnp.ndarray: The optimized model parameters.
        """
        m = self.n // M

        self.errors = jnp.zeros(n_epochs)

        for epoch in range(n_epochs):
            self.scheduler.reset()
            for i in range(m):
                idxs = np.random.choice(self.n, M)
                xi = self.X[idxs]
                yi = self.y[idxs]

                gradients = self.gradient(xi, yi, theta)
                change = self.scheduler.update_change(gradients)

                theta = update_theta(theta, change)

            self.errors = assign(
                self.errors, epoch, self.cost_func(self.X, self.y, theta)
            )

        return theta

    def predict(self, x: jnp.ndarray, theta: jnp.ndarray, dim: int = 2) -> jnp.ndarray:
        """
        Predicts the output values for the given input data using the model parameters.

        Args:
            x (jnp.ndarray): The input data.
            theta (jnp.ndarray): The model parameters.
            dim (int): The degree of the polynomial to use for the design matrix. Default is 2.

        Returns:
            jnp.ndarray: The predicted output values.
        """
        X = design(x, dim, len(x))
        return lax.dot(X, theta)
