import jax.numpy as np
from jax import grad, jit
import numpy as onp
from utils import assign
from pure_functions import (
    g_trial,
    g_analytic,
    cost_function,
)
import Activators
import FiniteDiffMain as FD
from jax.tree_util import Partial


class Solution:
    def __init__(
        self,
        Nx: int = 20,
        Nt: int = 20,
        x_range: (int, int) = (0, 1),
        T: int = 1,
        num_iter: int = 2500,
    ) -> None:
        self.Nx = Nx
        self.Nt = Nt
        self.x0 = x_range[0]
        self.L = x_range[1]
        self.T = T
        self.num_iter = num_iter
        self.x = np.linspace(self.x0, self.L, self.Nx)
        self.t = np.linspace(0, self.T, self.Nt)

        # Undefined parameters
        self.num_hidden_neurons = None
        self.lmb = None
        self.activation_function = None
        self.P = None

    def solve_pde_deep_neural_network(self, num_hidden_neurons, lmb, act_fun, seed=15):
        # onp.random.seed(seed)

        self.num_hidden_neurons = num_hidden_neurons
        self.lmb = lmb
        self.activation_function = act_fun

        # Rewrites activation function in pure_function.py to be able to use different ones
        N_hidden = np.size(self.num_hidden_neurons)

        P = [None] * (N_hidden + 1)

        P[0] = onp.random.randn(self.num_hidden_neurons[0], 2 + 1)

        for layer in range(1, N_hidden):
            P[layer] = onp.random.randn(
                self.num_hidden_neurons[layer], self.num_hidden_neurons[layer - 1] + 1
            )

        P[-1] = onp.random.randn(1, self.num_hidden_neurons[-1] + 1)

        cost_function_grad = jit(grad(cost_function, 0))

        for i in range(self.num_iter):
            cost_grad = cost_function_grad(P, self.x, self.t, act_fun)

            for layer in range(N_hidden + 1):
                P[layer] = P[layer] - self.lmb * cost_grad[layer]

        self.final_cost = cost_function(P, self.x, self.t, act_fun)

        self.P = P

    def solve_pde_numerical(self, dx: float, T: float = 1.0, mat: bool = False):
        self.numerical_sol = FD.compare(dx, T, mat)

    def solve_pde_analytical(self, x: np.ndarray, t: float):
        self.analytic_sol = FD.analytic(x, t)

    def stability(self):
        """
        Looks at the stability / difference between analytic, numerical and DNN
        such that it can be plottet with number of hidden nodes etc
        """

        self.g_dnn_ag = np.zeros((self.Nx, self.Nt))
        self.G_analytical = np.zeros((self.Nx, self.Nt))

        for i, x_ in enumerate(self.x):
            for j, t_ in enumerate(self.t):
                point = np.array([x_, t_])
                self.g_dnn_ag = assign(
                    self.g_dnn_ag,
                    (i, j),
                    g_trial(point, self.P, self.activation_function),
                )
                self.G_analytical = assign(self.G_analytical, (i, j), g_analytic(point))

        self.diff_ag = np.abs(self.g_dnn_ag - self.G_analytical)

        print(f"Final cost: {self.final_cost}")

        print(
            f"Max absolute difference between the analytical solution and the network: {np.max(self.diff_ag):g}"
        )


if __name__ == "__main__":
    sol = Solution(20, 20, (0, 1), 1, 2500)
    sol.solve_pde_deep_neural_network([100, 25], 0.01, Partial(Activators.sigmoid))
    sol.stability()
