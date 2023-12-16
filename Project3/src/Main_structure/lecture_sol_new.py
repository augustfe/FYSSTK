import jax.numpy as np
from jax import jacobian, hessian, grad, jit, vmap
import numpy as onp
from matplotlib import cm
from matplotlib import pyplot as plt
from utils import assign


@jit
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


@jit
def deep_neural_network(deep_params, x):
    # x is now a point and a 1D numpy array; make it a column vector
    num_coordinates = np.size(x, 0)
    x = x.reshape(num_coordinates, -1)

    num_points = np.size(x, 1)
    # N_hidden is the number of hidden layers
    # -1 since params consist of parameters to all the hidden layers AND the output layer
    N_hidden = len(deep_params) - 1

    # Assume that the input layer does nothing to the input x
    x_input = x
    x_prev = x_input

    # Hidden layers:

    for layer in range(N_hidden):
        # From the list of parameters P; find the correct weigths and bias for this layer
        w_hidden = deep_params[layer]

        # Add a row of ones to include bias
        x_prev = np.concatenate((np.ones((1, num_points)), x_prev), axis=0)

        z_hidden = np.matmul(w_hidden, x_prev)
        x_hidden = sigmoid(z_hidden)

        # Update x_prev such that next layer can use the output from this layer
        x_prev = x_hidden

    # Output layer:

    # Get the weights and bias for this layer
    w_output = deep_params[-1]

    # Include bias:
    x_prev = np.concatenate((np.ones((1, num_points)), x_prev), axis=0)

    z_output = np.matmul(w_output, x_prev)
    x_output = z_output

    return x_output[0][0]

# Define the trial solution and cost function


@jit
def u(x):
    return np.sin(np.pi * x)


@jit
def g_trial(point, P):
    x, t = point
    return (1 - t) * u(x) + x * (1 - x) * t * deep_neural_network(P, point)


# The right side of the ODE:
@jit
def f(point):
    return 0.0


@jit
def innercost(point, P):
    # The inner cost function, evaluating each point
    g_t_jacobian_func = jit(jacobian(g_trial))
    g_t_hessian_func = jit(hessian(g_trial))

    g_t_jacobian = g_t_jacobian_func(point, P)
    g_t_hessian = g_t_hessian_func(point, P)

    g_t_dt = g_t_jacobian[1]
    g_t_d2x = g_t_hessian[0][0]

    func = f(point)

    err_sqr = ((g_t_dt - g_t_d2x) - func) ** 2
    return err_sqr


# The cost function:
@jit
def cost_function(P, x, t):
    total_points = np.array([[x_, t_] for x_ in x for t_ in t])
    vec_cost = vmap(innercost, (0, None), 0)
    cost_sum = np.sum(vec_cost(total_points, P))
    return cost_sum / (np.size(x) * np.size(t))


# For comparison, define the analytical solution
@jit
def g_analytic(point):
    x, t = point
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)


class Solution:

    def __init__(self,
                 Nx: int,
                 Nt: int,
                 x_range: (int, int),
                 T: int,
                 num_hidden_neurons: [int, int],
                 num_iter: int,
                 lmb: float) -> None:

        self.Nx = Nx
        self.Nt = Nt
        self.x0 = x_range[0]
        self.x_end = x_range[1]
        self.T = T
        self.num_hidden_neurons = num_hidden_neurons
        self.num_iter = num_iter
        self.lmb = lmb
        self.x = np.linspace(self.x0, self.x_end, self.Nx)
        self.t = np.linspace(0, self.T, self.Nt)

    def solve_pde_deep_neural_network(self):

        N_hidden = np.size(self.num_hidden_neurons)

        P = [None] * (N_hidden + 1)

        P[0] = onp.random.randn(self.num_hidden_neurons[0], 2 + 1)

        for layer in range(1, N_hidden):
            P[layer] = onp.random.randn(
                self.num_hidden_neurons[layer], self.num_hidden_neurons[layer - 1] + 1)

        P[-1] = onp.random.randn(1, self.num_hidden_neurons[-1] + 1)

        cost_function_grad = jit(grad(cost_function, 0))

        for i in range(self.num_iter):
            cost_grad = cost_function_grad(P, self.x, self.t)

            for layer in range(N_hidden + 1):
                P[layer] = P[layer] - self.lmb * cost_grad[layer]

        self.final_cost = cost_function(P, self.x, self.t)

        self.P = P

    def run(self, seed, show=True):

        onp.random.seed(seed)

        self.solve_pde_deep_neural_network()

        self.g_dnn_ag = np.zeros((self.Nx, self.Nt))
        self.G_analytical = np.zeros((self.Nx, self.Nt))

        for i, x_ in enumerate(self.x):
            for j, t_ in enumerate(self.x):

                point = np.array([x_, t_])
                self.g_dnn_ag = assign(
                    self.g_dnn_ag, (i, j), g_trial(point, self.P))
                self.G_analytical = assign(
                    self.G_analytical, (i, j), g_analytic(point))

        self.diff_ag = np.abs(self.g_dnn_ag - self.G_analytical)

        print(f"Final cost: {self.final_cost}")

        print(
            f"Max absolute difference between the analytical solution and the network: {np.max(self.diff_ag):g}"
        )

        if show:

            self.plot_solution()

    def plot_solution(self):

        T, X = np.meshgrid(self.t, self.x)

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")
        ax.set_title(
            f"Solution from the deep neural network w/ {len(self.num_hidden_neurons)} layer"
        )
        ax.plot_surface(T, X, self.g_dnn_ag, linewidth=0,
                        antialiased=False, cmap=cm.viridis)
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Position $x$")

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")
        ax.set_title("Analytical solution")
        ax.plot_surface(T, X, self.G_analytical, linewidth=0,
                        antialiased=False, cmap=cm.viridis)
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Position $x$")

        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(projection="3d")
        ax.set_title("Difference")
        ax.plot_surface(T, X, self.diff_ag, linewidth=0,
                        antialiased=False, cmap=cm.viridis)
        ax.set_xlabel("Time $t$")
        ax.set_ylabel("Position $x$")

        # Take some slices of the 3D plots just to see the solutions at particular times
        indx1 = 0
        indx2 = int(self.Nt / 2)
        indx3 = self.Nt - 1

        t1 = self.t[indx1]
        t2 = self.t[indx2]
        t3 = self.t[indx3]

        # Slice the results from the DNN
        res1 = self.g_dnn_ag[:, indx1]
        res2 = self.g_dnn_ag[:, indx2]
        res3 = self.g_dnn_ag[:, indx3]

        # Slice the analytical results
        res_analytical1 = self.G_analytical[:, indx1]
        res_analytical2 = self.G_analytical[:, indx2]
        res_analytical3 = self.G_analytical[:, indx3]

        # Plot the slices
        plt.figure(figsize=(10, 10))
        plt.title(f"Computed solutions at time = {t1:g}")
        plt.plot(self.x, res1)
        plt.plot(self.x, res_analytical1)
        plt.legend(["dnn", "analytical"])

        plt.figure(figsize=(10, 10))
        plt.title(f"Computed solutions at time = {t2:g}")
        plt.plot(self.x, res2)
        plt.plot(self.x, res_analytical2)
        plt.legend(["dnn", "analytical"])

        plt.figure(figsize=(10, 10))
        plt.title(f"Computed solutions at time = {t3:g}")
        plt.plot(self.x, res3)
        plt.plot(self.x, res_analytical3)
        plt.legend(["dnn", "analytical"])
        plt.show()


if __name__ == "__main__":

    sol = Solution(20, 20, (0, 1), 1, [100, 25], 2500, 0.01)
    sol.run(15)
