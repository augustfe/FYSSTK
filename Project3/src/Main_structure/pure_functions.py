from typing import Any
import jax.numpy as np
from jax import jacobian, hessian, grad, jit, vmap
import numpy as onp
from utils import assign
import Activators
from typing import Callable


@jit
def deep_neural_network(deep_params, x, activation_func: Callable[[float], float]):
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
        x_hidden = activation_func(z_hidden)

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
def g_trial(point, P, activation_func: Callable[[float], float]):
    x, t = point
    return (1 - t) * u(x) + x * (1 - x) * t * deep_neural_network(P, point, activation_func)


# The right side of the ODE:
@jit
def f(point):
    return 0.0


@jit
def innercost(point, P, activation_func: Callable[[float], float]):
    # The inner cost function, evaluating each point
    g_t_jacobian_func = jit(jacobian(g_trial))
    g_t_hessian_func = jit(hessian(g_trial))

    g_t_jacobian = g_t_jacobian_func(point, P, activation_func)
    g_t_hessian = g_t_hessian_func(point, P, activation_func)

    g_t_dt = g_t_jacobian[1]
    g_t_d2x = g_t_hessian[0][0]

    func = f(point)

    err_sqr = ((g_t_dt - g_t_d2x) - func) ** 2
    return err_sqr


# The cost function:
@jit
def cost_function(P, x, t, activation_func: Callable[[float], float]):
    total_points = np.array([[x_, t_] for x_ in x for t_ in t])
    vec_cost = vmap(innercost, (0, None, None), 0)
    cost_sum = np.sum(vec_cost(total_points, P, activation_func))
    return cost_sum / (np.size(x) * np.size(t))


# For comparison, define the analytical solution
@jit
def g_analytic(point):
    x, t = point
    return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)
