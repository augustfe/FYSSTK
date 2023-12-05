import jax.numpy as jnp
from jax import jit, grad
from heat_1d_cost import loss_1d_heat
from functools import partial
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_Cost_1d_heat_random_func():
    def simpleFunc(input):
        t, x = input.T
        return t + x

    cost_function = loss_1d_heat(
        u0=jnp.array([0.5]),
        x0=jnp.array([0.0]),
        u_b=1.0,
        mu1=0.5,
        mu2=0.5,
        alpha=1.0
    )

    # Fake data for the inner and boundary domain.
    t_inner = jnp.array([0.1, 0.2, 0.3])
    x_inner = jnp.array([0.1, 0.2, 0.3])
    t_boundary = jnp.array([0.0, 1.0])
    x_boundary = jnp.array([-1.0, 1.0])
    batch = (t_inner, x_inner, t_boundary, x_boundary)
    cost_function = jit(partial(cost_function, simpleFunc))

    loss_val = cost_function(batch)
    d_cost_func = jit(grad(cost_function))
    gradient = d_cost_func(batch)
    print([arr.shape for arr in batch])
    print(f"Loss value: {loss_val}")
    print(f"gradient: {gradient}")


def test_Cost_1d_heat_analytical_sol():
    def analytical_heat(arr):
        t, x = arr.T
        return jnp.exp(-jnp.pi**2 * t) * jnp.sin(jnp.pi * x)
    L = 1
    x = jnp.linspace(0, L, 20)

    u0 = jnp.sin(jnp.pi*x)
    cost_function = loss_1d_heat(
        u0=u0,
        x0=x,
        u_b=0,
        mu1=0.5,
        mu2=0.5,
        alpha=1.0
    )

    t_inner = jnp.array([0.1, 0.2, 0.3])
    x_inner = jnp.array([0.1, 0.2, 0.3])
    t_boundary = jnp.array([0.1, 0.13])
    x_boundary = jnp.array([0.0, 1.0])
    batch = (t_inner, x_inner, t_boundary, x_boundary)
    cost_function = jit(partial(cost_function, analytical_heat))
    loss_val = cost_function(batch)
    d_cost_func = jit(grad(cost_function))
    gradient = d_cost_func(batch)
    print(f"Loss value: {loss_val}")
    print(f"gradient: {gradient}")


if __name__ == "__main__":
    # test_Cost_1d_heat_random_func()
    test_Cost_1d_heat_analytical_sol()
