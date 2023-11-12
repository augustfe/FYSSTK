import jax.numpy as jnp
from jax import jit, lax
from functools import partial


@jit
def assign(arr, idx, val):
    arr = arr.at[idx].set(val)
    return arr


@jit
def assign_row(arr, idx, val):
    arr = arr.at[idx, :].set(val)
    return arr


@partial(jit, static_argnums=(1, 2))
def design(x: jnp.ndarray, dim: int, n: int) -> jnp.ndarray:
    """
    Computes the design matrix for the given input data.

    Args:
        x (jnp.ndarray): The input data.
        dim (int): The degree of the polynomial to use for the design matrix.
        n (int): The number of data points. Default is None.

    Returns:
        jnp.ndarray: The design matrix.
    """
    X = jnp.ones((n, dim + 1))
    for i in range(1, dim + 1):
        X = X.at[:, i].set((x**i).ravel())

    return X


@jit
def update_theta(theta: jnp.ndarray, change: jnp.ndarray):
    return lax.sub(theta, change)
