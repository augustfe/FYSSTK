import jax.numpy as jnp
from jax import jit


def example_func(arr, M):
    return jnp.dot(arr, M)


def create_tridiagonal_matrix(n, r):
    """
    Create a tridiagonal matrix used in the finite difference method for
    solving the heat equation.

    Parameters:
    - n (int): The size of the matrix (number of spatial points).
    - r (float): The coefficient used in the finite difference method, usually defined as dt / dx^2.

    Returns:
    - M (jax.numpy.ndarray): The constructed tridiagonal matrix.
    """
    main_diag = 2 * jnp.ones(n)
    off_diag = -1 * jnp.ones(n - 1)
    matrix = jnp.diag(main_diag) + jnp.diag(off_diag, -1) + \
        jnp.diag(off_diag, 1)
    M = jnp.eye(n) - r * matrix
    return M


arr1 = jnp.arange(12).reshape(3, 4)
M1 = jnp.arange(4, 16).reshape(4, 3)

arr2 = jnp.arange(100, 120).reshape(4, 5)
M2 = jnp.arange(20).reshape(5, 4)


good_example_jit = jit(example_func)


print(good_example_jit(arr1, M1))
print(good_example_jit(arr2, M2))

example_jit = jit(create_tridiagonal_matrix, static_argnums=(0,))

print(example_jit(4, 2))
# (static_argnums=(0,))


def step_iteration_method_vectorized(u_old, r):
    """
    Perform one time step using an iterative method approach, vectorized to avoid explicit loops.

    Parameters:
    - u_old (jax.numpy.ndarray): The temperature distribution at the previous time step.
    - r (float): The coefficient used in the finite difference method, usually defined as dt / dx^2.

    Returns:
    - u_new (jax.numpy.ndarray): The updated temperature distribution after the time step.
    """
    u_new = jnp.zeros_like(u_old)
    u_new = u_new.at[1:-1].set(r * u_old[:-2] +
                               (1 - 2 * r) * u_old[1:-1] + r * u_old[2:])
    u_new = u_new.at[0].set((1 - 2 * r) * u_old[0] + r * u_old[1])
    u_new = u_new.at[-1].set(r * u_old[-2] + (1 - 2 * r) * u_old[-1])
    return u_new


good_example_jit = jit(step_iteration_method_vectorized)

print(good_example_jit(jnp.arange(12), 2))
print(good_example_jit(jnp.arange(14), 2))
