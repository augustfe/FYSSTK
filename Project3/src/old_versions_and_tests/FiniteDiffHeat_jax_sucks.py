import jax.numpy as jnp
from jax import jit
from functools import partial


@partial(jit, static_argnums=(0,))
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


# Note that. However we only need to pass nx as
# static since u_old only changes shape whenever nx changes :)
@jit
def step_matrix_method(M, u_old):
    """
    Perform one time step using the matrix method approach.

    Parameters:
    - u_old (jax.numpy.ndarray): The temperature distribution at the previous time step.
    - M (jax.numpy.ndarray): The tridiagonal matrix representing the finite difference operator.

    Returns:
    - (jax.numpy.ndarray): The updated temperature distribution after the time step.
    """
    return jnp.dot(M, u_old)

# Note that both nx and u_old change. However we only need to pass nx as
# static since u_old only changes shape whenever nx changes :)


@jit
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


class FiniteDifferenceHeat:
    """
    Base class for finite difference method solutions to the heat equation.

    Attributes:
    - dx (float): Spatial discretization step.
    - dt (float): Temporal discretization step.
    - r (float): Discretization coefficient, defined as dt / dx^2.
    - nx (int): The number of spatial points.
    - nt (int): The number of temporal points.
    - x (jax.numpy.ndarray): The array of spatial coordinates.
    - t (jax.numpy.ndarray): The array of temporal coordinates.
    - u0 (jax.numpy.ndarray): The initial temperature distribution.
    """

    def __init__(self, dx, dt, T, L=1):
        if dt > 0.5*dx**2:
            raise ValueError(
                r"dt can not be larger than $\frac{1}{2dx^2} for numerical stability$")
        self.dx = dx
        self.dt = dt
        # r factor needed in iteration
        self.r = self.dt / self.dx**2

        self.nx = int(L / dx) + 1
        self.nt = int(T / dt) + 1
        # x and t needed for plotting
        self.x = jnp.linspace(0, L, self.nx)
        self.t = jnp.linspace(0, T, self.nt)
        # initial condition
        self.u0 = jnp.sin(jnp.pi * self.x)

    def reset_inital_conditions(self):
        """
        Reset the initial temperature distribution as the first row of the solution grid.
        """
        self.u = jnp.zeros((self.nt, self.nx))
        self.u = self.u.at[0].set(self.u0)


class MatrixMethod(FiniteDifferenceHeat):
    """
    Class that implements the matrix method for solving the heat equation using finite difference.
    """

    def solve(self):
        """
        Solve the heat equation using the matrix method.

        Returns:
        - (jax.numpy.ndarray): The matrix with the temperature distribution over time and space.
        """
        self.reset_inital_conditions()
        M = create_tridiagonal_matrix(self.nx, self.r)
        # loop is not jit decorated as there is loop dependency on prev/u_old for
        # each iteration
        for k in range(1, self.nt):
            self.u = self.u.at[k].set(step_matrix_method(self.u[k-1, :], M))
        return self.u


class IterationMethod(FiniteDifferenceHeat):
    """
    Class that implements the iterative method for solving the heat equation using finite difference.
    """

    def solve(self):
        """
        Solve the heat equation using an iterative method.

        Returns:
        - (jax.numpy.ndarray): The matrix with the temperature distribution over time and space.
        """
        self.reset_inital_conditions()
        # loop is not jit decorated as there is loop dependency on prev/u_old for
        # each iteration
        for k in range(1, self.nt):
            self.u = self.u.at[k].set(step_iteration_method_vectorized(self.nx,
                                                                       self.u[k-1, :], self.r))
        return self.u
