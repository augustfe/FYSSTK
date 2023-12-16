# import numpy as np
from jax import jit, lax
import jax.numpy as np
from utils import assign
from jax.experimental import sparse


# @jit
def create_tridiagonal_matrix(n: int, r: float) -> np.ndarray:
    """
    Create a tridiagonal matrix used in the finite difference method for
    solving the heat equation.
    """
    main_diag = 2 * np.ones(n)
    off_diag = -1 * np.ones(n - 1)
    matrix = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
    M = np.eye(n) - r * matrix
    M_sp = sparse.BCOO.fromdense(M)
    return M_sp


@jit
@sparse.sparsify
def step_matrix_method(M: np.ndarray, u_old: np.ndarray) -> np.ndarray:
    """
    Perform one time step using the matrix method approach.
    """
    return lax.dot(M, u_old)


@jit
def assign_middle(u_new, u_mid):
    u_new = u_new.at[1:-1].set(u_mid)
    return u_new


@jit
def step_iteration_method_vectorized(u_old: np.ndarray, r: float) -> np.ndarray:
    """
    Perform one time step using an iterative method approach.
    """
    u_new = np.zeros_like(u_old)

    u_top = (1 - 2 * r) * u_old[0] + r * u_old[1]
    u_mid = r * u_old[:-2] + (1 - 2 * r) * u_old[1:-1] + r * u_old[2:]
    u_bot = r * u_old[-2] + (1 - 2 * r) * u_old[-1]

    u_new = assign_middle(u_new, u_mid)
    u_new = assign(u_new, 0, u_top)
    u_new = assign(u_new, -1, u_bot)

    return u_new


class FiniteDifferenceHeat:
    """
    Base class for finite difference method solutions to the heat equation.
    """

    def __init__(self, dx: float, dt: float, T: int, L: int = 1) -> None:
        # TODO: Add functionality for changing midpoint.
        if dt > 0.5 * dx**2:
            raise ValueError(
                "dt cannot be larger than 0.5 * dx^2 for numerical stability"
            )
        self.dx = dx
        self.dt = dt
        self.r = self.dt / self.dx**2
        self.nx = int(L // dx + 1)
        self.nt = int(T // dt + 1)
        self.x = np.linspace(0, L, self.nx)
        self.t = np.linspace(0, T, self.nt)
        self.u0 = np.sin(np.pi * self.x)

    def reset_initial_conditions(self) -> None:
        """
        Reset the initial temperature distribution as the first row of the solution grid.
        """
        # TODO: Remove and consider consequences.
        # self.u = np.zeros((self.nt, self.nx))
        # self.u = assign(self.u, 0, self.u0)
        # self.u[0] = self.u0


class MatrixMethod(FiniteDifferenceHeat):
    """
    Class that implements the matrix method for solving the heat equation using finite difference.
    """

    def solve(self) -> np.ndarray:
        """
        Solve the heat equation using the matrix method.
        """
        self.reset_initial_conditions()
        M = create_tridiagonal_matrix(self.nx, self.r)
        print(f"Will compute {self.nt} time steps for {self.nx} spatial steps")
        mid = self.nt // 2

        u_mid = solve_mat(self.u0, 0, mid, M)
        u_new = solve_mat(u_mid, mid, self.nt, M)

        return (self.u0, 0.0), (u_mid, mid * self.dt), (u_new, (self.nt - 1) * self.dt)


class IterationMethod(FiniteDifferenceHeat):
    """
    Class that implements the iterative method for solving the heat equation using finite difference.
    """

    def solve(self) -> tuple[tuple[np.ndarray, int]]:
        """
        Solve the heat equation using an iterative method.
        """
        self.reset_initial_conditions()
        print(f"Will compute {self.nt} time steps for {self.nx} spatial steps")
        mid = self.nt // 2

        u_mid = solve_iter(self.u0, 0, mid, self.r)
        u_new = solve_iter(u_mid, mid, self.nt, self.r)

        return (self.u0, 0.0), (u_mid, mid * self.dt), (u_new, (self.nt - 1) * self.dt)


@jit
def solve_iter(u_init: np.ndarray, lower: int, upper: int, r: float) -> np.ndarray:
    """Iterate with the vectorized iterative method.

    Broken into own function to allow for jit compilation, as jit normally
    tries to unroll loops into just a list of instructions, the number of
    iterations has to be known at compile time. This also causes extremely
    long compilation times for long loops, however this can be alleviated
    through jax.lax.fori_loop. This function takes in the iteration bounds,
    as well as the body of the iteration loop, giving massive performance
    benefits.

    bodyfun must have the signature (int, T) -> T, where T is the type of
    the value to be iterated over.

    Args:
        u_init (np.ndarray): Initial values to iterate from.
        lower (int): Starting index of the iteration.
        upper (int): End index of the iteration.
        r (float): dt/dx^2, stability criterion.

    Returns:
        np.ndarray: value after (upper - lower) iterations.
    """
    val = u_init

    def bodyfun(i: int, val: np.ndarray) -> np.ndarray:
        val = step_iteration_method_vectorized(val, r)
        return val

    val = lax.fori_loop(lower, upper, bodyfun, val)
    return val


@jit
def solve_mat(u_init: np.ndarray, lower: int, upper: int, M: np.ndarray) -> np.ndarray:
    """Iterate with the matrix method.

    Args:
        u_init (np.ndarray): Initial values to iterate from.
        lower (int): Lower bound for iteration.
        upper (int): Upper bound for iteration.
        M (np.ndarray): Sparse scheme matrix for the iteration.

    Returns:
        np.ndarray: Values after (upper - lower) iterations.
    """
    val = u_init

    def bodyfun(i: int, val: np.ndarray) -> np.ndarray:
        val = step_matrix_method(M, val)
        return val

    val = lax.fori_loop(lower, upper, bodyfun, val)
    return val
