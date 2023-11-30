import numpy as np


def create_tridiagonal_matrix(n, r):
    """
    Create a tridiagonal matrix used in the finite difference method for
    solving the heat equation.
    """
    main_diag = 2 * np.ones(n)
    off_diag = -1 * np.ones(n - 1)
    matrix = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
    M = np.eye(n) - r * matrix
    return M


def step_matrix_method(M, u_old):
    """
    Perform one time step using the matrix method approach.
    """
    return np.dot(M, u_old)


def step_iteration_method_vectorized(u_old, r):
    """
    Perform one time step using an iterative method approach.
    """
    u_new = np.zeros_like(u_old)
    u_new[1:-1] = r * u_old[:-2] + (1 - 2 * r) * u_old[1:-1] + r * u_old[2:]
    u_new[0] = (1 - 2 * r) * u_old[0] + r * u_old[1]
    u_new[-1] = r * u_old[-2] + (1 - 2 * r) * u_old[-1]
    return u_new


class FiniteDifferenceHeat:
    """
    Base class for finite difference method solutions to the heat equation.
    """

    def __init__(self, dx, dt, T, L=1):
        if dt > 0.5 * dx ** 2:
            raise ValueError(
                "dt cannot be larger than 0.5 * dx^2 for numerical stability")
        self.dx = dx
        self.dt = dt
        self.r = self.dt / self.dx ** 2
        self.nx = int(L / dx) + 1
        self.nt = int(T / dt) + 1
        self.x = np.linspace(0, L, self.nx)
        self.t = np.linspace(0, T, self.nt)
        self.u0 = np.sin(np.pi * self.x)

    def reset_initial_conditions(self):
        """
        Reset the initial temperature distribution as the first row of the solution grid.
        """
        self.u = np.zeros((self.nt, self.nx))
        self.u[0] = self.u0


class MatrixMethod(FiniteDifferenceHeat):
    """
    Class that implements the matrix method for solving the heat equation using finite difference.
    """

    def solve(self):
        """
        Solve the heat equation using the matrix method.
        """
        self.reset_initial_conditions()
        M = create_tridiagonal_matrix(self.nx, self.r)
        for k in range(1, self.nt):
            self.u[k] = step_matrix_method(M, self.u[k-1])
        return self.u


class IterationMethod(FiniteDifferenceHeat):
    """
    Class that implements the iterative method for solving the heat equation using finite difference.
    """

    def solve(self):
        """
        Solve the heat equation using an iterative method.
        """
        self.reset_initial_conditions()
        for k in range(1, self.nt):
            self.u[k] = step_iteration_method_vectorized(self.u[k-1], self.r)
        return self.u
