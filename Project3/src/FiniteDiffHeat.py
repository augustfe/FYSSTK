import numpy as np


# @profile
def create_tridiagonal_matrix(n):
    main_diag = 2 * np.ones(n)
    off_diag = -1 * np.ones(n - 1)
    matrix = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
    return matrix


class FiniteDifferenceHeat:
    def __init__(self, dx, dt, u0, nt):
        self.dx = dx
        self.dt = dt
        self.r = self.dt/self.dx**2
        self.nx = u0.size  # numer of points along the x-axis
        self.u = np.zeros((nt, self.nx))
        self.u[0, :] = u0
        self.nt = nt


class MatrixMethod(FiniteDifferenceHeat):
    # @profile
    def solve(self):
        A = create_tridiagonal_matrix(self.nx)
        self.M = np.eye(self.nx) - self.r*A
        for k in range(1, self.nt):
            self.u[k, :] = self.M @ self.u[k-1]


class IterationMethod(FiniteDifferenceHeat):
    # @profile
    def solve(self):
        for k in range(1, self.nt):
            self.itr(k)

    def itr(self, k):
        u = self.u[k-1, :]
        u_new = np.zeros(self.nx)
        u_new[0] = (1-2*self.r)*u[0] + self.r*u[1]
        for i in range(1, self.nx-1):
            u_new[i] = self.r*u[i-1] + (1-2*self.r)*u[i] + self.r*u[i+1]
        u_new[-1] = self.r*u[i-1] + (1-2*self.r)*u[i]
        self.u[k:,] = u_new
