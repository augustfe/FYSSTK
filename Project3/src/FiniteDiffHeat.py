import numpy as np


# @profile
def create_tridiagonal_matrix(n):
    main_diag = 2 * np.ones(n)
    off_diag = -1 * np.ones(n - 1)
    matrix = np.diag(main_diag) + np.diag(off_diag, -1) + np.diag(off_diag, 1)
    return matrix


class FiniteDifferenceHeat:
    def __init__(self, dx, dt, T, L=1):
        self.dx = dx
        self.dt = dt
        self.r = self.dt/self.dx**2
        self.nx = int(L/dx) + 1  # numer of points along the x-axis
        self.nt = int(T/dt) + 1  # number of points along the t-axis

        # x and t used for plotting heat
        self.x = np.linspace(0, L, self.nx)
        self.t = np.linspace(0, T, self.nt)

        # initial heat distrobution
        self.u0 = np.sin(np.pi*self.x)

    def reset_inital_conditions(self):
        self.u = np.zeros((self.nt, self.nx))
        self.u[0, :] = self.u0


class MatrixMethod(FiniteDifferenceHeat):
    # @profile
    def solve(self):
        self.reset_inital_conditions()
        A = create_tridiagonal_matrix(self.nx)
        self.M = np.eye(self.nx) - self.r*A
        for k in range(1, self.nt):
            self.u[k, :] = self.M @ self.u[k-1]
        return self.u


class IterationMethod(FiniteDifferenceHeat):
    # @profile
    def solve(self):
        self.reset_inital_conditions()
        for k in range(1, self.nt):
            self.itr(k)
        return self.u

    def itr(self, k):
        u = self.u[k-1, :]
        u_new = np.zeros(self.nx)
        u_new[0] = (1-2*self.r)*u[0] + self.r*u[1]
        for i in range(1, self.nx-1):
            u_new[i] = self.r*u[i-1] + (1-2*self.r)*u[i] + self.r*u[i+1]
        u_new[-1] = self.r*u[i-1] + (1-2*self.r)*u[i]
        self.u[k:,] = u_new
