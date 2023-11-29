from FiniteDiffHeat import MatrixMethod, IterationMethod
import numpy as np


def f_dt(dx):
    return (dx**2)/2


L = 1  # rod length
dx = 0.01
nx = int(L/dx) + 1
x = np.linspace(0, L, nx)

T = 1  # final time
dt = f_dt(dx)
nt = int(T/dt) + 1
t = np.linspace(0, T, nt)


# inital heat distribution
u0 = np.sin(np.pi*x)
MM = MatrixMethod(dx, dt, u0, nt)
IM = IterationMethod(dx, dt, u0, nt)
MM.solve()
IM.solve()
