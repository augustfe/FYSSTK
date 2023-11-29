import numpy as np

pi2 = -np.pi**2


def analytical_heat(x, t):
    return np.exp(pi2*t)*np.sin(np.pi*x)
