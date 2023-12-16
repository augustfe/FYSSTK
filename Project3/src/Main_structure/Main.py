import matplotlib.pyplot as plt
from Solution import Solution
import numpy as onp


def main():

    sol = Solution('FD')
    dxs = [1e-1, 1e-3]
    L = 1
    T = 1
    for dx in dxs:
        x_arr = onp.arange(0, L, dx, dtype=float)
        sol.solve_pde_numerical(dx, T)
        for u_num, t in sol.numerical_sol:
            # plt.axes().set_aspect("equal")
            # plt.ylim(-1, 1)
            plt.plot(x_arr, u_num, label="Numerical")
            sol.solve_pde_analytical(x_arr, t)
            plt.plot(x_arr, sol.analytic_sol, label="Analytic")
            plt.title(rf"$\Delta x = {dx}$ at $t = {t}$")
            plt.legend()
            plt.show()


main()
