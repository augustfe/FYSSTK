from FiniteDiffHeat import MatrixMethod, IterationMethod
import jax.numpy as np
from jax import jit, lax


@jit
def analytic(x: np.ndarray, t: float):
    res = lax.mul(
        lax.exp(
            lax.mul(
                -lax.integer_pow(
                    np.pi,
                    2,
                ),
                t,
            )
        ),
        lax.sin(
            lax.mul(
                np.pi,
                x,
            )
        ),
    )
    return res
    # return np.exp(-np.pi**2 * t) * np.sin(np.pi * x)


def compare(
    dx: float, T: float = 1.0, mat: bool = False
) -> tuple[tuple[np.ndarray, int]]:
    dt = 0.5 * dx**2
    L = 1
    u_iter = None

    # Matrix method
    if mat:
        MM = MatrixMethod(dx, dt, T, L)
        u_mat = MM.solve()
        return u_mat

    # Iteration method
    IM = IterationMethod(dx, dt, T, L)
    u_iter = IM.solve()

    return u_iter


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    dxs = [1e-1, 1e-3]
    L = 1
    T = 1
    for dx in dxs:
        x_arr = np.arange(0, L, dx, dtype=float)
        u_nums = compare(dx, T)
        for u_num, t in u_nums:
            # plt.axes().set_aspect("equal")
            plt.plot(x_arr, u_num, label="Numerical")
            analytic_sol = analytic(x_arr, t)
            plt.plot(x_arr, analytic_sol, label="Analytic")
            plt.title(rf"$\Delta x = {dx}$ at $t = {t}$")
            plt.legend()
            # plt.show()
