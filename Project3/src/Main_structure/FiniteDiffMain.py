from FiniteDiffHeat import IterationMethod
import jax.numpy as np
from jax import jit, lax
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pathlib import Path


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


if __name__ == "__main__":
    figures_dir = Path(__file__).parent.parent.parent / "figures/1dHeat"
    figures_dir.mkdir(exist_ok=True, parents=True)

    dxs = [1e-1, 1e-3]
    L = 1
    T = 0.5
    nSavePoints = 100
    for dx in dxs:
        dt = 0.5 * dx**2  # stability constraint
        x_arr = np.linspace(0, L, int(L / dx))

        IM = IterationMethod(dx, dt, nSavePoints=nSavePoints, T=T, L=L)
        u_numerical, solve_times = IM.solve()
        u_numerical = u_numerical.T

        T_, X_ = np.meshgrid(solve_times, x_arr)

        u_analytical = analytic(X_.flatten(), T_.flatten())
        u_analytical = u_analytical.reshape(X_.shape)

        u_difference = u_analytical - u_numerical
        mse = mean_squared_error(u_analytical, u_numerical)
        fig = plt.figure()

        ax = fig.add_subplot(111, projection="3d")

        ax.plot_surface(T_, X_, u_difference, cmap="viridis")
        title = (
            rf"Analytical solution - Forward Euler ($\Delta x = {dx}$, MSE : {mse:.2e})"
        )
        ax.set_title(title)

        ax.set_xlabel("Time")
        ax.set_ylabel("Space")
        ax.set_zlabel("u")

        plt.savefig(f"{figures_dir}/{dx=}.pdf", bbox_inches="tight")

        # plt.show()
