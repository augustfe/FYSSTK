from FiniteDiffHeat import MatrixMethod, IterationMethod
import numpy as np
import pandas as pd

np.random.seed(2023)


def max_dt(dx):
    """
    Calculates the maximum time step to ensure stability based on the spatial step size.

    Parameters:
    dx (float): Spatial step size.

    Returns:
    float: Maximum stable time step.
    """
    return 0.5 * dx**2


def MSE_n_samples(x, t, u_tilde, n_samples=100):
    """
    Estimates the Mean Squared Error (MSE) of the numerical solution against the analytical solution
    for n_samples randomly selected space-time points.

    Parameters:
    x (numpy.ndarray): 1D array of spatial coordinates.
    t (numpy.ndarray): 1D array of temporal coordinates.
    u_tilde (numpy.ndarray): 2D array containing the numerical solution with time along the rows and space along
    the columns.
    n_samples (int, optional): Number of samples to be used in the estimation. Defaults to 100.

    Returns:
    float: Mean Squared Error of the numerical solution.
    """
    indx_x = np.random.choice(np.arange(x.shape[0]), size=n_samples)
    indx_t = np.random.choice(np.arange(t.shape[0]), size=n_samples)

    sample_x = x[indx_x]
    sample_t = t[indx_t]

    # Analytical solution for the given samples
    u = np.exp(-np.pi**2 * sample_t) * np.sin(np.pi * sample_x)
    sample_u_tilde = u_tilde[indx_t, indx_x]

    # return MSE
    return np.mean((u - sample_u_tilde) ** 2)


def error_varying_stepsize(ndx, ndt, T, n_samples):
    """
    Calculates the error of the numerical solution over a range of spatial and temporal step sizes.

    Parameters:
    ndx (int): Number of spatial step points to consider.
    ndt (int): Number of temporal step points to consider.
    T (float): Total duration of the time simulation.
    n_samples (int): Number of random samples to be used for MSE calculation.

    Returns:
    pandas.DataFrame: DataFrame containing the calculated errors over the specified range of step sizes.
    """
    # smaller dx/dt makes it slow af
    dX = np.logspace(-6, -1, ndx)
    dT = np.logspace(-6, -2, ndt)
    error_arr = np.full((ndx, ndt), np.nan)

    for i, dx in enumerate(dX):
        max_dt_val = max_dt(dx)
        for j, dt in enumerate(dT):
            if dt > max_dt_val:
                continue
            else:
                mm = MatrixMethod(dx, dt, T)
                u_tilde = mm.solve()
                error_arr[i, j] = MSE_n_samples(
                    mm.x, mm.t, u_tilde, n_samples=n_samples
                )

    df = pd.DataFrame(
        error_arr, index=[f"{dx:.2e}" for dx in dX], columns=[f"{dt:.2e}" for dt in dT]
    )

    return df


def error_varying_dx(ndx, T, n_samples):
    """
    Calculates MSE of finite difference sol for different dt and dx. dt is set to its largest
    admissable value with repsect to numerical stability

    Parameters:
    ndx (int): Number of spatial step points to consider.
    ndt (int): Number of temporal step points to consider.
    T (float): Total duration of the time simulation.
    n_samples (int): Number of random samples to be used for MSE calculation.

    Returns:
    np.ndarray: 2D numpy array containing the calculated errors over the specified range of step sizes.
    """
    dX = np.logspace(-3, -1, ndx)
    error_arr = np.full(ndx, np.nan)
    for i, dx in enumerate(dX):
        # Assuming we want to use the maximum dt when calculating MSE
        dt = max_dt(dx)
        IM = IterationMethod(dx, dt, T)
        u_tilde = IM.solve()
        error_arr[i] = MSE_n_samples(IM.x, IM.t, u_tilde, n_samples=n_samples)
    # Return numpy array directly instead of pandas dataframe
    return error_arr, dX


if __name__ == "__main__":
    # from plotutils import plotHeatmap, PlotErrorPerVariable
    T = 0.4
    n_samples = 4

    # how to create heatmap
    """
    df = error_varying_stepsize(4, 4, T, 100)
    plotHeatmap(
        df,
        title=r"MSE for varying stepsize",
        x_label=r"$Delta x$",
        y_label=r"$Delta t$",
        savePlots=False,
        showPlots=True,
        figsPath=None,
        saveName="heatmap_MSE_varying_stepsize",
    )
    """
    from plotutils import plot_error_heat_dx
    # how to plot error vs dx
    errors, dX = error_varying_dx(2, T, n_samples)
    plot_error_heat_dx(
        errors=errors,
        dx_array=dX,
        savePlots=False,
        showPlots=True,
        figsPath=None,
        saveName='heat_eq_error_vs_dx'
    )
