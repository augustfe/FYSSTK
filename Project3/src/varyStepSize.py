from FiniteDiffHeat import MatrixMethod
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
    dX = np.logspace(-6, -1, ndx)
    dT = np.logspace(-6, -1, ndt)
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
        error_arr, index=[f"{dx:.2f}" for dx in dX], columns=[f"{dt:.2e}" for dt in dT]
    )

    return df


"""
plotHeatmap(
df,
title=r"MSE for varying",
x_label=r"",
y_label=r,
savePlots=savePlots,
showPlots=showPlots,
figsPath=figspath,
saveName="momentum_heatmap_eta_rho",
)
"""
