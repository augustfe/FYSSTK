from jax import grad, jit, vmap
from functools import partial
import jax.numpy as jnp


def mse_loss(y_true, y_pred, weight=1.0):
    return weight * jnp.mean((y_true - y_pred) ** 2)


def NNu_unpacked(NNu):
    """
    Unpack the function representation to separate time and space inputs.

    Parameters:
    NNu: function
        A neural network function taking a single argument.

    Returns:
    A new function that takes two arguments: time and space.
    """

    def unpacked(t, x):
        return NNu(jnp.stack([t, x], axis=-1))

    return unpacked


def make_d2_dx2_d_dt(NNu):
    """
    Create and jit-compile the second-order spatial derivative and time
    derivative functions of a neural network.

    Parameters:
    NNu: function
        A neural network function taking two arguments, time and space.

    Returns:
    d2_dx2: function
        A jit-compiled vmap-wrapper function for computing second-order
        spatial derivatives.
    d_dt: function
        A jit-compiled vmap-wrapper function for computing time derivatives.
    """
    dNNu_dx = grad(NNu, 1)
    d2NNu_dx2 = grad(dNNu_dx, 1)
    dNNu_dt = grad(NNu, 0)

    d2_dx2 = jit(vmap(d2NNu_dx2, in_axes=(0, 0)))
    d_dt = jit(vmap(dNNu_dt, in_axes=(0, 0)))

    return d2_dx2, d_dt


def loss_calculator(NNu, d2_dx2, d_dt, batch, u0, x0, u_b, mu1, mu2, alpha, target_batch, mse_weight):
    """
    Calculate the combined PDE and boundary condition loss, with an optional
    regression term for supervised learning.

    Parameters:
    NNu: function
        The neural network predicting the solution to the PDE.
    d2_dx2: function
        A function that computes the second spatial derivative of the NNu predictions.
    d_dt: function
        A function that computes the time derivative of the NNu predictions.
    batch: tuple of jnp.ndarray
        A tuple containing batch data for inner domain and boundary.
    u0: jnp.ndarray
        Initial condition values.
    x0: jnp.ndarray
        Positions where initial conditions are defined.
    u_b: float
        Boundary condition value.
    mu1: float
        Weighting factor for PDE loss.
    mu2: float
        Weighting factor for boundary condition loss.
    alpha: float
        Diffusion coefficient in the PDE.
    target_batch: jnp.ndarray or None
        Target values for supervised learning. Set to None if not used.
    mse_weight: float
        Weight for the regression term in the total loss function.

    Returns:
    A scalar representing the total loss.
    """
    t_inner, x_inner, t_boundary, x_boundary = batch

    # Inner domain loss
    u_inner_pred = NNu(t_inner, x_inner)
    pde_loss_val = mse_loss(
        d_dt(t_inner, x_inner) - alpha * d2_dx2(t_inner, x_inner),
        jnp.zeros_like(u_inner_pred)
    )

    # Initial condition loss
    u_initial_pred = NNu(jnp.zeros_like(x0), x0)
    init_loss_val = mse_loss(u0, u_initial_pred)

    # Boundary condition loss
    t = jnp.concatenate([t_boundary, t_boundary])
    x = jnp.concatenate([jnp.full_like(t_boundary, x_boundary[0]),
                         jnp.full_like(t_boundary, x_boundary[1])])
    u_boundary_pred = NNu(t, x)
    boundary_loss_val = mse_loss(
        u_boundary_pred, jnp.full_like(u_boundary_pred, u_b)
    )

    # Total loss calculation
    total_loss = (
        mu1 * pde_loss_val +
        mu2 * boundary_loss_val +
        (1 - mu2) * init_loss_val
    )

    # Regression term (if target_batch provided and weight is non-zero)
    if target_batch is not None and mse_weight > 0:
        regression_loss_val = mse_loss(target_batch, u_inner_pred, mse_weight)
        total_loss += regression_loss_val

    return total_loss


def Cost_1d_heat(
    u0: jnp.ndarray,
    x0: jnp.ndarray,
    u_b: float,
    mu1: float,
    mu2: float,
    alpha: float = 1,
    mse_weight: float = 0.0
):
    """
    Create a cost function for the 1D heat equation with configurable parameters.

    Parameters:
    u0: jnp.ndarray
        Initial condition values.
    x0: jnp.ndarray
        Positions where initial conditions are defined.
    u_b: float
        Boundary condition value.
    mu1: float
        Weighting factor for PDE loss.
    mu2: float
        Weighting factor for boundary condition loss.
    alpha: float
        Diffusion coefficient in the PDE.
    mse_weight: float
        Weight for the regression term in the total loss function. If set to 0,
        regression is not included in the loss calculation.

    Returns:
    A function that computes the total loss given a neural network and batch data.
    """
    @partial(jit, static_argnums=(0, 1, 2))
    def wrapped_loss_calculator(NNu, d2_dx2, d_dt, batch, target_batch):
        return loss_calculator(NNu, d2_dx2, d_dt, batch, u0, x0, u_b, mu1, mu2, alpha, target_batch, mse_weight)

    def func(NNu, batch: jnp.ndarray, target_batch: jnp.ndarray = None):
        """
        Compute the total loss for a given neural network and batch data.

        Parameters:
        NNu: function
            The neural network predicting the solution to the PDE.
        batch: jnp.ndarray
            A batch of data to be used in loss calculation.
        target_batch: jnp.ndarray or None, optional
            Target values for supervised learning. If not provided, regression is not included in the loss calculation.

        Returns:
        A scalar representing the total loss.
        """
        NNu = NNu_unpacked(NNu)
        d2_dx2, d_dt = make_d2_dx2_d_dt(NNu)

        return wrapped_loss_calculator(NNu, d2_dx2, d_dt, batch, target_batch)

    return func


"""
Example usage:
NNu = ... # Define your neural network structure here
batch = ... # Define your data batch here


#Initialize with parameters
cost_function = Cost_1d_heat(
    u0=jnp.array([0.0]),
    x0=jnp.array([0.0]),
    u_b=1.0,
    mu1=0.5,
    mu2=0.5,
    alpha=1.0
)

# total_loss = cost_function(NNu, batch)
"""
