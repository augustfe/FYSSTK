from jax import grad, jit, vmap
from functools import partial
import jax.numpy as jnp


def mse_loss(y_true, y_pred, weight=1.0):
    return weight * jnp.mean((y_true - y_pred) ** 2)


def NNu_unpacked(NNu):

    def unpacked(t, x, theta):
        return NNu(jnp.stack([t, x], axis=-1), theta)

    return unpacked


def make_d2_dx2_d_dt(NNu):
    dNNu_dx = grad(NNu, 1)
    d2NNu_dx2 = grad(dNNu_dx, 1)
    dNNu_dt = grad(NNu, 0)

    def d2_dx2(t, x, theta):
        return d2NNu_dx2(t, x, theta)

    def d_dt(t, x, theta):
        return dNNu_dt(t, x, theta)

    d2_dx2_jitted = jit(vmap(d2_dx2, in_axes=(0, 0, None)))
    d_dt_jitted = jit(vmap(d_dt, in_axes=(0, 0, None)))

    return d2_dx2_jitted, d_dt_jitted


def loss_calculator(theta, d2_dx2, d_dt, batch, NNu, u0, x0, u_b, mu1, mu2, alpha):
    t_inner, x_inner, t_boundary, x_boundary = batch

    pde_loss_val = jnp.mean(
        (d_dt(t_inner, x_inner, theta) - alpha * d2_dx2(t_inner, x_inner, theta))**2)

    # Initial condition loss
    u_initial_pred = NNu(theta, jnp.zeros_like(x0), x0)
    init_loss_val = mse_loss(u0, u_initial_pred)

    # Boundary condition loss
    t = jnp.concatenate([t_boundary, t_boundary])
    x = jnp.concatenate([jnp.full_like(t_boundary, x_boundary[0]),
                         jnp.full_like(t_boundary, x_boundary[1])])
    u_boundary_pred = NNu(theta, t, x)
    boundary_loss_val = mse_loss(
        u_boundary_pred, jnp.full_like(u_boundary_pred, u_b)
    )

    # Total loss calculation
    total_loss = (
        mu1 * pde_loss_val +
        mu2 * boundary_loss_val +
        (1 - mu2) * init_loss_val
    )

    return total_loss


def loss_1d_heat(
    u0: jnp.ndarray,
    x0: jnp.ndarray,
    u_b: float,
    mu1: float,
    mu2: float,
    alpha: float = 1
) -> callable:

    @partial(jit, static_argnums=(1, 2, 3))
    def wrapped_loss_calculator(weights, NNu, d2_dx2, d_dt, batch):
        return loss_calculator(weights, NNu, d2_dx2, d_dt, batch, u0, x0, u_b, mu1, mu2, alpha)

    def func(weights, batch: jnp.ndarray, NNu):

        NNu = NNu_unpacked(NNu)
        d2_dx2, d_dt = make_d2_dx2_d_dt(NNu)

        return wrapped_loss_calculator(weights, NNu, d2_dx2, d_dt, batch)

    return func
