from jax import jit
from functools import partial
import jax.numpy as jnp


@jit
def mse_loss(y_true, y_pred, weight=1.0):
    return weight * jnp.mean((y_true - y_pred) ** 2)


def loss_calculator(theta, batch, NNu, d2_dx2, d_dt, u0, x0, u_b, mu1, mu2, alpha):
    t_inner, x_inner, t_boundary, x_boundary = batch

    pde_loss_val = jnp.mean(
        (d_dt(t_inner, x_inner, theta) - alpha * d2_dx2(t_inner, x_inner, theta))**2)

    # Initial condition loss
    u_initial_pred = NNu(jnp.zeros_like(x0), x0, theta)
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


def cost_1d_heat(
    u0: jnp.ndarray,
    x0: jnp.ndarray,
    u_b: float,
    NNu: callable,
    d2_dx2: callable,
    d_dt: callable,
    mu1: float,
    mu2: float,
    alpha: float = 1.0
) -> callable:
    # return callable which should only be called with theta and batch f(theta, batch)
    return partial(jit(loss_calculator, static_argnums=(2, 3, 4, 5, 6, 7, 8, 9, 10)),
                   NNu=NNu, d2_dx2=d2_dx2, d_dt=d_dt, u0=u0, x0=x0, u_b=u_b, mu1=mu1, mu2=mu2, alpha=alpha)
