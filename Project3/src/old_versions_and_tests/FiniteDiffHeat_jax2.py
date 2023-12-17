import jax.numpy as jnp
from jax import jit, lax
from functools import partial


@partial(jit, static_argnums=(0,))
def create_tridiagonal_matrix(n, r):
    # Diagonals
    main_diag = (1 - 2 * r) * jnp.ones(n)
    off_diag = r * jnp.ones(n - 1)
    # Creating the tridiagonal matrix structure
    upper = jnp.diag(off_diag, 1)
    middle = jnp.diag(main_diag, 0)
    lower = jnp.diag(off_diag, -1)
    return upper + middle + lower


@jit
def step_matrix_method(M, u_old):
    return jnp.dot(M, u_old)


@jit
def step(u, r):
    u_next = r * u[:-2] + (1 - 2 * r) * u[1:-1] + r * u[2:]
    return u_next.at[-1].set(r * u[-2] + (1 - 2 * r) * u[-1])


@jit
def solve_heat_equation(u_init, M, nt):
    def scan_fun(carry, _):
        u = carry
        u_next = step_matrix_method(M, u)
        return u_next, u_next
    _, unrolled = lax.scan(scan_fun, u_init, None, length=nt-1)
    return jnp.vstack((u_init[None, :], unrolled))


class FiniteDifferenceHeatImproved:
    def __init__(self, dx, dt, T, L=1):
        if dt > 0.5 * dx ** 2:
            raise ValueError(
                "dt must be less than or equal to 0.5 * dx^2 for stability")
        self.dx = dx
        self.dt = dt
        self.r = self.dt / self.dx ** 2
        self.nx = int(L / dx) + 1
        self.nt = int(T / dt) + 1
        self.x = jnp.linspace(0, L, self.nx)
        self.u0 = jnp.sin(jnp.pi * self.x)
        self.M = create_tridiagonal_matrix(self.nx, self.r)

    def solve(self):
        u_final = solve_heat_equation(self.u0, self.M, self.nt)
        return u_final
