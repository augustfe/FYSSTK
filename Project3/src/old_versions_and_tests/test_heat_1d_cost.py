from heat_1d_cost import loss_1d_heat
# import pytest
import jax.numpy as jnp
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_loss_1d_heat():
    u0 = jnp.array([1, 1])  # initial condition array
    x0 = jnp.array([2, 2])  # initial x array
    u_b = 0.5  # boundary value
    mu1 = 0.5  # PDE MSE weight
    mu2 = 0.5  # Boundary MSE weight
    alpha = 1  # known existent parameter

    # Some dummy function for NNu
    def dummy_func(input_, theta):
        return jnp.sum(input_) + theta

    theta = 0.1  # some initial theta
    batch = (jnp.array([0]), jnp.array([0]), jnp.array(
        [0, 0]), jnp.array([1, 1]))  # a example dummy batch

    loss = loss_1d_heat(u0, x0, u_b, mu1, mu2, alpha)  # get the loss function

    # calculate the result of the loss function
    result = loss(theta, batch, dummy_func)

    assert isinstance(result, float)  # test if the result is a float
    # for more rigorous tests, you might want to create a known condition where the result is known

    print('test_loss_1d_heat passed.')


if __name__ == "__main__":
    test_loss_1d_heat()
