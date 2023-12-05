import jax.numpy as jnp
from jax import jit
import numpy as np
from Activators import sigmoid


def initialize_theta_He(dimensions):
    theta = []
    for i in range(len(dimensions) - 1):
        weight_matrix = np.random.randn(
            dimensions[i]+1, dimensions[i+1]) / np.sqrt(dimensions[i])
        theta.append(weight_matrix)

    return [jnp.array(w) for w in theta]


def initalize_theta_random(dimensions):
    theta = []
    for i in range(len(dimensions) - 1):
        # Weights
        weight_matrix = np.random.randn(
            dimensions[i] + 1, dimensions[i + 1]
        )

        # Bias
        weight_matrix[0, :] = np.random.randn(dimensions[i + 1]) * 0.01

        theta.append(weight_matrix)
    return [jnp.array(w) for w in theta]


@jit
def feed_forward(X, theta, hidden_func=sigmoid, output_func=sigmoid):
    a = X
    for W in theta[:-1]:
        a = jnp.concatenate(
            (jnp.ones((a.shape[0], 1)), a), axis=1)  # Add bias term
        z = jnp.dot(a, W)
        a = hidden_func(z)

    # Add bias term for output layer
    a = jnp.concatenate((jnp.ones((a.shape[0], 1)), a), axis=1)
    z = jnp.dot(a, theta[-1])
    a = output_func(z)
    return a


if __name__ == "__main__":

    # Example usage
    np.random.seed(2023)

    dimensions = (784, 128, 64, 10)
    # Initialize parms
    theta = initialize_theta_He(dimensions)

    # a bunch of ones as an example
    X_batch = jnp.ones((1, dimensions[0]))

    # feed forward pass
    outputs = feed_forward(X_batch, theta)
