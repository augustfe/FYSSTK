import time
from Activators import sigmoid
import jax.numpy as jnp
from jax import jit, lax
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Assume Activators.sigmoid is defined correctly

np.random.seed(2023)


@jit
def fast_z(a, W):
    return lax.add(lax.dot(a, W[1:]), lax.broadcast(W[0], (a.shape[0],)))


@jit
def feed_forward(X, weights, hidden_func=sigmoid, output_func=sigmoid):
    a = X
    for W in weights[:-1]:
        z = fast_z(a, W)
        a = hidden_func(z)

    z = fast_z(a, weights[-1])
    a = output_func(z)
    return a


def initialize_weights(dimensions):
    weights = []
    for i in range(len(dimensions) - 1):
        weight_matrix = np.random.randn(
            dimensions[i]+1, dimensions[i+1]) / np.sqrt(dimensions[i])
        weights.append(weight_matrix)
    return [jnp.array(w) for w in weights]


# Example usage
dimensions = (784, 128, 64, 64, 30, 10)

# Initialize weights
weights = initialize_weights(dimensions)

# Example data (could be a batch of inputs)
X_batch = jnp.ones((1, dimensions[0]))  # Replace with actual data

# Warmup - jitting functions
_ = feed_forward(X_batch, weights).block_until_ready()

# Benchmark fast_z feed_forward
start_time = time.time()
for _ in range(1000):
    outputs = feed_forward(X_batch, weights).block_until_ready()
end_time = time.time()
print("Feed-forward using fast_z: {:.6f} sec".format(end_time - start_time))

# Cleanup functions for another measurement
del feed_forward
del fast_z

# Another version of feed-forward without fast_z for comparison


@jit
def feed_forward_no_fast_z(X, weights, hidden_func=sigmoid, output_func=sigmoid):
    a = X
    for W in weights[:-1]:
        a = jnp.concatenate(
            (jnp.ones((a.shape[0], 1)), a), axis=1)  # Add bias term
        z = jnp.dot(a, W)
        a = hidden_func(z)

    # Add bias term for output layer
    a = jnp.concatenate((jnp.ones((a.shape[0], 1)), a), axis=1)
    z = jnp.dot(a, weights[-1])
    a = output_func(z)
    return a


# Warmup - jitting functions
_ = feed_forward_no_fast_z(X_batch, weights).block_until_ready()

# Benchmark feed_forward_no_fast_z
start_time = time.time()
for _ in range(1000):
    outputs = feed_forward_no_fast_z(X_batch, weights).block_until_ready()
end_time = time.time()
print("Feed-forward without fast_z: {:.6f} sec".format(end_time - start_time))
