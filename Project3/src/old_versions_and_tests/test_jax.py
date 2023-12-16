from jax import grad, jit
import jax.numpy as jnp
from jax import random
from functools import partial


key = random.PRNGKey(0)

"""
def sigmoid(x):
    return 0.5 * (jnp.tanh(x / 2) + 1)

# Outputs probability of a label being true.


def predict(W, b, inputs):
    return sigmoid(jnp.dot(inputs, W) + b)


# Build a toy dataset.
inputs = jnp.array([[0.52, 1.12,  0.77],
                   [0.88, -1.08, 0.15],
                   [0.52, 0.06, -1.30],
                   [0.74, -2.49, 1.39]])
targets = jnp.array([True, True, False, True])

# Training loss is the negative log-likelihood of the training examples.


def loss(W, b):
    preds = predict(W, b, inputs)
    label_probs = preds * targets + (1 - preds) * (1 - targets)
    return -jnp.sum(jnp.log(label_probs))


# Initialize random model coefficients
key, W_key, b_key = random.split(key, 3)
W = random.normal(W_key, (3,))
b = random.normal(b_key, ())


# Differentiate `loss` with respect to the first positional argument:
W_grad = grad(loss, argnums=0)(W, b)
print('W_grad', W_grad)

# Since argnums=0 is the default, this does the same thing:
W_grad = grad(loss)(W, b)
print('W_grad', W_grad)

# But we can choose different values too, and drop the keyword:
b_grad = grad(loss, 1)(W, b)
print('b_grad', b_grad)

# Including tuple values
W_grad, b_grad = grad(loss, (0, 1))(W, b)
print('W_grad', W_grad)
print('b_grad', b_grad)
"""


def f(g, t, x):
    return g(2 * t + 4 * x)


def g(x):
    return jnp.sin(x)


a = jnp.array([1.0, 2.0])
t, x = a

df = grad(f, 1)
df_with_g_sin = partial(df, g)
jit_df_with_g_sin = jit(df_with_g_sin)

print(jit_df_with_g_sin(t, x))