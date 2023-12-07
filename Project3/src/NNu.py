from jax import jit, grad, vmap
import jax.numpy as jnp
import numpy as np


def unpack(f):
    def unpacked(t, x, theta):
        return f(t, x, theta)[0]
    return unpacked


def NNu_unpacked(NNu, theta):
    def unpacked(t, x):
        return NNu(jnp.stack([t, x], axis=-1), theta).reshape(-1,)
    return unpacked


def dNNu_dx_vectorized(NNu_up, t, x):
    # Take the gradient of NNu_up (unpacked) with respect to x at each value of t and x.
    # This vmap will apply the grad over the first axis of t and x
    dNNu_dx = jit(vmap(grad(NNu_up, 1), (0, None), 0))
    return dNNu_dx(t, x)


def dNNu_dt_vectorized(NNu_up, t, x):
    # Take the gradient of NNu_up (unpacked) with respect to t at each value of t and x.
    # This vmap will apply the grad over the first axis of t and x
    dNNu_dt = jit(vmap(grad(NNu_up, 0), (None, 0), 0))
    return dNNu_dt(t, x)


def make_d2_dx2_d_dt(NNu):

    # First derivatives
    dNNu_dx = dNNu_dx_vectorized(NNu)
    dNNu_dt = dNNu_dt_vectorized(NNu)

    # Second derivative with respect to x
    d2NNu_dx2 = 

    # Return the second derivative with respect to x and the first derivative with respect to t
    return d2NNu_dx2, dNNu_dt


def test_no_hidden_single_input():
    from feed_forward_functional import fully_connected as nnu

    NNu = NNu_unpacked(nnu)

    d2_dx2, d_dt = make_d2_dx2_d_dt(NNu)
    theta = [jnp.ones((3, 1))]  # colapse the two input args to one output

    t = jnp.asarray([0.3])
    x = jnp.asarray([0.9])
    print("\nTesting with single input no hidden layers...")
    nnu = NNu(t, x, theta)
    print(nnu)
    d_dt = d_dt(t, x, theta)
    print(d_dt)
    d2_dx2 = d2_dx2(t, x, theta)

    print("NNu:", nnu)
    print("d_dt:", d_dt)
    print("d2_dx2:", d2_dx2)

    print("All tests passed.")


def test_no_hidden_vec_input():
    from feed_forward_functional import fully_connected as nnu

    NNu = NNu_unpacked(nnu)

    theta = [jnp.ones((3, 1))]
    t = jnp.array([1.5, 2.5, 3.5])
    x = jnp.array([0.5, 1.5, 2.5])
    d2_dx2, d_dt = make_d2_dx2_d_dt(NNu, theta)

    print("\nTesting with matrix input no hidden layers...")
    output = NNu(t, x, theta)
    # since we have three inputs for t and x we want three differen derivative values
    print("NNu:", output)
    assert (output.size == 3)

    partial_t = d_dt(t, x)
    print("d_dt:", partial_t)
    assert (partial_t.size == 3)  # three inputs three derivatives

    partial_2x = d2_dx2(t, x)
    print("d2_dx2:", partial_2x)
    quit()
    assert (partial_2x.size == 3)  # again three inputs -> thre derivatives

    print("All tests passed.")


def test_2_layers_vec_input():
    from feed_forward_functional import fully_connected as nnu

    NNu = NNu_unpacked(nnu)

    d2_dx2, d_dt = make_d2_dx2_d_dt(NNu)
    theta = [jnp.asarray(np.random.normal(size=(3, 2))),
             jnp.asarray(np.random.normal(size=(3, 3)))]

    t = jnp.array([1.5, 2.5, 3.5])
    x = jnp.array([0.5, 1.5, 2.5])

    print("\nTesting with 2 layers...")
    nnu = NNu(t, x, theta)
    d_dt = d_dt(t, x, theta)
    d2_dx2 = d2_dx2(t, x, theta)

    print("NNu:", nnu)
    print("d_dt:", d_dt)
    print("d2_dx2:", d2_dx2)

    print("All tests passed.")


def test_analytical():
    def f(t, x, theta):
        return jnp.exp(-jnp.pi**2 * t)*jnp.sin(jnp.sin(jnp.pi * x))
    d2_dx2, d_dt = make_d2_dx2_d_dt(f)
    theta = [jnp.asarray(np.random.normal(size=(3, 2))),
             jnp.asarray(np.random.normal(size=(3, 3)))]
    t = jnp.array([1.5, 2.5, 3.5])
    x = jnp.array([0.5, 0.9, 0.2])

    d2_dx2, d_dt = make_d2_dx2_d_dt(f, theta)
    t = jnp.array([1.5, 2.5, 3.5])
    x = jnp.array([0.5, 1.5, 2.5])

    print("\nTesting with 2 layers...")
    nnu = f(t, x, theta)
    d_dt = d_dt(t, x, theta)
    d2_dx2 = d2_dx2(t, x, theta)

    print("NNu:", nnu)
    print("d_dt:", d_dt)
    print("d2_dx2:", d2_dx2)

    print("All tests passed.")


if __name__ == "__main__":
    # test_no_hidden_single_input()
    test_no_hidden_vec_input()
    # test_2_layers_vec_input()
    # test_analytical()
