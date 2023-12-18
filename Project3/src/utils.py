from jax import jit
import jax.numpy as np


@jit
def assign(arr: np.ndarray, idx: int, val: float) -> np.ndarray:
    """Assign a value to an array at a given index.

    Convenience function for setting a value in-place in a JAX array.

    Args:
        arr (np.ndarray): Array to set value to.
        idx (int): Index to set value at.
        val (float): Value to set

    Returns:
        np.ndarray: Updated array.
    """
    arr = arr.at[idx].set(val)
    return arr


@jit
def assign_middle(u_new: np.ndarray, u_mid: np.ndarray) -> np.ndarray:
    """Assign the middle of a JAX array to another JAX array.

    Convenience function for setting the middle values of a JAX array.

    Args:
        u_new (np.ndarray): Array to set value to.
        u_mid (np.ndarray): Array to set value from.

    Returns:
        np.ndarray: Updated array.
    """
    u_new = u_new.at[1:-1].set(u_mid)
    return u_new
