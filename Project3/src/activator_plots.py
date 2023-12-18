from jax.nn import relu, sigmoid, tanh, leaky_relu, swish, elu
from plotutils import setup_axis
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

path = Path(__file__).parent.parent / "figures/activators"
path.mkdir(exist_ok=True, parents=True)

functions = [relu, sigmoid, tanh, swish, elu]
xlims = [(-1, 1), (-5, 5), (-5, 5), (-5, 5), (-5, 5)]
func_names = ["ReLU", "Sigmoid", "Tanh", "Swish", "ELU"]

for i, (func, xlim, func_name) in enumerate(zip(functions, xlims, func_names)):
    x = np.linspace(xlim[0], xlim[1], 100)
    y = func(x)
    ylim = [np.min(y) * 1.1, np.max(y) * 1.1]

    ax = setup_axis(xlim=xlim, ylim=ylim)

    ax.plot(x, y, color=f"C{i}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$", rotation=0)
    ax.set_title(func_name)

    plt.savefig(path / f"{func.__name__}.pdf", bbox_inches="tight")
    plt.clf()

x = np.linspace(-1, 1, 100)
y = leaky_relu(x, negative_slope=0.1)
ylim = [np.min(y) * 1.1, np.max(y) * 1.1]

ax = setup_axis(xlim=[-1, 1], ylim=ylim)

ax.plot(x, y, color=f"C{i + 1}")
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$", rotation=0)
ax.set_title("Leaky ReLU")

plt.savefig(path / f"leaky_relu.pdf", bbox_inches="tight")
plt.clf()

ax = setup_axis(xlim=[-2, 2], ylim=[-1, 2])
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$", rotation=0)
x = np.linspace(-2, 2, 100)
for func, xlim, func_name in zip(functions, xlims, func_names):
    y = func(x)

    ax.plot(x, y, label=func_name)

ax.plot(x, leaky_relu(x, negative_slope=0.1), label="Leaky ReLU")

ax.set_title("Activation functions")
ax.legend()
plt.savefig(path / "activators.pdf", bbox_inches="tight")
plt.clf()
