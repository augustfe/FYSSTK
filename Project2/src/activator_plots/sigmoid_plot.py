import numpy as np
from plotutils import setup_axis
from Activators import sigmoid
import matplotlib.pyplot as plt
from pathlib import Path

path = Path(__file__).parent.parent.parent / "figures/activators"

ax = setup_axis(xlim=[-4, 4], ylim=[-0.1, 1.1])

x = np.linspace(-4, 4, 100)
y = sigmoid(x)

ax.plot(x, y)
ax.set_xlabel(r"$x$")
ax.set_ylabel(r"$y$", rotation=0)
ax.set_title("Sigmoid Activation Function")

plt.savefig(path / "sigmoid.pdf", bbox_inches="tight")
