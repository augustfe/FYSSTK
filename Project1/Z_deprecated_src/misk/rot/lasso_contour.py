import numpy as np
import matplotlib.pyplot as plt

# plt.rcParams['text.usetex'] = True

# Create a grid of beta1 and beta2 values
beta1 = np.linspace(-2, 2, 400)
beta2 = np.linspace(-2, 2, 400)
Beta1, Beta2 = np.meshgrid(beta1, beta2)

# Create the first subplot
plt.figure(figsize=(12, 5))


# Subplot 1
plt.subplot(1, 2, 1)
plt.contour(Beta1, Beta2, Beta1**2 + Beta2**2, levels=[1], colors="k")
plt.imshow(
    np.abs(Beta1) + np.abs(Beta2) < 1,
    extent=[-2, 2, -2, 2],
    origin="lower",
    cmap="Blues",
    alpha=0.5,
)
plt.title("|beta1| + |beta2| < 1")
plt.xlabel("beta1")
plt.ylabel("beta2")
plt.axis("off")

# Add crosshair axes
plt.axhline(y=0, color="gray", linestyle="-")
plt.axvline(x=0, color="gray", linestyle="-")

# Subplot 2
plt.subplot(1, 2, 2)
plt.axis("off")
plt.imshow(
    Beta1**2 + Beta2**2 < 1,
    extent=[-2, 2, -2, 2],
    origin="lower",
    cmap="Blues",
    alpha=0.5,
)


plt.title("beta^2 + beta^2", fontsize=12)
plt.xlabel("beta", fontsize=12)
plt.ylabel("beta", fontsize=12)


# Add contour line at beta2=1, beta1=0
plt.axhline(y=0, color="gray", linestyle="-")
plt.axvline(x=0, color="gray", linestyle="-")

plt.tight_layout()
plt.show()
