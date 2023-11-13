import numpy as np
import matplotlib.pyplot as plt


# Define the function
def f(x):
    return x**2


# Define the derivative of the function
def df(x):
    return 2 * x


# Perform gradient descent
def gradient_descent(starting_point, learning_rate, num_iterations):
    points = np.zeros(num_iterations)
    points[0] = starting_point

    for i in range(num_iterations - 1):
        current_point = points[i]
        gradient = df(current_point)
        new_point = current_point - learning_rate * gradient
        points[i + 1] = new_point

    return points


# Set the hyperparameters
learning_rate = 0.1
num_iterations = 100
starting_point = -4

# Perform gradient descent and store the points
points = gradient_descent(starting_point, learning_rate, num_iterations)

# Plot the function and the gradient descent path
x = np.linspace(-5, 5, 100)
y = f(x)

plt.plot(x, y, label="$f(x) = x^2$")
plt.scatter(points, f(points), c="r")
plt.legend(loc="upper left", bbox_to_anchor=(0.07, 0.93))
plt.axis("off")
# plt.title(f"$f(x) = x^2$")
index = np.searchsorted(points, -1e-2, side="right")
print(index)

plt.show()
