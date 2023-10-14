from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np
from random import random, seed

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from util import FrankeFunction


def plot_prediction_vs_true(N: int = 400, polyDegrees: list[int] = [1, 4, 15]) -> None:
    """
    Plots model prediction vs true FrankeFunction using OLS with degrees given
    in polyDegrees. Also does resampling with cross validation to find the
    prediction means and standard deviations for the predictions at each degree.
    These values are shown in the plot title.


    Parameters:
        N (int): a random number, default = 300.
        polyDegrees: degree of polynomials, a list of integer numbers, default = [1, 4, 15].
    """

    # Generate random x and y coordinates for training
    x = np.random.rand(N)
    y = np.random.rand(N)
    z_true = FrankeFunction(x, y)

    z = z_true + z_true.mean() * np.random.randn(N) * 0.30

    # Create the figure and subplots
    fig, axs = plt.subplots(1, len(polyDegrees), figsize=(15, 5))

    # Loop over the different polynomial degrees to fit and plot the models
    for i in range(len(polyDegrees)):
        ax = axs[i]
        ax.axis('off')
        ax = fig.add_subplot(1, len(polyDegrees), i + 1, projection="3d")
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

        # Remove x, y, and z axis labels and units
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_zlabel('')

        # Set axis limits
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_zlim(-0.10, 1.40)

        # Calculate the Franke function for enough points to plot
        X, Y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
        Z = FrankeFunction(X, Y)

        # Plot the true Franke Function
        ax.plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5
        )

        # Make the polynomial regression model
        polynomial_features = PolynomialFeatures(
            degree=polyDegrees[i], include_bias=False
        )
        linear_regression = LinearRegression()
        pipeline = Pipeline(
            [
                ("polynomial_features", polynomial_features),
                ("linear_regression", linear_regression),
            ]
        )

        # Fit the model
        pipeline.fit(np.column_stack((x, y)), z)

        # Perform cross-validation and get scores
        scores = cross_val_score(
            pipeline,
            np.column_stack((x, y)),
            z,
            scoring="neg_mean_squared_error",
            cv=10,
        )

        # Create test data coordinates for predictions
        X_test, Y_test = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
        xy_test = np.column_stack((X_test.ravel(), Y_test.ravel()))

        # Predict Z values for the test coordinates
        Z_pred = pipeline.predict(xy_test)
        Z_pred = Z_pred.reshape((20, 20))

        # Plot the model predictions
        surf = ax.plot_surface(
            X_test,
            Y_test,
            Z_pred,
            cmap=cm.BrBG,
            linewidth=0,
            antialiased=False,
            alpha=0.5,
        )

        # Scatter plot the x, y training values
        ax.scatter(x, y, z, c="black", s=10)

        # Add the MSE and standard deviation to the plot title

        ax.set_title(f"Degree: {polyDegrees[i]}\nMSE = {scores.mean():.2e}(+/- {scores.std():.2e})")


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    np.random.seed(2018)
    plot_prediction_vs_true()
