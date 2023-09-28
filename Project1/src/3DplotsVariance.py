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





def plot_prediction_vs_true(N: int = 300, polyDegrees: list[int] = [1, 4, 15]) -> None:
    """
    Plots model prediction vs true FrankeFunction using OLS with degrees given
    in Polydegrees. Also does resampling with cross validation to find the
    prediction means and standard deviations for the predictions at eatch degree.
    These valuea are shown in the plot title


    Parameters:
        N (int): a random number, default = 300.
        polyDegrees: degree of polynomials, a list of integer numbers, default = [1, 4, 15].

    """

    # Generate random x and y coordinates for training
    x = np.random.rand(N)
    y = np.random.rand(N)
    z_true = FrankeFunction(x, y)

    z = z_true + z_true.mean() * np.random.randn(N) * 0.30


    # Loop over the different polynomial degrees to fit and plot the models
    for i in range(len(polyDegrees)):

        # need to make new figure each time we plot
        fig = plt.figure(figsize=plt.figaspect(1) * 2)  # make it a bit larger
        ax = fig.add_subplot(projection="3d")
        # idk what this does
        ax.zaxis.set_major_locator(LinearLocator(10))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.02f"))

        # Set axis labels and limits
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_zlim(-0.10, 1.40)

        # Calculate the Franke function for enough points to plot
        X, Y = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
        Z = FrankeFunction(X, Y)

        # Plot the true Franke Function
        ax.plot_surface(
            X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False, alpha=0.5
        )

        #make model
        polynomial_features = PolynomialFeatures(degree=polyDegrees[i], include_bias=False)
        linear_regression = LinearRegression()
        pipeline = Pipeline(
            [
                ("polynomial_features", polynomial_features),
                ("linear_regression", linear_regression),
            ]
        )
        #fit model
        pipeline.fit(np.column_stack((x, y)), z)

        # get cross-val scores
        scores = cross_val_score(
            pipeline, np.column_stack((x, y)), z, scoring="neg_mean_squared_error", cv=10
        )

        # To lazy to make use train_test_split
        # so just make some test data, doesn't need to be random like train data
        X_test, Y_test = np.meshgrid(np.linspace(0, 1, 20), np.linspace(0, 1, 20))
        # predict
        Z_pred = pipeline.predict(np.column_stack((X_test.ravel(), Y_test.ravel())))
        Z_pred = Z_pred.reshape((20, 20))

        # Plot the model
        surf = ax.plot_surface(
            X_test, Y_test, Z_pred, cmap=cm.BrBG, linewidth=0, antialiased=False, alpha=0.5
        )

        # Scatter plot the x,y training values to see what we workin with
        ax.scatter(x, y, z, c="black", s=10)
        #add varaince and mean of model in title of plot
        plt.title(
            "Degree: {}\nMSE = {:.2e}(+/- {:.2e})".format(
                polyDegrees[i], -scores.mean(), scores.std()
            )
        )

        # Add colorbar
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

if __name__ == "__main__":
    np.random.seed(2018)
    plot_prediction_vs_true()
