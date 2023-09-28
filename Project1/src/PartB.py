import numpy as np
import matplotlib.pyplot as plt

from Ridge import Ridge
from util import FrankeFunction, create_X, ScaleandCenterData
from sklearn.model_selection import train_test_split



def RidgeofFranke(N: int = 100)-> tuple[list[float], list[float], list[float]]:
    """
    Fits a linear regression model to the Franke function using ridge regression, for each
    polynomial degree less than maxDim. Returns a tuple containing the lists MSEs_train,
    MSEs_test and R2s (R-squared scores)

    Parameters:
    -----------
    N :(int)
        The number of data points to generate. The default value is 100

    Returns:
    --------
        MSE_trains: list[list[float]]
            MSE of training z and prediction for training x,y
            Each inner list consists of MSEs for same degree,
            but different lambdas

        MSE_tests: list[list[float]]
            MSE of test z and prediction for test x,y
            Each inner list consists of MSEs for same degree,
            but different lambdas

        R2s : list[list[float]]
            R-squared score of test z and prediction for test x,y
            Each inner list consists of R2s for same degree,
            but different lambdas

    """
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)
    # x, y = np.meshgrid(x, y)

    z_true = FrankeFunction(x, y)
    # z = z_true + np.random.randn(N * N) * 0.2
    z = z_true + z_true.mean() * np.random.randn(N) * 0.2

    MSE_trains = [[] for _ in lmbdaVals]
    MSE_tests = [[] for _ in lmbdaVals]
    R2s = [[] for _ in lmbdaVals]

    for dim in range(maxDim + 1):
        X = create_X(x, y, dim)
        # X = ScaleandCenterData(X)

        X_train, X_test, y_train, y_test = train_test_split(X, z, random_state=2018)
        X_train_mean = X_train.mean(axis=0)
        X_train_scaled = X_train - X_train_mean
        X_test_scaled = X_test - X_train_mean

        y_train_mean = y_train.mean()
        y_train_scaled = y_train - y_train_mean
        y_test_scaled = y_test - y_train_mean

        for i, lmbda in enumerate(lmbdaVals):
            scores = Ridge(
                X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, lmbda
            )
            MSE_trains[i].append(scores[0])
            MSE_tests[i].append(scores[1])
            R2s[i].append(scores[2])

    return MSE_trains, MSE_tests, R2s

#I think it's best if we only use this function for comparing a cuple of different lambda vals
def plotScores(MSE_train: list[list[float]], MSE_test: list[list[float]], R2: list[list[float]])-> None:
    """
    Plots MSE_train, MSE_test, and R2 values as a function of polynomial
    degree for different lambdas using ridge regression.

    Paramaters:
    -----------
        MSE_train: list[list[float]]
            MSE of training z and prediction for training x,y

        MSE_test: list[list[float]]
            MSE of test z and prediction for test x,y

        R2 : list[list[float]]
            R-squared score of test z and prediction for test x,y
    """
    fig, ax1 = plt.subplots()

    xVals = [i for i in range(maxDim + 1)]

    color = "tab:red"
    ax1.set_xlabel("Polynomial dimension")
    ax1.set_xticks(xVals)
    ax1.set_ylabel("MSE score", color=color)
    for i, lmbda in enumerate(lmbdaVals):
        ax1.plot(xVals, MSE_train[i], label=rf"MSE train $\lambda$={lmbda}")
        ax1.plot(xVals, MSE_test[i], label=rf"MSE test $\lambda$={lmbda}")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.set_ylabel("R2", color=color)
    for i, lmbda in enumerate(lmbdaVals):
        ax2.plot(xVals, R2[i], label=rf"R2 score $\lambda$={lmbda}")
    ax2.tick_params(axis="y", labelcolor=color)

    fig.legend()
    fig.suptitle("Scores by polynomial degree for Ridge")
    fig.tight_layout()

def ridge_vs_OLS():
    """
    fucntion for comparing the analasys in A to that of here. Again
    I think it's best to only do a cuple of different lambda vals.

    """
    return

def plot_3D_lambda_vs_polydegree(MSE_test: list[list[float]], lambdaVals: list[float]):
    """Plots a 3D surface plot of the test MSE as a function of polynomial degree and lambda values.

    Args:
        MSE_test (list[list[float]]): The test mean square error for different lambda values and polynomial degrees.
        lambdaVals (list[float]): The lambda values used for regularization.

    Returns:
        None: Displays the 3D surface plot of the test mean square error.
    """

      # Define polynomial degrees and lambda values
    degrees = np.arange(1, len(MSE_test[0]) + 1)

    # Initialize arrays to store MSE values
    mse_train = np.array(MSE_test)

    fig = plt.figure(figsize=plt.figaspect(1) * 2)  # make it a bit larger
    ax = fig.add_subplot(projection='3d')

    X, Y = np.meshgrid(degrees, lambdaVals)
    ax.plot_surface(X, np.log10(Y), mse_train, cmap=plt.cm.coolwarm, rstride=1, cstride=1)

    # Set labels and title
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('log10($\lambda$)')
    ax.set_zlabel('Test MSE')
    ax.set_title('Test MSE as a function of complexity and $\lambda$')

    plt.show()






if __name__ == "__main__":
    np.random.seed(2018)

    n_lmbdas = 16
    lmbdaVals = np.logspace(-3, 5, n_lmbdas)
    maxDim = 15

    scores = RidgeofFranke()
    #plotScores(*scores)
    MSE_test = scores[1]
    plot_3D_lambda_vs_polydegree(MSE_test, lmbdaVals)
    #plt.show()
