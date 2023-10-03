from PartE import bootstrap
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import cross_val_score, KFold
import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from Ridge import create_ridge_beta
from OLS import create_OLS_beta
from util import create_X, MSE, FrankeFunction
from sklearn.model_selection import train_test_split


def sklearn_cross_val_OLS(x, y, z, polyDegrees: list[int], nfolds):
    n_degrees = len(polyDegrees)

    error = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)

    for i,degree in enumerate(polyDegrees):
        X = create_X(x, y, degree)

        linear_regression = LinearRegression()

        linear_regression.fit(X, z)

        scores = cross_val_score(linear_regression, X, z,
                                 scoring="neg_mean_squared_error", cv=nfolds)
        error[i] = -scores.mean()
        variance[i] = scores.std()

    return error, variance

def sklearn_cross_val_lambdas():
    return

def sklearn_cross_val():
    return



def kfold_score_degrees(x, y, z, polyDegrees: list[int], kfolds: int, beta_func: callable = create_OLS_beta):
    n_degrees = len(polyDegrees)

    error = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)

    Kfold = KFold(n_splits = kfolds)

    for i,degree in enumerate(polyDegrees):
        scores = np.zeros(kfolds)

        X = create_X(x, y, degree)

        for j, (train_i, test_i) in enumerate(Kfold.split(X)):

            X_train = X[train_i]; X_test = X[test_i]
            z_test = z[test_i]; z_train = z[train_i]

            linear_regression = LinearRegression()

            linear_regression.fit(X_train, z_train)

            z_pred = linear_regression.predict(X_test)

            scores[j] = MSE(z_pred, z_test)

        error[i] = scores.mean()
        variance[i] = scores.std()

    return error, variance

def kfold_score_lambdas():
    return


def bootstrap_vs_cross_val_OLS():
    """
    Here we wish to compare our bootstrap to cross val.
    guessing that plotting the variance, bias and errors in the same plot
    is fine
    """
    N = 500
    n_boostraps = 500
    maxdegree = 13

    kfolds = 7
    # Make data set.
    x = np.random.uniform(0, 1, N)
    y = np.random.uniform(0, 1, N)

    z_true = FrankeFunction(x, y)
    # z = z_true + np.random.randn(N * N) * 0.2
    z = z_true + z_true.mean() * np.random.randn(N) * 0.20

    polyDegrees = list(range(maxdegree))

    #error_boot, bias_boot, variance_boot = bootstrap(x, y, z, polyDegrees, n_boostraps)
    error_CV, varaince_CV = sklearn_cross_val_OLS(x, y, z, polyDegrees, kfolds)
    error_kfold, variance_kfold = kfold_score_degrees(x, y, z, polyDegrees, kfolds)

    #plt.plot(polyDegrees, error_boot, label="Boostrap Error")
    #plt.plot(polyDegrees, variance_boot, label="Boostrap Variance")
    plt.plot(polyDegrees, error_kfold,'b' ,label="Kfold Error")
    #plt.plot(polyDegrees, variance_kfold, label="Kfold variance")
    plt.plot(polyDegrees, error_CV, 'r--' ,label="cross val Error")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    np.random.seed(0)
    bootstrap_vs_cross_val_OLS()
