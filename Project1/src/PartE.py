import numpy as np
import matplotlib.pyplot as plt

from util import FrankeFunction, create_X, MSE
from sklearn.model_selection import train_test_split
from OLS import OLS, create_OLS_beta
from PartA import OLSofFranke
from sklearn.pipeline import make_pipeline
from sklearn.utils import resample


def hastieFig():
    """plots figure similar to fig 2.11 in Hastie, Tibshirani, and Friedman.
        all the same as PartA, but without the R2 value

        note to teammates: graph kinda ugly sometimes
    """

    #if N is too large MSE_train won't diverge
    scores = OLSofFranke(N = 30)

    MSE_train = scores[0]
    MSE_test = scores[1]
    fig, ax1 = plt.subplots()

    xVals = [i for i in range(len(MSE_train))]

    color = "tab:red"
    ax1.set_xlabel("Polynomial dimension")
    ax1.set_xticks(xVals)
    ax1.set_ylabel("MSE score", color=color)
    ax1.plot(xVals, MSE_train, label="MSE train", color="r")
    ax1.plot(xVals, MSE_test, label="MSE test", color="g")
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()

    color = "tab:blue"
    ax2.tick_params(axis="y", labelcolor=color)

    fig.legend()
    fig.suptitle("MSE by polynomial degree for OLS")
    fig.tight_layout()

    plt.show()

def bootstrapp(n_bstraps, X_train, z_train, polydegrees):

    n_degrees = len(polydegrees)

    error = np.zeros(n_degrees)
    bias = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)

    for degree in polydegrees:
            for i in range(n_bstraps):
                x_, z_ = resample(X_train, z_train)
                beta_hat = create_OLS_beta(x_, z_)
                z_pred[:, i] = (X_test @ beta_hat).ravel()


            error[degree] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
            bias[degree] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
            variance[degree] = np.mean( np.var(z_pred, axis=1, keepdims=True))


def variance(z_pred: np.ndarray):
    """ Retruns mean of variance array which is
        equal to the variance if we assume that our
        vals are unifromly distributed
    """
    return np.mean(np.var(z_pred, axis=1, keepdims=True))

def bias(z_test, z_pred):
    z_pred_mean = np.mean(z_pred, axis=1, keepdims=True)
    return np.mean( (z_test - z_pred_mean)**2 )

def error(z_test, z_pred):
    return np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )




def plot_Bias_VS_Varaince():
    """
    Plots variance, error(MSE), and bias using bootstrapping as resampling technique.

    Parameters:
    -----------
    data : ndarray
        A one-dimensional numpy array containing the data set to be analyzed.

    num_samples : int
        Optional, default is 1000. Number of random samples to generate using bootstrapping.


    Plots:
    ------
    Variance of y_tilde
    Mean squared error (MSE) of the estimate
    Bias of the estimate

    """
    N = 40
    n_boostraps = 100
    maxdegree = 14


    # Make data set.
    x = np.sort(np.random.uniform(0, 1, N))
    y = np.sort(np.random.uniform(0, 1, N))

    z_true = FrankeFunction(x, y)
    # z = z_true + np.random.randn(N * N) * 0.2
    z = z_true + z_true.mean() * np.random.randn(N) * 0.2


    error = np.zeros(maxdegree)
    bias = np.zeros(maxdegree)
    variance = np.zeros(maxdegree)
    polydegree = np.zeros(maxdegree)

    for degree in range(maxdegree):
        X = create_X(x, y, dim)
        # X = ScaleandCenterData(X)

        X_train, X_test, z_train, z_test = train_test_split(X, z)


        z_pred = np.empty((z_test.shape[0], n_boostraps))

        for i in range(n_boostraps):
            x_, z_ = resample(X_train, z_train)
            beta_hat = create_OLS_beta(x_, z_)
            z_pred[:, i] = (X_test @ beta_hat).ravel()


        polydegree[degree] = degree
        error[degree] = np.mean( np.mean((z_test - z_pred)**2, axis=1, keepdims=True) )
        bias[degree] = np.mean( (z_test - np.mean(z_pred, axis=1, keepdims=True))**2 )
        variance[degree] = np.mean( np.var(z_pred, axis=1, keepdims=True) )

    plt.plot(polydegree, error, label='Error')
    plt.plot(polydegree, bias, label='bias')
    plt.plot(polydegree, variance, label='Variance')
    plt.legend()
    plt.show()


"""
ok, what bits of code is it that we are reusing?

we are creating the x, y and corresponding z vals in A, B, C, E

we'll be doing some resampling stuff for partE and partF

boostrapp be a func that straps for single degree? or should it do the boostrapp
for

"""


if __name__ == "__main__":
    hastieFig()
