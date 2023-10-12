# here we can write our funcs such that we take a sklear model as input
from Models import OLS, Ridge, Lasso
import numpy as np
from metrics import *
from sklearn.utils import resample
from globals import *


def bootstrap_polydegrees(data, polyDegrees, n_boostraps, model=OLS()):
    """
    Performs boostrap on different polydegrees
    """
    n_degrees = len(polyDegrees)

    error = np.zeros(n_degrees)
    bias = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)

    for j, dim in enumerate(polyDegrees):
        X_train = model.create_X(data.x_train, data.y_train, dim)
        X_test = model.create_X(data.x_test, data.y_test, dim)

        z_test, z_train = data.z_test, data.z_train
        z_pred = np.empty((z_test.shape[0], n_boostraps))

        for i in range(n_boostraps):
            X_, z_ = resample(X_train, z_train)
            model.fit(X_, z_)
            z_pred[:, i] = model.predict(X_test).ravel()

        error[j] = mean_MSE(z_test, z_pred)
        bias[j] = get_bias(z_test, z_pred)
        variance[j] = get_variance(z_pred)

    return error, bias, variance


def boostrap_lambdas(data, model=Ridge()):
    """
    performs boostrap on polydegrees and hyperparamater lambds
    for regularized regression
    """
    n_degrees = len(polyDegrees)
    n_lambds = lambds.size

    error = np.zeros((n_degrees, n_lambds))
    bias = np.zeros((n_degrees, n_lambds))
    variance = np.zeros((n_degrees, n_lambds))

    for i, dim in enumerate(polyDegrees):
        X_train = model.create_X(data.x_train, data.y_train, dim)
        X_test = model.create_X(data.x_test, data.y_test, dim)

        z_test, z_train = data.z_test, data.z_train

        z_test = z_test.reshape(z_test.shape[0], 1)
        for i, lambd in enumerate(lambds):

            z_pred = np.empty((z_test.shape[0], n_boostraps))
            for k in range(n_boostraps):
                X_, z_ = resample(X_train, z_train)
                model.fit(X_, z_, lambd)
                z_pred[:, k] = model.predict(X_test).ravel()

            error[i, j] = mean_MSE(z_test, z_pred)
            bias[i, j] = get_bias(z_test, z_pred)
            variance[i, j] = get_variance(z_pred)

    return error, bias, variance


def sklearn_cross_val(data, polyDegrees: list[int], nfolds, model=OLS()):
    """
    sklearn cross val for on polynomial degrees
    """
    n_degrees = len(polyDegrees)

    error = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)

    for i, degree in enumerate(polyDegrees):
        X = model.create_X(data.x, data.y, degree)

        scores = cross_val_score(
            model, X, data.z, scoring="neg_mean_squared_error", cv=nfolds
        )
        error[i] = -scores.mean()
        variance[i] = scores.std()

    return error, variance


def kfold_score_degrees(data, polyDegrees: list[int], kfolds: int, model=OLS()):
    """
    HomeCooked cross-val using Kfold. Only for polynomial degrees.
    """
    n_degrees = len(polyDegrees)

    error = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)

    Kfold = KFold(n_splits=kfolds)

    for i, degree in enumerate(polyDegrees):
        scores = np.zeros(kfolds)

        X = model.create_X(data.x_, data.y_, degree)

        for j, (train_i, test_i) in enumerate(Kfold.split(X)):

            X_train = X[train_i]
            X_test = X[test_i]
            z_test = data.z_[test_i]
            z_train = data.z_[train_i]

            model.fit(X_train, z_train)

            z_pred = model.predict(X_test)

            scores[j] = MSE(z_pred, z_test)

        error[i] = scores.mean()
        variance[i] = scores.std()
    return error, variance


def sklearn_cross_val_lambdas(data, polyDegrees: list[int], nfolds, model=Ridge()):
    """
    sklearn cross val for polydegrees and lambda
    """
    n_degrees = len(polyDegrees)
    n_lambds = len(lambds)
    error = np.zeros((n_degrees, n_lambds))
    variance = np.zeros((n_degrees, n_lambds))

    for i, degree in enumerate(polyDegrees):
        X = model.create_X(data.x_, data.y_, degree)
        for j, lambd in enumerate(lambds):
            model = modelType(lambd)
            scores = cross_val_score(
                model, X, data.z_, scoring="neg_mean_squared_error", cv=nfolds
            )
            error[i, j] = -scores.mean()
            variance[i, j] = scores.std()

    return error, variance


def kfold_score_degrees(data, polyDegrees: list[int], kfolds: int, modelType=Ridge):
    """
    HomeCooked cross-val using Kfold. for polydegrees and lambda.
    """
    n_degrees = len(polyDegrees)
    n_lambds = lambds.size

    error = np.zeros((n_degrees, n_lambds))
    variance = np.zeros((n_degrees, n_lambds))

    Kfold = KFold(n_splits=kfolds)

    for i, degree in enumerate(polyDegrees):
        scores = np.zeros(kfolds)

        X = model.create_X(data.x_, data.y_, degree)

        for j, lambd in enumerate(lambds):
            model = modelType(lambd)
            for k, (train_i, test_i) in enumerate(Kfold.split(X)):

                X_train = X[train_i]
                X_test = X[test_i]
                z_test = data.z_[test_i]
                z_train = data.z_[train_i]

                model.fit(X_train, z_train)

                z_pred = model.predict(X_test)

                scores[k] = MSE(z_pred, z_test)

            error[i, j] = scores.mean()
            variance[i, j] = scores.std()
    return error, variance
