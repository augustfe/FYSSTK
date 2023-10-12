#here we can write our funcs such that we take a sklear model as input
from HomeCookedModels import OLS
import numpy as np
from metrics import*
from sklearn.utils import resample


def bootstrap(data, polyDegrees, n_boostraps, model = OLS()):
    n_degrees = len(polyDegrees)

    error = np.zeros(n_degrees)
    bias = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)

    for j, dim in enumerate(polyDegrees):
        X_train = data.create_X(data.x_train, data.y_train, dim)
        X_test = data.create_X(data.x_test, data.y_test, dim)

        z_test, z_train = data.z_test, data.z_train

        z_test = z_test.reshape(z_test.shape[0], 1)
        z_pred = np.empty((z_test.shape[0], n_boostraps))

        for i in range(n_boostraps):
            X_, z_ = resample(X_train, z_train)
            model.fit(X_,z_)
            z_pred[:, i] = model.predict(X_test).ravel()

        error[j] = mean_MSE(z_test, z_pred)
        bias[j] = get_bias(z_test, z_pred)
        variance[j] = get_variance(z_pred)

        return error, bias, variance

def sklearn_cross_val(x, y, z, polyDegrees: list[int], nfolds, model = OLS()):
    n_degrees = len(polyDegrees)

    error = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)

    for i,degree in enumerate(polyDegrees):
        X = create_X(x, y, degree)

        model.fit(X, z)

        scores = cross_val_score(model, X, z,
                                 scoring="neg_mean_squared_error", cv=nfolds)
        error[i] = -scores.mean()
        variance[i] = scores.std()

    return error, variance

def kfold_score_degrees(data, polyDegrees: list[int], kfolds: int, model = OLS()):
    n_degrees = len(polyDegrees)

    error = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)

    Kfold = KFold(n_splits = kfolds)

    for i,degree in enumerate(polyDegrees):
        scores = np.zeros(kfolds)

        X = data.create_X(data.x, data.y, degree)

        for j, (train_i, test_i) in enumerate(Kfold.split(X)):

            X_train = X[train_i]; X_test = X[test_i]
            z_test = z[test_i]; z_train = z[train_i]


            model.fit(X_train, z_train)

            z_pred = model.predict(X_test)

            scores[j] = MSE(z_pred, z_test)

        error[i] = scores.mean()
        variance[i] = scores.std()

    return error, variance