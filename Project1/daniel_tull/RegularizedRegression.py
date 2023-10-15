import numpy as np
from Models import Ridge
from heatmap import create_heatmap
import matplotlib.pyplot as plt
import numpy as np

from resampling import (
    bootstrap_lambdas,
    sklearn_cross_val_lambdas,
    HomeMade_cross_val_lambdas,
)
from sklearn.linear_model import Lasso as SKLasso
from sklearn.linear_model import Ridge as SKRidge
from sklearn.linear_model import LinearRegression as OLSSKL
from sklearn.preprocessing import StandardScaler
from metrics import *
from globals import *
from tqdm import tqdm


def heatmap_no_resampling(data, model, polyDegrees, lambdas, **kwargs):
    ndegrees = len(polyDegrees)

    nlambdas = lambdas.size

    MSETrain = np.zeros((ndegrees, nlambdas))
    MSETest = np.zeros((ndegrees, nlambdas))
    R2Scores = np.zeros((ndegrees, nlambdas))

    scaler = StandardScaler()

    pbar = tqdm(
        total=ndegrees * nlambdas, desc=f"No resampling {model.__class__.__name__}"
    )

    for j, dim in enumerate(polyDegrees):
        for i, lmbda in enumerate(lambdas):
            X_train = model.create_X(data.x_train, data.y_train, dim)
            X_test = model.create_X(data.x_test, data.y_test, dim)

            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            beta = model.fit(X_train, data.z_train, lmbda)
            # betas.append(beta)

            # z_tilde = model.predict(X_train)
            z_pred = model.predict(X_test)

            # MSETrain[dim, i] = MSE(data.z_train, z_tilde)
            MSETest[j, i] = MSE(data.z_test, z_pred)
            # R2Scores[j, i] = R2Score(data.z_test, z_pred)
            pbar.update(1)
    create_heatmap(MSETest, polyDegrees=polyDegrees, lambdas=lambdas, **kwargs)


def heatmap_bootstrap(
    data, polyDegrees, lambdas, n_bootstraps, model, var=False, **kwargs
):
    error, bias, variance = bootstrap_lambdas(
        data, polyDegrees=polyDegrees, lambdas=lambdas, n_bootstraps=100, model=model
    )
    if var:
        create_heatmap(variance, lambdas=lambdas, polyDegrees=polyDegrees, **kwargs)
    else:
        create_heatmap(error, lambdas=lambdas, polyDegrees=polyDegrees, **kwargs)


def heatmap_HomeMade_cross_val(
    data,
    polyDegrees,
    lambdas,
    kfolds,
    model,
    var=False,
    **kwargs,
):
    error, variance = HomeMade_cross_val_lambdas(
        data, kfolds=kfolds, polyDegrees=polyDegrees, lambdas=lambdas, model=model
    )
    if var:
        create_heatmap(variance, lambdas=lambdas, polyDegrees=polyDegrees, **kwargs)
    else:
        create_heatmap(error, lambdas=lambdas, polyDegrees=polyDegrees, **kwargs)


def heatmap_sklearn_cross_val(
    data,
    polyDegrees,
    lambdas,
    kfolds,
    model,
    var=False,
    **kwargs,
):
    error, variance = sklearn_cross_val_lambdas(
        data, kfolds=kfolds, polyDegrees=polyDegrees, lambdas=lambdas, model=model
    )

    if var:
        create_heatmap(variance, lambdas=lambdas, polyDegrees=polyDegrees, **kwargs)
    else:
        create_heatmap(error, lambdas=lambdas, polyDegrees=polyDegrees, **kwargs)
