import numpy as np
from Models import Ridge
from heatmap import create_heatmap
import matplotlib.pyplot as plt
import numpy as np

from resampling import bootstrap_lambdas, sklearn_cross_val_lambdas, HomeMade_cross_val_lambdas
from sklearn.linear_model import Lasso as LassoSKL
from sklearn.linear_model import Ridge as RidgeSKL
from sklearn.linear_model import LinearRegression as OLSSKL
from sklearn.preprocessing import StandardScaler
from metrics import *
from globals import *


def heatmap_no_resampling(data, model=Ridge(), title=None):
    nlambds = lambds.size

    MSETrain = np.zeros((maxDim, nlambds))
    MSETest = np.zeros((maxDim, nlambds))
    R2Scores = np.zeros((maxDim, nlambds))

    scaler = StandardScaler()

    for dim in range(maxDim):
        for i, lmbda in enumerate(lambds):

            X_train = model.create_X(data.x_train, data.y_train, dim)
            X_test = model.create_X(data.x_test, data.y_test, dim)

            scaler.fit(X_train)
            scaler.transform(X_train)
            scaler.transform(X_test)

            beta = model.fit(X_train, data.z_train, lmbda)
            # betas.append(beta)

            #z_tilde = model.predict(X_train)
            z_pred = model.predict(X_test)

            #MSETrain[dim, i] = MSE(data.z_train, z_tilde)
            MSETest[dim, i] = MSE(data.z_test, z_pred)
            R2Scores[dim, i] = R2Score(data.z_test, z_pred)
    if title == None:
        title = model.modelName
    create_heatmap(MSETest, lambds, title)


def heatmap_bootstrap(data, model=Ridge(), var = False, title=None):
    n_boostraps = 100
    error, bias, variance = bootstrap_lambdas(data, n_boostraps, Ridge())
    if title == None:
        title = model.modelName
    if var:
        create_heatmap(variance, lambds, title)
    else:
        create_heatmap(error, lambds, title)


def heatmap_HomeMade_cross_val(data, model = Ridge(), var = False, title=None):
    error, variance = HomeMade_cross_val_lambdas(data, kfolds = 5, model=model)
    if title == None:
        title = model.modelName
    if var:
        create_heatmap(variance, lambds, title)
    else:
        create_heatmap(error, lambds, title)


def heatmap_sklearn_cross_val(data, model = RidgeSKL(), var = False, title=None):
    error, variance = sklearn_cross_val_lambdas(data, kfolds=5, model=model)
    if title == None:
        title = model.__class__.__name__
    if var:
        create_heatmap(variance, lambds, title)
    else:
        create_heatmap(error, lambds, title)
