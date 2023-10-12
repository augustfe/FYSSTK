import numpy as np
from Models import Ridge
from heatmap import create_heatmap
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import Lasso
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

            z_tilde = model.predict(X_train)
            z_pred = model.predict(X_test)

            MSETrain[dim, i] = MSE(data.z_train, z_tilde)
            MSETest[dim, i] = MSE(data.z_test, z_pred)
            R2Scores[dim, i] = R2Score(data.z_test, z_pred)
    if title == None:
        title = model.modelName
    create_heatmap(MSETest, lambds, title)


def heatmap_boostrap(data, model=Ridge(), title=None):
    nlambds = lambds.size

    MSETrain = np.zeros((maxDim, nlambds))
    MSETest = np.zeros((maxDim, nlambds))
    R2Scores = np.zeros((maxDim, nlambds))

    scaler = StandardScaler()
    for i, lmbda in enumerate(lambds):
        for dim in range(maxDim):
            X_train = model.create_X(data.x_train, data.y_train, dim)
            X_test = model.create_X(data.x_test, data.y_test, dim)

            scaler.fit(X_train)
            scaler.transform(X_train)
            scaler.transform(X_test)

            model = modelType(lmbda)
            # here we got to boostrap our model
            beta = model.fit(X_train, data.z_train)
            # betas.append(beta)

            z_tilde = model.predict(X_train)
            z_pred = model.predict(X_test)

            MSETrain[dim, i] = MSE(data.z_train, z_tilde)
            MSETest[dim, i] = MSE(data.z_test, z_pred)
            R2Scores[dim, i] = R2Score(data.z_test, z_pred)
    if title == None:
        title = modelType.modelName
    create_heatmap(MSETest, lambds, title)


def heatmap_HomeMade_cross_val():
    return


def heatmap_sklearn_cross_val():
    return
