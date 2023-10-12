import numpy as np
import globals
from HomeCookedModels import Ridge
from heatmap import create_heatmap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from metrics import*
def Ridge_no_resampling(data):
    lambds = globals.lambds
    maxDim = globals.maxDim
    nlambds = lambds.size

    MSETrain = np.zeros((maxDim, nlambds))
    MSETest = np.zeros((maxDim, nlambds))
    R2Scores = np.zeros((maxDim, nlambds))

    scaler = StandardScaler()

    for dim in range(maxDim):
        for i, lmbda in enumerate(lambds):

            X_train = data.create_X(data.x_train, data.y_train, dim)
            X_test = data.create_X(data.x_test, data.y_test, dim)

            scaler.fit(X_train)
            scaler.transform(X_train)
            scaler.transform(X_test)
            model = Ridge(lmbda)
            beta = model.fit(X_train, data.z_train)
            # betas.append(beta)

            z_tilde = model.predict(X_train)
            z_pred = model.predict(X_test)

            MSETrain[dim, i] = MSE(data.z_train, z_tilde)
            MSETest[dim, i] = MSE(data.z_test, z_pred)
            R2Scores[dim, i] = R2Score(data.z_test, z_pred)
    title = "Ridge"
    create_heatmap(MSETest, lambds, title)


def Ridge_bootstrap():
    return

def Ridge_cross_val():
    return
