import create_heatmap from heatmap
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

def Lasso(data, nLambdas: int):
    methodname = "Lasso"
    minLmbda = -3
    maxLmbda = 5
    lmbdas = np.logspace(minLmbda, maxLmbda, nLambdas)
    MSETrain = np.zeros((data.maxDim, nLambdas))
    MSETest = np.zeros((data.maxDim, nLambdas))
    R2Scores = np.zeros((data.maxDim, nLambdas))

    scaler = StandardScaler()

    for dim in range(data.maxDim):
        for i, lmbda in enumerate(lmbdas):
            X_train = data.create_X(data.x_train, data.y_train, dim)
            X_test = data.create_X(data.x_test, data.y_test, dim)

            scaler.fit(X_train)
            scaler.transform(X_train)
            scaler.transform(X_test)

            clf = Lasso(alpha=lmbda)
            clf.fit(X_train, data.y_train)

            z_tilde = clf.predict(X_train)
            z_pred = clf.predict(X_test)
            # betas.append(beta)

            MSETrain[dim, i] = data.MSE(data.z_train, z_tilde)
            MSETest[dim, i] = data.MSE(data.z_test, z_pred)
            R2Scores[dim, i] = data.R2Score(data.z_test, z_pred)
    title = "sa"
    data.lambda_heat_map(MSETest, lmbdas, title)
