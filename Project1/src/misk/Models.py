import numpy as np
from sklearn.linear_model import Lasso as Lass

class Model:
    def __init__(self):
        self.fitted = False

    def predict(self, X):
        if self.fitted:
            return X @ self.beta
        else:
            raise ValueError("Model can not predict before being fitted")


class Ridge(Model):
    def __init__(self, lambd: int = 1):
        self.modelName = "Ridge"
        self.lambd = lambd

    def fit(self, X, y):
        Identity = np.identity(X.shape[1])
        self.beta = np.linalg.pinv(X.T @ X + self.lambd * Identity) @ X.T @ y
        self.fitted = True
        return self.beta


class OLS(Model):
    def __init__(self):
        self.name = "OLS"
    def fit(self, X, y):
        self.beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.fitted = True
        return self.beta

class Lasso(Model):
    def __init__(self, lambd: float = 1.0):
        self.modelName = "Lasso"
        self.lambd = lambd
        self.lasso_model = Lass(alpha=lambd)

    def fit(self, X, y):
        self.lasso_model.fit(X, y)
        self.beta = np.hstack((self.lasso_model.intercept_, self.lasso_model.coef_))
        self.fitted = True
        return self.beta
