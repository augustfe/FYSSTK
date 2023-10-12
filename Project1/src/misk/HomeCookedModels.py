import numpy as np

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
        self.lambd = lambd

    def fit(self, X, y):
        Identity = np.identity(X.shape[1])
        self.beta = np.linalg.pinv(X.T @ X + self.lambd * Identity) @ X.T @ y
        self.fitted = True
        return self.beta

class OLS(Model):
    def fit(self, X, y):
        self.beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.fitted = True
        return self.beta
