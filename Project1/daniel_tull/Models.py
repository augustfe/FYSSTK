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

    @staticmethod
    def create_X(x: np.array, y: np.array, n: int) -> np.ndarray:
        "Create design matrix"
        if len(x.shape) > 1:
            x = np.ravel(x)
            y = np.ravel(y)

        N = len(x)
        lgth = (n + 1) * (n + 2) // 2
        X = np.ones((N, lgth))
        for i in range(1, n + 1):
            q = i * (i + 1) // 2
            for k in range(i + 1):
                X[:, q + k] = (x ** (i - k)) * (y**k)

        return X


class Ridge(Model):
    def __init__(self, lambd=None):
        super().__init__()
        self.modelName = "Ridge"
        self.lambd = lambd

    def fit(self, X, y, lambd: int = None):
        if lambd != None:
            self.lambd = lambd
        if self.lambd == None:
            raise ValueError("No lambda provided")
        Identity = np.identity(X.shape[1])
        self.beta = np.linalg.pinv(X.T @ X + self.lambd * Identity) @ X.T @ y
        self.fitted = True
        return self.beta


class OLS(Model):
    def __init__(self):
        super().__init__()
        self.name = "OLS"

    def fit(self, X, y):
        self.beta = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.fitted = True
        return self.beta


class Lasso(Model):
    def __init__(self, lambd=None):
        super().__init__()
        self.modelName = "Lasso"
        self.lambd = lambd
        self.lasso_model = None

    def fit(self, X, y, lambd=None):
        if lambd != None:
            self.lambd = lambd
        if self.lambd == None:
            raise ValueError("No lambda/alpha provided")

        self.lambd = lambd
        self.lasso_model = Lass(alpha=lambd, max_iter=1000000)
        self.lasso_model.fit(X, y)
        self.beta = np.hstack((self.lasso_model.intercept_, self.lasso_model.coef_))
        self.fitted = True
        return self.beta

    def predict(self, X):
        if self.fitted:
            return self.lasso_model.predict(X)
        else:
            raise ValueError("Model can not predict before being fitted")
