import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from util import MSE, R2Score


def pureSKLearnLasso(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.array,
    y_test: np.array,
    alpha: float = 0.1,
    scale: bool = False,
) -> tuple[float, float, float]:
    """Using scikit to solve with Lasso.

    inputs:
        X_train: Training values
        X_test: Testing data
        y_train: Training values
        y_test: Testing data
        alpha (float): Regularization strength
        scale (bool): whether to scale data in func
    returns:
        Mean squared error of trained and predicted.
    """
    # Scale the data
    if scale:
        scaler = StandardScaler().fit(X_train)

        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

    # Initialize and fit model
    clf = Lasso(alpha=alpha)
    clf.fit(X_train, y_train)

    ztilde = clf.predict(X_train)
    z_pred_Lasso = clf.predict(X_test)

    MSE_test = MSE(y_train, ztilde)
    R2 = R2Score(y_test, z_pred_Lasso)

    return MSE_train, MSE_test, R2
