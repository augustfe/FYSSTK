import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from util import MSE


def pureSKLearnLasso(
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.array,
    y_test: np.array,
    alpha: float = 0.1,
) -> tuple[float, float]:
    """Using scikit to solve with Lasso.

    inputs:
        X_train: Training values
        X_test: Testing data
        y_train: Training values
        y_test: Testing data
        alpha (float): Regularization strength
    returns:
        Mean squared error of trained and predicted.
    """
    # Scale the data
    scaler = StandardScaler().fit(X_train)

    scaled_X_train = scaler.transform(X_train)
    scaled_X_test = scaler.transform(X_test)

    # Initialize and fit model
    clf = Lasso(alpha=alpha)
    clf.fit(scaled_X_train, y_train)

    ztilde = clf.predict(scaled_X_train)
    z_pred_Lasso = clf.predict(scaled_X_test)

    MSE_train = MSE(y_train, ztilde)
    MSE_test = MSE(y_test, z_pred_Lasso)

    return MSE_train, MSE_test
