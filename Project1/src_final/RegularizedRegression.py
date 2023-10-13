import numpy as np
from Models import Ridge, Model
from Data import Data
from heatmap import create_heatmap

from resampling import (
    bootstrap_lambdas,
    sklearn_cross_val_lambdas,
    HomeMade_cross_val_lambdas,
)
import sklearn.linear_model as sklm
from sklearn.preprocessing import StandardScaler
from metrics import MSE, R2Score
from tqdm import tqdm


def heatmap_no_resampling(
    data: Data,
    maxDim: int = 15,
    lmbds: np.array = np.logspace(-3, 5, 10),
    model: Model = Ridge(),
    title: str = None,
    **kwargs,
) -> None:
    """Heatmap of lambda vs dimension with no resampling.

    inputs:
        data (Data): Dataset
        maxDim (int): Maximal polynomial degree
        lmbds (np.array): Regularization weights
        model (Model): Regression model to apply
        title (str): Title of the resulting plot
    """
    nlambds = lmbds.size

    # MSETrain = np.zeros((maxDim, nlambds))
    MSETest = np.zeros((maxDim, nlambds))
    R2Scores = np.zeros((maxDim, nlambds))

    scaler = StandardScaler()

    pbar = tqdm(
        total=maxDim * nlambds, desc=f"No resampling {model.__class__.__name__}"
    )

    for dim in range(maxDim):
        for i, lmbda in enumerate(lmbds):
            X_train = model.create_X(data.x_train, data.y_train, dim)
            X_test = model.create_X(data.x_test, data.y_test, dim)

            scaler.fit(X_train)
            scaler.transform(X_train)
            scaler.transform(X_test)

            model.fit(X_train, data.z_train, lmbda)
            # betas.append(beta)

            # z_tilde = model.predict(X_train)
            z_pred = model.predict(X_test)

            # MSETrain[dim, i] = MSE(data.z_train, z_tilde)
            MSETest[dim, i] = MSE(data.z_test, z_pred)
            R2Scores[dim, i] = R2Score(data.z_test, z_pred)
            pbar.update(1)

    if title is None:
        title = f"{model.__class__.__name__} with no resampling."
    create_heatmap(MSETest, lmbds, title, **kwargs)


def heatmap_bootstrap(
    data: Data,
    lmbds: np.array = np.logspace(-3, 5, 10),
    model: Model = Ridge(),
    title: str = None,
    var: bool = False,
    **kwargs,
) -> None:
    """Heatmap of lambda vs degree with bootstrapping.

    inputs:
        data (Data): Dataset
        lmbds (np.array): Regularization weights
        model (Model): Regression model to apply
        title (str): Title of the resulting plot
        var (bool): Whether to plot variance
    """
    n_boostraps = 100
    error, bias, variance = bootstrap_lambdas(data, n_boostraps, Ridge(), **kwargs)
    if title is None:
        title = f"{model.__class__.__name__} with bootstrapping " + (
            "(Variance)" if var else "(Error)"
        )
    if var:
        create_heatmap(variance, lmbds, title, **kwargs)
    else:
        create_heatmap(error, lmbds, title, **kwargs)


def heatmap_HomeMade_cross_val(
    data: Data,
    lmbds: np.array = np.logspace(-3, 5, 10),
    model: Model = Ridge(),
    title: str = None,
    var: bool = False,
    **kwargs,
) -> None:
    """Heatmap of lambda vs degree with Cross Validation.

    inputs:
        data (Data): Dataset
        lmbds (np.array): Regularization weights
        model (Model): Regression model to apply
        title (str): Title of the resulting plot
        var (bool): Whether to plot variance
    """
    error, variance = HomeMade_cross_val_lambdas(data, kfolds=5, model=model)
    if title is None:
        title = f"{model.__class__.__name__} with Cross Validation " + (
            "(Variance)" if var else "(Error)"
        )
    if var:
        create_heatmap(variance, lmbds, title, **kwargs)
    else:
        create_heatmap(error, lmbds, title, **kwargs)


def heatmap_sklearn_cross_val(
    data: Data,
    lmbds: np.array = np.logspace(-3, 5, 10),
    model: sklm.Ridge | sklm.Lasso = sklm.Ridge(),
    title=None,
    var: bool = False,
    **kwargs,
) -> None:
    """Heatmap of lambda vs degree with Cross Validation from Scikit-learn.

    inputs:
        data (Data): Dataset
        lmbds (np.array): Regularization weights
        model (Model): Regression model to apply
        title (str): Title of the resulting plot
        var (bool): Whether to plot variance
    """
    error, variance = sklearn_cross_val_lambdas(data, kfolds=5, model=model)
    if title is None:
        title = f"{model.__class__.__name__} with Scikit-learn CV " + (
            "(Variance)" if var else "(Error)"
        )
    if var:
        create_heatmap(variance, lmbds, title, **kwargs)
    else:
        create_heatmap(error, lmbds, title, **kwargs)
