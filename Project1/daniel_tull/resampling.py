# here we can write our funcs such that we take a sklear model as input
from Models import OLS, Ridge, Lasso, Model
import numpy as np
from metrics import *
from sklearn.utils import resample
from sklearn.pipeline import Pipeline

# from globals import *
from sklearn.model_selection import cross_val_score, KFold
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

# from sklearn.linear_model import Lasso as LassoSKL
# from sklearn.linear_model import Ridge as RidgeSKL
# from sklearn.linear_model import LinearRegression as OLSSKL


def bootstrap_degrees(data, polyDegrees, n_bootstraps, model):
    """
    Performs bootstrap on different polydegrees
    """
    n_degrees = len(polyDegrees)

    error = np.zeros(n_degrees)
    bias = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)
    scaler = StandardScaler()
    pbar = tqdm(
        total=len(polyDegrees) * n_bootstraps,
        desc=f"Bootstrap {model.__class__.__name__}",
    )
    for j, dim in enumerate(polyDegrees):
        X_train = model.create_X(data.x_train, data.y_train, dim)
        X_test = model.create_X(data.x_test, data.y_test, dim)

        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        z_test, z_train = data.z_test, data.z_train
        z_pred = np.empty((z_test.shape[0], n_bootstraps))

        for i in range(n_bootstraps):
            X_, z_ = resample(X_train, z_train, replace=True)
            model.fit(X_, z_)
            z_pred[:, i] = model.predict(X_test).ravel()
            pbar.update(1)

        error[j] = mean_MSE(z_test, z_pred)
        bias[j] = get_bias(z_test, z_pred)
        variance[j] = get_variance(z_pred)

    return error, bias, variance


def bootstrap_lambdas(data, polyDegrees, lambdas, model, n_bootstraps):
    """
    performs bootstrap on polydegrees and hyperparamater lambdas
    for regularized regression
    """
    n_degrees = len(polyDegrees)
    n_lambdas = lambdas.size

    error = np.zeros((n_degrees, n_lambdas))
    bias = np.zeros((n_degrees, n_lambdas))
    variance = np.zeros((n_degrees, n_lambdas))
    scaler = StandardScaler()

    # for i, dim in tqdm(enumerate(polyDegrees)):
    pbar = tqdm(
        total=n_degrees * n_lambdas * n_bootstraps,
        desc=f"Bootstrap for {model.__class__.__name__}",
    )

    for i, degree in enumerate(polyDegrees):
        X_train = model.create_X(data.x_train, data.y_train, degree)
        X_test = model.create_X(data.x_test, data.y_test, degree)

        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        z_test, z_train = data.z_test, data.z_train

        z_test = z_test.reshape(z_test.shape[0], 1)
        for j, lambd in enumerate(lambdas):
            z_pred = np.empty((z_test.shape[0], n_bootstraps))
            for k in range(n_bootstraps):
                X_, z_ = resample(X_train, z_train, replace=True)
                model.fit(X_, z_, lambd)
                z_pred[:, k] = model.predict(X_test).ravel()
                pbar.update(1)

            error[i, j] = mean_MSE(z_test, z_pred)
            bias[i, j] = get_bias(z_test, z_pred)
            variance[i, j] = get_variance(z_pred)

    return error, bias, variance


def sklearn_cross_val(data, polyDegrees, kfolds, model):
    """
    sklearn cross val for on polynomial degrees
    """

    n_degrees = len(polyDegrees)

    error = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)
    dummy_model = Model()
    scaler= StandardScaler()
    cv = KFold(n_splits=kfolds, shuffle=True)
    for i, degree in tqdm(enumerate(polyDegrees), total=n_degrees):
        X = dummy_model.create_X(data.x, data.y, degree)

        pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])

        scores = cross_val_score(
            pipeline,
            X,
            data.z,
            scoring="neg_mean_squared_error",
            cv=cv,
            n_jobs=-1,
        )
        error[i] = -scores.mean()
        variance[i] = scores.std()

    return error, variance


def kfold_score_degrees(data, polyDegrees, kfolds: int = 5, model=OLS()):
    """
    HomeCooked cross-val using Kfold. Only for polynomial degrees.
    """
    scaler = StandardScaler()

    n_degrees = len(polyDegrees)

    error = np.zeros(n_degrees)
    variance = np.zeros(n_degrees)

    Kfold = KFold(n_splits=kfolds, shuffle=True)

    pbar = tqdm(total=n_degrees * kfolds, desc=f"K-Fold {model.__class__.__name__}")
    for i, degree in enumerate(polyDegrees):
        scores = np.zeros(kfolds)

        X = model.create_X(data.x_, data.y_, degree)


        for j, (train_i, test_i) in enumerate(Kfold.split(X)):
            X_train = X[train_i]
            X_test = X[test_i]
            z_train = data.z_[train_i]
            z_test = data.z_[test_i]

            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)

            model.fit(X_train, z_train)

            z_pred = model.predict(X_test)

            scores[j] = MSE(z_pred, z_test)
            pbar.update(1)

        # print(scores)
        error[i] = scores.mean()
        variance[i] = scores.std()
    return error, variance


def sklearn_cross_val_lambdas(data, polyDegrees, lambdas, kfolds, model):
    """
    sklearn cross val for polydegrees and lambda
    """

    n_degrees = len(polyDegrees)
    n_lambdas = len(lambdas)
    error = np.zeros((n_degrees, n_lambdas))
    variance = np.zeros((n_degrees, n_lambdas))

    dummy_model = Model()  # only needed because of where create X is
    cv = KFold(n_splits=kfolds, shuffle=True)

    pbar = tqdm(
        total=n_degrees * n_lambdas, desc=f"sklearn CV {model.__class__.__name__}"
    )
    for i, degree in enumerate(polyDegrees):
        X = dummy_model.create_X(data.x_, data.y_, degree)
        for j, lambd in enumerate(lambdas):
            model.alpha = lambd

            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('model', model)
            ])
            scores = cross_val_score(
                pipeline,
                X,
                data.z_,
                scoring="neg_mean_squared_error",
                cv=cv,
                n_jobs=-1,
            )
            error[i, j] = -scores.mean()
            variance[i, j] = scores.std()
            pbar.update(1)

    return error, variance

def Newsklearn_cross_val_lambdas(data, polyDegrees, lambdas, kfolds, model):
    """
    sklearn cross val for polydegrees and lambda
    """

    n_degrees = len(polyDegrees)
    n_lambdas = len(lambdas)
    error = np.zeros((n_degrees, n_lambdas))
    variance = np.zeros((n_degrees, n_lambdas))

    cv = KFold(n_splits=kfolds, shuffle=True)

    pipeline = Pipeline([
        ('poly', PolynomialFeatures()),
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    param_grid = {
        'poly__degree': polyDegrees,
        'model__alpha': lambdas,
    }

    grid_search = GridSearchCV(pipeline, param_grid=param_grid, scoring='neg_mean_squared_error', cv=cv, n_jobs=-1, verbose=1)
    grid_search.fit(data.x_, data.z_)

    error = -grid_search.cv_results_['mean_test_score'].reshape(n_degrees, n_lambdas)
    variance = grid_search.cv_results_['std_test_score'].reshape(n_degrees, n_lambdas)

    return error, variance





def HomeMade_cross_val_lambdas(data, polyDegrees, lambdas, kfolds, model):
    """
    HomeCooked cross-val using Kfold. for polydegrees and lambda.
    """
    n_degrees = len(polyDegrees)
    n_lambdas = lambdas.size

    error = np.zeros((n_degrees, n_lambdas))
    variance = np.zeros((n_degrees, n_lambdas))

    Kfold = KFold(n_splits=kfolds, shuffle=True)

    pbar = tqdm(
        total=n_degrees * n_lambdas * kfolds,
        desc=f"Homemade CV {model.__class__.__name__}",
    )
    for i, degree in enumerate(polyDegrees):
        scores = np.zeros(kfolds)

        X = model.create_X(data.x_, data.y_, degree)

        for j, lambd in enumerate(lambdas):
            for k, (train_i, test_i) in enumerate(Kfold.split(X)):
                X_train = X[train_i]
                X_test = X[test_i]
                z_test = data.z_[test_i]
                z_train = data.z_[train_i]

                scaler.fit(X_train)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)

                model.fit(X_train, z_train, lambd)

                z_pred = model.predict(X_test)

                scores[k] = MSE(z_pred, z_test)
                pbar.update(1)
            # print(scores)
            error[i, j] = scores.mean()
            variance[i, j] = scores.std()
    return error, variance
