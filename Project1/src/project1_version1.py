import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from util import MSE, R2Score, create_X, FrankeFunction
from OLS import create_OLS_beta
from Ridge import create_ridge_beta


n = 5
N = 100
x = np.sort(np.random.uniform(0, 1, N))
y = np.sort(np.random.uniform(0, 1, N))
x, y = np.meshgrid(x, y)
z = FrankeFunction(x, y) + 0.2 * np.random.randn(N, N)

X_lst = []
for i in range(n):
    X_lst.append(create_X(x, y, i))

train_OLS_MSE = []
train_Ridge_MSE = []
test_OLS_MSE = []
test_Ridge_MSE = []
train_Lasso_MSE = []
test_Lasso_MSE = []


for dim in range(n):
    designX = create_X(x, y, dim)

    # Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(
        designX, np.ravel(z), test_size=0.2
    )

    # -------- Train sets ---------- #

    # OLS
    beta_OLS = create_OLS_beta(X_train, y_train)
    ztilde = X_train @ beta_OLS

    # Ridge
    beta_ridge = create_ridge_beta(X_train, y_train, 0.1)
    z_tilde_ridge = X_train @ beta_ridge

    # Lasso
    model = make_pipeline(PolynomialFeatures(degree=dim), Lasso(fit_intercept=True))
    clf = model.fit(X_train, y_train)
    # clf = Lasso(alpha=0.5, fit_intercept=True)
    # clf.fit(X_train, y_train)
    z_tilde_lasso = clf.predict(X_train)

    # ---------- Test sets ------------ #

    # OLS
    beta_OLS_t = create_OLS_beta(X_test, y_test)
    z_pred_OLS = X_test @ beta_OLS_t

    # Ridge
    beta_ridge_t = create_ridge_beta(X_test, y_test, 0.5)
    z_pred_Ridge = X_test @ beta_ridge_t

    # Lasso
    z_pred_Lasso = clf.predict(X_test)

    # MSE
    train_OLS_MSE.append(MSE(y_train, ztilde))
    train_Ridge_MSE.append(MSE(y_train, z_tilde_ridge))
    train_Lasso_MSE.append(MSE(y_train, z_tilde_lasso))
    test_OLS_MSE.append(MSE(y_test, z_pred_OLS))
    test_Ridge_MSE.append(MSE(y_test, z_pred_Ridge))
    test_Lasso_MSE.append(MSE(y_test, z_pred_Lasso))

    # print(z_tilde_lasso)
    # print(z_tilde_ridge)


xaxis = [i for i in range(1, 6)]
plt.plot(xaxis, train_OLS_MSE, label="OLS_Train")
plt.plot(xaxis, train_Ridge_MSE, label="Ridge_Train")
plt.plot(xaxis, train_Lasso_MSE, label="Lasso_Train")
plt.plot(xaxis, test_OLS_MSE, label="OLS_Test")
plt.plot(xaxis, test_Ridge_MSE, label="Ridge_Test")
plt.plot(xaxis, test_Lasso_MSE, label="Lasso_Test")
plt.legend()
plt.show()
