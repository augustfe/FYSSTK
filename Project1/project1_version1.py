import matplotlib.pyplot as plt
import numpy as np
from random import random, seed
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

def MSE(y, y_pred):
    n = y.shape[0]
    return np.sum((y - y_pred)**2) / n

def R2Score(y, y_pred):
    s1 = np.sum((y - y_pred)**2)
    m = np.sum(y_pred) / y_pred.shape[0]
    s2 = np.sum((y - m)**2)

    return 1 - s1/s2

def FrankeFunction(x,y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4

def create_X(x,y,n):
    if len(x.shape) > 1:
        x = np.ravel(x)
        y = np.ravel(y)

    N = len(x)
    l = int((n+1)*(n+2)/2)
    X = np.ones((N,l))

    for i in range(1,n+1):
        q = int((i)*(i+1)/2)
        for k in range(i+1):
            X[:,q+k] = (x**(i-k))*(y**k)
            #print(f"x^{i-k} * y^{k}")
    #print()

    return X

def create_beta(X, z, lmbd = 0):
    I = np.identity(X.shape[1])
    return (np.linalg.pinv(X.T @ X + lmbd*I) @ X.T @ z)

n = 5
N = 100
x = np.sort(np.random.uniform(0,1,N))
y = np.sort(np.random.uniform(0,1,N))
x, y = np.meshgrid(x,y)
z = FrankeFunction(x,y) + 0.2 * np.random.randn(N,N)

X_lst = []
for i in range(n):
    X_lst.append(create_X(x,y,i))

train_OLS_MSE = []
train_Ridge_MSE = []
test_OLS_MSE = []
test_Ridge_MSE = []
train_Lasso_MSE = []
test_Lasso_MSE = []


for dim in range(n):
    designX = create_X(x, y, dim)

    # Splitting dataset
    X_train, X_test, y_train, y_test = train_test_split(designX, np.ravel(z), test_size=0.2)

    # -------- Train sets ---------- #

    # OLS
    beta_OLS = create_beta(X_train,y_train,0)
    ztilde = X_train @ beta_OLS

    # Ridge
    beta_ridge = create_beta(X_train,y_train,0.1)
    z_tilde_ridge = X_train @ beta_ridge

    # Lasso
    model = make_pipeline(PolynomialFeatures(degree=dim), Lasso(fit_intercept=True))
    clf = model.fit(X_train, y_train)
    #clf = Lasso(alpha=0.5, fit_intercept=True)
    #clf.fit(X_train, y_train)
    z_tilde_lasso = clf.predict(X_train)

    # ---------- Test sets ------------ #

    # OLS
    beta_OLS_t = create_beta(X_test, y_test, 0)
    z_pred_OLS = X_test @ beta_OLS_t
    

    # Ridge
    beta_ridge_t = create_beta(X_test, y_test, 0.5)
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

    #print(z_tilde_lasso)
    #print(z_tilde_ridge)



xaxis = [i for i in range(1,6)]
plt.plot(xaxis, train_OLS_MSE, label='OLS_Train')
plt.plot(xaxis, train_Ridge_MSE, label='Ridge_Train')
plt.plot(xaxis, train_Lasso_MSE, label='Lasso_Train')
plt.plot(xaxis, test_OLS_MSE, label='OLS_Test')
plt.plot(xaxis, test_Ridge_MSE, label='Ridge_Test')
plt.plot(xaxis, test_Lasso_MSE, label='Lasso_Test')
plt.legend()
plt.show()
