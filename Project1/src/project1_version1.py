import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import train_test_split
from util import create_X, FrankeFunction
from OLS import OLS
from Ridge import Ridge
from Lasso import pureSKLearnLasso

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

    # MSE
    MSE_OLS = OLS(X_train, X_test, y_train, y_test)
    train_OLS_MSE.append(MSE_OLS[0])
    test_OLS_MSE.append(MSE_OLS[1])

    MSE_Ridge = Ridge(X_train, X_test, y_train, y_test, lmbda=0.1)
    train_Ridge_MSE.append(MSE_Ridge[0])
    test_Ridge_MSE.append(MSE_Ridge[1])

    MSE_Lasso = pureSKLearnLasso(X_train, X_test, y_train, y_test, alpha=0.01)
    train_Lasso_MSE.append(MSE_Lasso[0])
    test_Lasso_MSE.append(MSE_Lasso[1])


xaxis = [i for i in range(1, 6)]
plt.plot(xaxis, train_OLS_MSE, label="OLS_Train")
plt.plot(xaxis, train_Ridge_MSE, label="Ridge_Train")
plt.plot(xaxis, train_Lasso_MSE, label="Lasso_Train")
plt.plot(xaxis, test_OLS_MSE, label="OLS_Test")
plt.plot(xaxis, test_Ridge_MSE, label="Ridge_Test")
plt.plot(xaxis, test_Lasso_MSE, label="Lasso_Test")
plt.legend()
plt.show()
