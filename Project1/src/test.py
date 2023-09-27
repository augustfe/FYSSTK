import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from OLS import *

# Generate some random data
np.random.seed(123)
x1 = np.random.randint(1, 101, size=10)
x2 = np.random.randint(1, 101, size=10)
y_true = 0.5 * x1 - 50 * x2 + 20
y_noise = np.random.normal(0, 5, size=10)
y = y_true + y_noise

# Fit the multiple linear regression model without scaling
print("MSETrain                   MSETest                 R2")
X = np.column_stack((x1, x2))
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y)
result = OLS(X_train, X_test, y_train, y_test)
print(result)



# Scale the input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print(X_scaled)
print(X)
# Fit the multiple linear regression model on the scaled input features
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y)
mean = np.mean(y_train)
maxelm = np.max(np.abs(y_train))
#y_train -= mean
#y_test -= mean
#y_train /= maxelm
#y_test /= maxelm

result = OLS(X_train, X_test, y_train, y_test)
print(result)
