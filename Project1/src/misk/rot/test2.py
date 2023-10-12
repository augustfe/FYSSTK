import numpy as np
from sklearn.utils import resample

X = np.arange(70).reshape(7, 10)
y = np.arange(7)
# y = y[: , np.newaxis]
print(X)
print(y)
_x, _y = resample(X, y)
print(_x)
print(_y)
print(_x.shape)
print(_y.shape)
