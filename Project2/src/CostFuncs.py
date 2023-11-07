import autograd.numpy as np


def CostCrossEntropy(target):
    def func(X):
        p0 = target * np.log(X + 1e-10)
        p1 = (1 - target) * np.log(1 - X + 1e-10)
        return -(1.0 / target.size) * np.sum(p0 + p1)

    return func


def CostOLS(target):
    def func(X):
        return (1.0 / target.shape[0]) * np.sum((target - X) ** 2)

    return func


def CostLogReg(target):
    def func(X):
        return -(1.0 / target.shape[0]) * np.sum(
            (target * np.log(X + 10e-10)) + ((1 - target) * np.log(1 - X + 10e-10))
        )

    return func
