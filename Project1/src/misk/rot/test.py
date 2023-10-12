from sklearn.model_selection import cross_val_score, KFold
import numpy as np

X = np.arange(35)
X = X.reshape(7,5)
Y = np.arange(100, 107)
Kfold = KFold(n_splits = 5)


for train_i, test_i in Kfold.split(X):
    #print(train_i, test_i)
    X_train = X[train_i]
    X_test = X[test_i]
    Y_train = Y[train_i]
    Y_test = Y[test_i]
#    print(X_test)
#    print(Y_train)
#    print(Y_test)

def yay():
    print("yay")
def ney():
    print("ney")
func = yay
func()
