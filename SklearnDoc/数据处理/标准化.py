import numpy as np
from sklearn import preprocessing

X_train = np.array([[1., -1., 2.],
                    [2., 0., 0.],
                    [0., 1., -1.]])
X_scaled = preprocessing.scale(X_train)
print(X_scaled.mean(axis=0))
print(X_scaled.std(axis=0))
