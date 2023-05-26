import numpy as np
from sklearn import preprocessing

# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
#
# iris = load_iris()
# X, y = iris.data, iris.target
# X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# quantile_transformer = preprocessing.QuantileTransformer(random_state=0)
# X_train_trans = quantile_transformer.fit_transform(X_train)
# X_test_trans = quantile_transformer.transform(X_test)
# print(np.percentile(X_train_trans[:, 0], [0, 25, 50, 75, 100]))

pt = preprocessing.PowerTransformer(method='box-cox', standardize=False)
X_lognormal = np.random.RandomState(616).lognormal(size=(3, 3))
print(X_lognormal)
print(pt.fit_transform(X_lognormal))
