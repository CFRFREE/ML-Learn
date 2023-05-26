from sklearn.linear_model import SGDClassifier

# X = [[0., 0.], [1., 1.]]
# y = [0, 1]
# clf = SGDClassifier(loss="hinge", penalty="l2")
# clf.fit(X, y)
# SGDClassifier(alpha=0.0001, average=False, class_weight=None,
#               early_stopping=False, epsilon=0.1, eta0=0.0, fit_intercept=True,
#               l1_ratio=0.15, learning_rate='optimal', loss='hinge',
#               max_iter=1000, n_iter_no_change=5, n_jobs=None, penalty='l2',
#               power_t=0.5, random_state=None, shuffle=True, tol=0.001,
#               validation_fraction=0.1, verbose=0, warm_start=False)
# print(clf.coef_)

import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.inspection import DecisionBoundaryDisplay

# import some data to play with
iris = datasets.load_iris()

# we only take the first two features. We could
# avoid this ugly slicing by using a two-dim dataset
X = iris.data[:, :2]
y = iris.target
colors = "bry"

# shuffle
idx = np.arange(X.shape[0])
np.random.seed(13)
np.random.shuffle(idx)
X = X[idx]
y = y[idx]

# standardize
mean = X.mean(axis=0)
std = X.std(axis=0)
X = (X - mean) / std

clf = SGDClassifier(alpha=0.001, max_iter=100).fit(X, y)
ax = plt.gca()
DecisionBoundaryDisplay.from_estimator(
	clf,
	X,
	cmap=plt.cm.Paired,
	ax=ax,
	response_method="predict",
	xlabel=iris.feature_names[0],
	ylabel=iris.feature_names[1],
)
plt.axis("tight")

# Plot also the training points
for i, color in zip(clf.classes_, colors):
	idx = np.where(y == i)
	plt.scatter(
		X[idx, 0],
		X[idx, 1],
		c=color,
		label=iris.target_names[i],
		edgecolor="black",
		s=20,
	)
plt.title("Decision surface of multi-class SGD")
plt.axis("tight")

# Plot the three one-against-all classifiers
xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
coef = clf.coef_
intercept = clf.intercept_


def plot_hyperplane(c, color):
	def line(x0):
		return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

	plt.plot([xmin, xmax], [line(xmin), line(xmax)], ls="--", color=color)


for i, color in zip(clf.classes_, colors):
	plot_hyperplane(i, color)
plt.legend()
plt.show()
