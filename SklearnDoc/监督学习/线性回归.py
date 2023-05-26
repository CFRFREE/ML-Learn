import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge

# X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
# print("X:", X)
# y = np.dot(X, np.array([1, 2])) + 3
# print("y:", y)
# reg = LinearRegression()
# reg.fit(X, y)
# print("w:", reg.coef_)
# print("b:", reg.intercept_)
# print("y_bar:", reg.predict(np.array([[3, 5]])))

X = np.reshape(np.arange(100), [-1, 1])
y = 0.3 * X + 1
reg = LinearRegression()
reg.fit(X, y)
reg2 = Ridge(alpha=.5)
reg2.fit(X, y)
print("w:", reg2.coef_)
print("b:", reg2.intercept_)
# plt.scatter(X, y, color="black")
# plt.plot(X, y, color="blue", linewidth=3)
# plt.show()
