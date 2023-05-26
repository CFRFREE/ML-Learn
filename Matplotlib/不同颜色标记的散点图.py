import matplotlib.pyplot as plt

x1 = [214, 5, 91, 81, 122, 16, 218, 22]
x2 = [12, 125, 149, 198, 22, 26, 28, 32]

plt.figure(1)
# You can specify the marker size two ways directly:
plt.plot(x1, 'bo', markersize=20)  # blue circle with size 10
plt.plot(x2, 'ro', ms=10, )  # ms is just an alias for markersize
plt.show()
