import matplotlib.pyplot as plt

y1 = [12, 14, 15, 18, 19, 13, 15, 16]
y2 = [22, 24, 25, 28, 29, 23, 25, 26]
y3 = [32, 34, 35, 38, 39, 33, 35, 36]
y4 = [42, 44, 45, 48, 49, 43, 45, 46]
y5 = [52, 54, 55, 58, 59, 53, 55, 56]

# Plot lines with different marker sizes
plt.plot(y1, y2, label='Y1-Y2', lw=2, marker='s', ms=10, ls='--')  # square
plt.plot(y1, y3, label='Y1-Y3', lw=2, marker='^', ms=10, ls='-.')  # triangle
plt.plot(y1, y4, label='Y1-Y4', lw=2, marker='o', ms=10, ls=':')  # circle
plt.plot(y1, y5, label='Y1-Y5', lw=2, marker='D', ms=10)  # diamond
plt.plot(y2, y5, label='Y2-Y5', lw=2, marker='P', ms=10)  # filled plus sign

plt.legend()
plt.show()
