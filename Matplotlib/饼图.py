import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(4, 4))
ax11 = fig.add_subplot(111)
# Data to plot
labels = 'Python', 'C++', 'Ruby', 'Java'
sizes = [250, 130, 75, 200]
colors = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue']

# Plot
w, l, p = ax11.pie(sizes, labels=labels, colors=colors,
                   autopct='%1.1f%%', startangle=140, pctdistance=1, radius=0.5)

pctdists = [.8, .5, .4, .2]

for t, d in zip(p, pctdists):
	xi, yi = t.get_position()
	ri = np.sqrt(xi ** 2 + yi ** 2)
	phi = np.arctan2(yi, xi)
	x = d * ri * np.cos(phi)
	y = d * ri * np.sin(phi)
	t.set_position((x, y))

plt.axis('equal')
plt.legend()
plt.show()
