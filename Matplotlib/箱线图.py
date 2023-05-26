import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [15, 15, 8, 12],
                   [15, 14, 1, 8], [7, 1, 1, 8], [5, 4, 9, 2]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4',
                         'Basket5', 'Basket6'])

box = plt.boxplot(df, patch_artist=True)

colors = ['blue', 'green', 'purple', 'tan', 'pink', 'red']

for patch, color in zip(box['boxes'], colors):
	patch.set_facecolor(color)

plt.show()
