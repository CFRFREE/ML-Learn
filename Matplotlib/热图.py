import matplotlib.pyplot as plt
import pandas as pd

df = pd.DataFrame([[10, 20, 30, 40], [7, 14, 21, 28], [55, 15, 8, 12],
                   [15, 14, 1, 8]],
                  columns=['Apple', 'Orange', 'Banana', 'Pear'],
                  index=['Basket1', 'Basket2', 'Basket3', 'Basket4']
                  )

plt.imshow(df, cmap="YlGnBu")
plt.colorbar()
plt.xticks(range(len(df)), df.columns, rotation=20)
plt.yticks(range(len(df)), df.index)
plt.show()
