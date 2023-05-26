import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(dict(
	A=[1, 2, 3, 4],
	B=[2, 3, 4, 5],
	C=[3, 4, 5, 6]
))

fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

df.plot.bar(ax=axes[0])
df.diff(axis=1).fillna(df).astype(df.dtypes).plot.bar(ax=axes[1], stacked=True)

plt.show()
