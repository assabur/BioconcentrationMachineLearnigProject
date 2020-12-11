import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#data = pd.read_csv('jeu2Donne.csv', sep=",")
df = pd.read_csv("jeu2Donne.csv")
corrMatrix = df.corr()
# Generate a mask for the upper triangle
mask = np.triu(np.ones_like(corrMatrix, dtype=bool))
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(11, 15))
# Draw the heatmap with the mask and correct aspect ratio
# Generate a custom diverging colormap
cmap = sns.diverging_palette(230, 20, as_cmap=True)
sns.heatmap(corrMatrix,mask=mask,cmap= cmap, vmax=1, center=0,
            square=True,linewidths=.5, cbar_kws={"shrink": .5})
plt.title("Correlation matrix QSAR fish bioconcentration factor ")
plt.show()
