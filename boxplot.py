import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, sharex=True, figsize=(16,8))
tips = pd.read_csv("jeu2Donne.csv")

sns.boxplot(ax=axes[0, 0], data=tips, x='Class', y='ON1V')
sns.boxplot(ax=axes[0, 1],  data=tips, x='Class', y='PCD')
sns.boxplot(ax=axes[1, 0], data=tips, x='Class', y='F04[C-O]')
sns.boxplot(ax=axes[1, 1], data=tips, x='Class', y='B02[C-N]')

#Class c'est le nom de ma colone  dans mon jeu de données
#il represente la classe (1,2,3)
tips.describe()

"""sns.boxplot(ax=axes[0, 0],  data=tips, x='Class', y='ON1V')
sns.boxplot(ax=axes[0, 1], data=tips, x='Class', y='MLOGP')
sns.boxplot(ax=axes[1, 0], data=tips, x='Class', y='B02[C-N]')
sns.boxplot(ax=axes[1, 1], data=tips, x='Class', y='F04[C-O]')"""

fig.suptitle('Boite à moustache')
plt.show()

