import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
df_tips = pd.read_csv('jeu2Donne.csv')

df_tips['Class'].plot(kind='hist');
#df_tips['logBCF'].plot(kind='hist');
#df_tips['PCD'].plot(kind='hist');
#df_tips['piPC09'].plot(kind='hist')
df_tips.info()
plt.show()
