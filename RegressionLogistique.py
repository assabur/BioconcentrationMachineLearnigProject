import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
#utilisation du module model_selection de scikit-learn (sklearn)
from sklearn import model_selection
df = pd.read_csv('jeu2Donne.csv')
#supprimons les colones qui nous interrese pas
#nHM,piPC09,MLOGP,ON1V,,,F04[C-O],Class
for column in ("CAS","SMILES","Set","PCD","X2Av","N-072","logBCF","B02[C-N]" ):
       df.drop([column],axis='columns',inplace=True)

X = df[df.columns[:-1]].values
Y_class = df['Class'].values
#Séparons nos données en un jeu d’entraînement et un jeu de test. Le jeu de test
X_train, X_test, y_train, y_test = \
	model_selection.train_test_split(X, Y_class,
                                	test_size=0.25# 25% des données dans le jeu de test
                                	)
""""
Nous pouvons maintenant standardiser les données d’entraînement
et appliquer la même transformation aux données de test :
"""
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

#création d'une instance de la classe
lr = LogisticRegression(solver="liblinear")
#exécution de l'instance sur la totalité des données (X,Y_class)
modele_all = lr.fit(X,Y_class)

#évaluation en validation croisée : 10 cross-validation
validation = model_selection.cross_val_score(lr,X,Y_class,cv=10,scoring='accuracy')
print(validation)


















""""
fig = plt.figure(figsize=(16, 12))
for feat_idx in range(X_train_std.shape[1]):
    ax = fig.add_subplot(3,4, (feat_idx+1))
    h = ax.hist(X_train_std[:, feat_idx], bins=50, color = 'steelblue', density=True, edgecolor='none')
    ax.set_title(df.columns[feat_idx], fontsize=14)

plt.show()"""



"""
# On utilise l'argument 'hue' pour fournir une variable de facteur
sns.lmplot( x="nHM", y="Class", data=df, fit_reg=False, legend=False)
plt.legend(loc='lower right')
plt.show()
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(14,4))
for xcol, ax in zip(['F04[C-O]', 'ON1V', 'piPC09'], axes):
       df.plot(kind='scatter', x=xcol, y='Class', ax=ax, alpha=0.5, color='r')
plt.show()
"""
