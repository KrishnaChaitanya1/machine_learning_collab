import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import normalize

plt.style.use('ggplot')

#load and clean data
df = pd.read_csv('train.csv')
df = df.dropna(axis=0)
y = df['Survived']
df = df.drop(['Survived','Name','Ticket','Cabin','Fare'],1)
df= pd.get_dummies(df)



#split data for later test
df_train, df_test, y_train, y_test = train_test_split(df, y, random_state=42)

#color code for later plot
color = []
for i in range(137):
	if y_train.iloc[i]==0:
		color.append('r')
	else:
		color.append('b')

#Normalize 
df_train = normalize(df_train)
df_test = normalize(df_test)


#pca for visualization (not needed for classification) 
#pca = PCA(n_components = 3, random_state = 42).fit(df_train)
#df_train = pca.transform(df_train,y_train)
#df_test = pca.transform(df_test, y_test)


#sklearn method for classification
knc = KNeighborsClassifier(n_neighbors=13)
knc = knc.fit(df_train, y_train)

#using fit to score test data
print(knc.score(df_test,y_test))

#plot a visualization(unfortunatly 3 dimensions is the max:(, blame God!)
fig = plt.figure()
ax = fig.add_subplot(111)
ax = Axes3D(fig)
ax.scatter(df_train[:,0],df_train[:,1],df_train[:,2], marker='.',c=color, alpha=0.3)
plt.show()
