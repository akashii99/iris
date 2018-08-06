# Load libraries
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
import pandas as pd
import matplotlib.pyplot as plt

file = 'Iris.xls'
data = pd.ExcelFile(file)
iris = data.parse('Iris')
#print(iris.shape)
print(iris.head())
#print(iris.describe())
print(iris.groupby('iris').size())
#iris.plot(kind='box',subplots=True,layout=(2,2))
#iris.hist()
#scatter_matrix(iris)
#plt.show()

# Machine Learning Part
array=iris.values
X=array[:,0:4]
Y=array[:,4]
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=42)

# Spot Check Algorithms
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))
# evaluate each model in turn
results = []
names = []
for name, model in models:
	cv_results = cross_val_score(model, X_train, Y_train, cv=10, scoring='accuracy')
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)

'''plotting all the algorithms
plt.boxplot(results).add_subplot(111)
plt.xticks(range(len(names)), names)
plt.ylabel('score')
plt.title('Algorithm Comparison')
plt.show()'''

'''# Compare Algorithms
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()'''

#KNN
knn=KNeighborsClassifier()
knn.fit(X_train,Y_train)
predictions=knn.predict(X_test)
print(accuracy_score(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))

#LDA
lda=LinearDiscriminantAnalysis()
lda.fit(X_train,Y_train)
predictions=lda.predict(X_test)
print(accuracy_score(Y_test,predictions))
print(confusion_matrix(Y_test,predictions))
print(classification_report(Y_test,predictions))