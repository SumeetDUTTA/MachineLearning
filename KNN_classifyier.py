# Importing required Modules
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

# Loading Dataset
iris = datasets.load_iris()

features = iris.data
labels = iris.target


# Training the Classifier
clf = KNeighborsClassifier()
clf.fit(features, labels)

# Predicting The Value
prediction = clf.predict([[1, 1, 1, 1]])
print(prediction)