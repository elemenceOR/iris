from sklearn import datasets, tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.5)
classifier = tree.DecisionTreeClassifier()
classifier.fit(x_train, y_train)
prediction = classifier.predict(x_test)

print(accuracy_score(y_test, prediction))
