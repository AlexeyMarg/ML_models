from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

tree_clf = RandomForestClassifier(n_estimators=10)
tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))