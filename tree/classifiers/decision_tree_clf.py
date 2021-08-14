from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data[:, 2:]
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

plot_tree(tree_clf)
plt.show()
