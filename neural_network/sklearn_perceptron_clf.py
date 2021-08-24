from sklearn.linear_model import Perceptron
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import  accuracy_score

iris = load_iris()
X = iris.data[:, (2, 3)]
y = (iris.target == 0).astype(np.int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

perc_clf = Perceptron()
perc_clf.fit(X_test, y_test)
y_pred = perc_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))