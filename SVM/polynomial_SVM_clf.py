from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_moons()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

polynomial_smv_classifier = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ('svm_clf', LinearSVC(C=10, loss='hinge', max_iter=10000))
])

polynomial_smv_classifier.fit(X_train, y_train)
y_predict = polynomial_smv_classifier.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, y_predict))

