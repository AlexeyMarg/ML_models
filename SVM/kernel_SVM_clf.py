from sklearn.datasets import make_moons
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X, y = make_moons()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

kernel_smv_classifier = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_clf', SVC(kernel='poly', degree=3, coef0=1, C=5))
])

kernel_smv_classifier.fit(X_train, y_train)
y_predict = kernel_smv_classifier.predict(X_test)
print('Accuracy: ', accuracy_score(y_test, y_predict))