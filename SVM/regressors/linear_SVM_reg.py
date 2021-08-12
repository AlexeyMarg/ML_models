import numpy as np
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

polynomial_smv_regressor = Pipeline([
    ('poly_features', PolynomialFeatures(degree=3)),
    ('scaler', StandardScaler()),
    ('svm_clf', LinearSVR(epsilon=1.5))
])

polynomial_smv_regressor = LinearSVR(epsilon=1.5)
polynomial_smv_regressor.fit(X, y)
y_pred = polynomial_smv_regressor.predict([[2.5]])
y_true = 0.5 * 2.5 ** 2 + 2.5 + 2
print(y_true)
print(y_pred)

