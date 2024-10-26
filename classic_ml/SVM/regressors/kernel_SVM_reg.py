import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

kernel_smv_regressor = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_reg', SVR(kernel='poly', degree=2, C=100, epsilon=0.1))
])

kernel_smv_regressor.fit(X, y)
y_pred = kernel_smv_regressor.predict([[2.5]])
y_true = 0.5 * 2.5 ** 2 + 2.5 + 2
print(y_true)
print(y_pred)

