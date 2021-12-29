from sklearn.datasets import make_swiss_roll
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = make_swiss_roll()

lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10)
X2D = lle.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X2D, y)

kernel_smv_regressor = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_reg', SVR(kernel='poly', degree=2, C=100, epsilon=0.1))
])

kernel_smv_regressor.fit(X_train, y_train)
y_predict = kernel_smv_regressor.predict(X_test)
print(mean_squared_error(y_test, y_predict))

