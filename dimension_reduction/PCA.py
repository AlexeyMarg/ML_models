from sklearn.datasets import  make_swiss_roll
from numpy.linalg import svd
from sklearn.decomposition import PCA
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X, y = make_swiss_roll()

def my_PCA(X):
    X_centered = X - X.mean(axis=0)
    U, S, Vt = svd(X_centered)
    W2 = Vt.T[:, :2]
    return X_centered.dot(W2)

#X2D = my_PCA(X)

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X2D, y)

kernel_smv_regressor = Pipeline([
    ('scaler', StandardScaler()),
    ('svm_reg', SVR(kernel='poly', degree=2, C=100, epsilon=0.1))
])

kernel_smv_regressor.fit(X_train, y_train)
y_predict = kernel_smv_regressor.predict(X_test)
print(mean_squared_error(y_test, y_predict))

