from sklearn.datasets import fetch_openml
import numpy as np
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score

print('Loading of data')
mnist = fetch_openml('mnist_784', version=1, cache=True, as_frame=False)
X, y = mnist['data'].astype(np.int8), mnist['target'].astype(np.int8)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
shuffle_index = np.random.permutation(60000)
X_train, y_train = X[shuffle_index], y[shuffle_index]

n_batches = 100
inc_pca = IncrementalPCA(n_components=154)
for X_batch in np.array_split(X_train, n_batches):
    inc_pca.partial_fit(X_batch)
X_reduced = inc_pca.transform(X_train)


sgd_clf = SGDClassifier(max_iter=10000)
sgd_clf.fit(X_reduced, y_train)

X_test_reduced = inc_pca.transform(X_test)
y_pred = sgd_clf.predict(X_test_reduced)
print(accuracy_score(y_test, y_pred))

