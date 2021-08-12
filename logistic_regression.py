from sklearn import datasets
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

iris = datasets.load_iris()
X = iris['data'][:, 3:]
y = (iris['target'] == 2).astype(np.int) # if virgin - 1, else 0
log_reg = LogisticRegression()
log_reg.fit(X, y)

X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new)
plt.plot(X_new, y_proba[:, 1], "g-", label="Ирис виргинский")
plt.plot(X_new, y_proba[:, 0], "b--", label="Не ирис виргинский")
#plt.show()

Х = iris["data"][:, (2, 3)] # длина лепестка , пмрина лепестка
у = iris["target"]
softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10)
softmax_reg.fit(X, у)
print(softmax_reg.predict(np.array([5, 2]).reshape(-1, 1)))
print(softmax_reg.predict_proba(np.array([5, 2]).reshape(-1, 1)))