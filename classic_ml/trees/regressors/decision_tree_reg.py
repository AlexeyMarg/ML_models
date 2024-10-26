from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt
import numpy as np

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y)

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X_train, y_train)

y_pred = tree_reg.predict(X_test)
print(mean_absolute_error(y_test, y_pred))

plot_tree(tree_reg)
plt.show()
