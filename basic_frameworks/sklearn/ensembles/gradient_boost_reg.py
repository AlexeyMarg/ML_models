import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

X_train, X_val, y_train, y_val = train_test_split(X, y)

gb_reg = GradientBoostingRegressor(max_depth=2, n_estimators=120)
gb_reg.fit(X_train, y_train)

# calculate error on each step of fitting
errors = [mean_squared_error(y_val, y_pred) for y_pred in gb_reg.staged_predict(X_val)]
# find best number of estimators
best_n_estimatiors = np.argmin(errors)

gb_reg_best = GradientBoostingRegressor(max_depth=2, n_estimators=best_n_estimatiors)
gb_reg_best.fit(X_train, y_train)
y_predict = gb_reg_best.predict(X_val)
print(mean_squared_error(y_val, y_predict))