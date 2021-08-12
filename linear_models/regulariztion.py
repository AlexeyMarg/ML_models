from linear_models.dataset_generation import generate_parabola_dataset, plot_dataset
from sklearn.linear_model import Ridge, SGDRegressor, Lasso, ElasticNet
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures

X, y = generate_parabola_dataset(100)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features.fit_transform(X)

print('Solution with ridge regularization (add norm of parameters squares sum to MSE, solve normal equation)')
ridge_reg = Ridge(alpha=1, solver='cholesky')
ridge_reg.fit(X_poly, y)
print(ridge_reg.intercept_, ridge_reg.coef_)
X_new = np.linspace(-3, 3, 100).reshape(-1, 1)
X_new_poly = poly_features.fit_transform(X_new)
y_predict_ridge = ridge_reg.predict(X_new_poly)

print('Use of SGDRegressor')
sgd_reg = SGDRegressor(penalty='l2')
sgd_reg.fit(X_poly, y)
print(sgd_reg.intercept_, sgd_reg.coef_)
y_predict_sgd = sgd_reg.predict(X_new_poly)

print('Solution with LASSO regularization (add sum of parameters modules to MSE, solve normal equation)')
lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_poly, y)
print(lasso_reg.intercept_, lasso_reg.coef_)
y_predict_lasso = lasso_reg.predict(X_new_poly)

print('Solution with elastic net regularization mix of rifge and lasso')
en_reg = ElasticNet(alpha=0.1, l1_ratio=0.5) #l1_ratio = 0 - ridge, 1 - lasso
en_reg.fit(X_poly, y)
print(en_reg.intercept_, en_reg.coef_)
y_predict_en = en_reg.predict(X_new_poly)


plt.plot(X_new, y_predict_ridge,
         X_new, y_predict_sgd,
         X_new, y_predict_lasso,
         X_new, y_predict_en)
plot_dataset(X, y)
