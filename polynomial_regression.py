from dataset_generation import generate_parabola_dataset, plot_dataset
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

X, y = generate_parabola_dataset(100)
#plot_dataset(X, y)

poly_features = PolynomialFeatures(degree=2, include_bias=False) #transformator to add square of theta to features
X_poly = poly_features.fit_transform(X)

lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)
print(lin_reg.intercept_, lin_reg.coef_)

X_new = np.linspace(-3, 3, 100).reshape(-1, 1)
X_new_poly = poly_features.fit_transform(X_new)
y_predict = lin_reg.predict(X_new_poly)
plt.plot(X_new, y_predict)
plot_dataset(X, y)
