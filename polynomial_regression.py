from dataset_generation import generate_parabola_dataset, plot_dataset
from sklearn.preprocessing import PolynomialFeatures

X, y = generate_parabola_dataset(100)
#plot_dataset(X, y)

poly_features = PolynomialFeatures(degree=2, include_bias=False) #transformator to add square of theta to features
X_poly = poly_features.fit_transform(X)
print(X_poly)