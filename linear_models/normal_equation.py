import  numpy as np
from linear_models.dataset_generation import generate_linear_dataset
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression



def solve_normal_equation(X, y):
    X_b = np.c_[np.ones((100, 1)), X] # add 1 for theta_0 in each X
    theta_hat = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
    return theta_hat

if __name__ == '__main__':
    X, y = generate_linear_dataset(100)
    print('Solution with numpy:')
    theta_hat = solve_normal_equation(X,y)
    print(theta_hat)

    print('Solution with sklearn:')
    lin_reg = LinearRegression()
    lin_reg.fit(X,y)
    print(lin_reg.intercept_, lin_reg.coef_)

    X_new = np.array([[0], [2]])
    X_new_b = np.c_[np.ones((2, 1)), X_new]
    y_predict_np = X_new_b.dot(theta_hat)
    y_predict_LinearRegressor = lin_reg.predict(X_new)
    plt.plot(X, y, 'o')
    plt.plot(X_new, y_predict_np,
             X_new, y_predict_LinearRegressor)
    plt.show()