import  numpy as np
from linear_models.dataset_generation import generate_linear_dataset

if __name__ == '__main__':
    m=100
    X, y = generate_linear_dataset(m)
    X_b = np.c_[np.ones((100, 1)), X]  # add 1 for theta_0 in each X

    eta = 0.1 # learning rate
    n_iterations = 1000
    theta = np.random.randn(2, 1) ## random initial theta
    for iteration in range(n_iterations):
        gradient = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta * gradient
    print('Solution with use of numpy:')
    print(theta)
