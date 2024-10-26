import numpy as np
from linear_models.dataset_generation import generate_linear_dataset
from sklearn.linear_model import SGDRegressor

n_epochs = 50
t0, t1 = 5, 50  # learning tare decreasing


def learning_schedule(t):
    return t0 / (t + t1)


if __name__ == '__main__':
    m = 100
    X, y = generate_linear_dataset(m)
    X_b = np.c_[np.ones((100, 1)), X]  # add 1 for theta_0 in each X

    print('Numpy solution')
    theta = np.random.randn(2, 1)  # random initial theta
    for epoch in range(n_epochs):
        for i in range(m):
            random_index = np.random.randint(m)
            xi = X_b[random_index: random_index + 1]
            yi = y[random_index: random_index + 1]
            gradients = 2 * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(epoch * m + i)
            theta = theta - eta * gradients
    print(theta)

    print('sklearn solution')
    sgd_reg = SGDRegressor()
    sgd_reg.fit(X, y)
    print(sgd_reg.intercept_, sgd_reg.coef_)

    print('minibatch numpy')
    n_iterations = 50
    minibatch_size = 20
    theta = np.random.randn(2, 1)  # random initialization
    t0, t1 = 200, 1000
    t = 0
    for epoch in range(n_iterations):
        shuffled_indices = np.random.permutation(m)
        X_b_shuffled = X_b[shuffled_indices]
        y_shuffled = y[shuffled_indices]
        for i in range(0, m, minibatch_size):
            t += 1
            xi = X_b_shuffled[i:i + minibatch_size]
            yi = y_shuffled[i:i + minibatch_size]
            gradients = 2 / minibatch_size * xi.T.dot(xi.dot(theta) - yi)
            eta = learning_schedule(t)
            theta = theta - eta * gradients
    print(theta)
