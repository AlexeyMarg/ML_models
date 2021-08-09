import numpy as np
from dataset_generation import generate_linear_dataset

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
