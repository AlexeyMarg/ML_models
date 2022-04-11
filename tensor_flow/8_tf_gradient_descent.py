from sklearn.datasets import fetch_california_housing
import numpy as np
import tensorflow as tf

def z_scaling(X):
    n, m = X.shape
    temp = X[:, 0]
    mean = tf.reduce_min(temp).numpy().astype(float)
    std = tf.math.reduce_std(temp).numpy().astype(float)
    temp = temp - mean
    if std != 0:
        X_norm = tf.divide(temp, std)
    else:
        X_norm = temp
    X_norm = tf.expand_dims(X_norm, 1)
    for i in range(1, m):
        temp = X[:, i]
        mean = tf.reduce_min(temp).numpy().astype(float)
        std = tf.math.reduce_std(temp).numpy().astype(float)
        temp_norm = temp - mean
        if std != 0:
            temp_norm = tf.divide(temp, std)
        temp_norm = tf.expand_dims(temp_norm, 1)
        X_norm = tf.concat([X_norm, temp_norm], 1)
    return X_norm

housing = fetch_california_housing()
m, n = housing.data.shape
m, n = housing.data.shape
housing_biased = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_biased, dtype=tf.float32, name='X')
X = z_scaling(X)
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')

n_epochs = 1000
learning_rate = 0.01

theta = tf.Variable(tf.random.uniform([n + 1, 1], -1.0, 1.0), name='theta')
for epoch in range(n_epochs):
    if epoch % 100 == 0:
        y_pred = tf.matmul(X, theta, name='predictions')
        error = y - y_pred
        mse = tf.reduce_min(tf.square(error), name='mse')
        gradients = 2 / m * tf.matmul(tf.transpose(X), error)
        theta.assign(theta - learning_rate * gradients)
        print('Epoch: ', epoch, ' MSE: ', mse)





