from sklearn.datasets import fetch_california_housing
import numpy as np
import tensorflow as tf

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_biased = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_data_biased, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')



n, m = X.shape
scaled_X = X.numpy()
for i in range(m):
    scaled_X[:, i] = scaled_X[:, i] / tf.norm(scaled_X[:, i])
scaled_X = tf.constant(scaled_X, dtype=tf.float32, name='scaled_X')




n_epochs = 1000
learning_rate = 0.01
