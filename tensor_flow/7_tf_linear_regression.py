from sklearn.datasets import fetch_california_housing
import numpy as np




housing = fetch_california_housing()
m, n = housing.data.shape
housing_biased = np.c_[np.ones((m, 1)), housing.data]

X = tf.constant(housing_biased, dtype=tf.float32, name='X')
y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name='y')

XT = tf.transpose(X)
theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)

print(theta)
