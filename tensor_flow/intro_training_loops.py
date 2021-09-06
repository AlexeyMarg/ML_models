import tensorflow as tf
import matplotlib.pyplot as plt

'''
Linear noised data preparation
'''

TRUE_W = 3.
TRUE_B = 2.

N_SAMPLES = 1000

# A vector of random x values
x = tf.random.normal(shape=[N_SAMPLES])
# Generate some noise
noise = tf.random.normal(shape=[N_SAMPLES])

# Calculate y
y = x * TRUE_W + TRUE_B + noise

plt.scatter(x, y, c='b')
plt.show()