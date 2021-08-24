import tensorflow as tf

#rank 0 tensor creating
print('Scalar tensors:')
r1 = tf.constant(1, dtype=tf.int16) # zero dimension
print(r1)
r2 = tf.constant(2, tf.int16, name='my_scalar')
print(r2)
r3 = tf.constant(1.23, tf.float32, name='my_float')
print(r3)
r4 = tf.constant('string', tf.string, name='my_string')
print(r4)

#rank 1 tensor creating
print('Rank 1 tensors:')
r1_vector = tf.constant([1, 2, 3], tf.int16)
print(r1_vector)
r2_vector = tf.constant([True, False, False], tf.bool)
print(r2_vector)

# rank3 tensors creating
print("Rank 3 tensor:")
r3_matrix = tf.constant([ [[1, 2],
                           [3, 4],
                           [5, 6]] ], tf.int16)
print(r3_matrix)
print('Shape of r3_matrix is ', r3_matrix.shape)

# creation of ones and zeros matrices
print(tf.zeros(10))
print(tf.ones([2, 2]))
print(tf.eye(3))
