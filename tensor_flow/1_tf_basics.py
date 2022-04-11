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
print('r3_matriz data type is ', r3_matrix.dtype)

# creation of ones and zeros matrices
print(tf.zeros(10))
print(tf.ones([2, 2]))
print(tf.eye(3))

# data type transformation
print('\ndata transformation')
print('r2 tensor is', r2, ' dtype is ', r2.dtype)
r2_toint = tf.cast(r2, dtype=tf.int32)
print('transform to int:', r2_toint, ' dtype is ', r2_toint.dtype)

#math operations
print('\nMath operations:')
x = tf.constant(2.0, dtype=tf.float32)
print('sqrt of ', x, 'is ', tf.sqrt(x))

y = tf.constant([1.0, 2.0], dtype=tf.float32)
z = tf.constant([3.0, 4.0], dtype=tf.float32)
print('Sum of ', y, 'and ', z, ' is ', tf.add(y, z))

print('Subtract of ', y, 'and ', z, ' is ', tf.subtract(y, z))

y = tf.constant([[1, 2]])
z = tf.constant([[3], [4]])
print('Multiplication of ', y, 'and ', z, ' is ', tf.multiply(y, z))
# tf.div(), tf.pow(a, b) ...

#variables creation
x = tf.constant([[1, 2], [3, 4]])
var = tf.Variable(x)
print('\nInitialized variable: ', var)
print('Assigned variable:', var.assign([[3, 4], [5, 6]]))