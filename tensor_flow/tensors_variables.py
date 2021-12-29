import tensorflow as tf
import numpy as np

# BASICS
# Create 0D tensor (int32)
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)

# Create 1D-vector
rank_1_tensor = tf.constant([2.0, 3.0, 4.0])
print(rank_1_tensor)

# Create 2D matrix with specific dtype
rank_2_tensor = tf.constant([[1, 2],
                             [3, 4]], dtype=tf.float16)

# Tensor to numpy array
print(np.array(rank_2_tensor))

# Tensor math operations
print('\n\nMath operations\n\n')
a = tf.constant([[1, 2],
                 [3, 4]])
b = tf.constant([[1, 1],
                 [1, 1]]) # Could have also said `tf.ones([2,2])`

print('a+b = ', tf.add(a, b), '\n')
print('element multiply: ', tf.multiply(a, b), '\n')
print('matrix multiply: ', tf.matmul(a, b), '\n')

'''
print(a + b, "\n") # element-wise addition
print(a * b, "\n") # element-wise multiplication
print(a @ b, "\n") # matrix multiplication
'''

c = tf.constant([[4.0, 5.0], [10.0, 1.0]])

# Find the largest value
print(tf.reduce_max(c))
# Find the index of the largest value
print(tf.argmax(c))
# Compute the softmax
print(tf.nn.softmax(c))

# Shapes
print('\n\nShapes\n\n')
rank_4_tensor = tf.zeros([3, 2, 4, 5])

print("Type of every element:", rank_4_tensor.dtype)
print("Number of axes:", rank_4_tensor.ndim)
print("Shape of tensor:", rank_4_tensor.shape)
print("Elements along axis 0 of tensor:", rank_4_tensor.shape[0])
print("Elements along the last axis of tensor:", rank_4_tensor.shape[-1])
print("Total number of elements (3*2*4*5): ", tf.size(rank_4_tensor).numpy())

# Indexing
print('\n\nIndexing\n\n')
print("Second row:", rank_2_tensor[1, :].numpy())
print("Second column:", rank_2_tensor[:, 1].numpy())
print("Last row:", rank_2_tensor[-1, :].numpy())
print("First item in last column:", rank_2_tensor[0, -1].numpy())
print("Skip the first row:")
print(rank_2_tensor[1:, :].numpy(), "\n")

# Indexing
print('\n\nManipulating Shapes\n\n')
x = tf.constant([[1], [2], [3]])
print(x.shape)
print(x.shape.as_list())
reshaped = tf.reshape(x, [1, 3])
print('Resyaped 1x3: ', reshaped, ' shape: ', reshaped.shape)
print('Automatically calculate number of elements with reshape: ', tf.reshape(rank_2_tensor, [-1]))

# Sparse tensors
print('\n\nSparse tensors\n\n')
sparse_tensor = tf.sparse.SparseTensor(indices=[[0, 0], [1, 2]],
                                       values=[1, 2],
                                       dense_shape=[3, 4])
print(sparse_tensor, "\n")

# You can convert sparse tensors to dense
print(tf.sparse.to_dense(sparse_tensor))