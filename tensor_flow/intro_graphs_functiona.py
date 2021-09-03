import tensorflow as tf
import timeit
from datetime import datetime

'''
You create and run a graph in TensorFlow by using tf.function, either as a direct call or as a decorator. tf.function 
takes a regular function as input and returns a Function. A Function is a Python callable that builds TensorFlow graphs from the Python function. 
You use a Function in the same way as its Python equivalent.
'''

# Define a Python function
def regular_function(x, y, b):
    x = tf.matmul(x, y)
    x = x + b
    return x
# `a_function_that_uses_a_graph` is a TensorFlow `Function`.
function_that_usses_tensorflow = tf.function(regular_function)

# Make some tensors.
x1 = tf.constant([ [1., 2.] ])
y1 = tf.constant([ [2.], [3.] ])
b1 = tf.constant(4.)

regular_value = regular_function(x1, y1, b1).numpy()
print(regular_value)
tensorflow_function_value = function_that_usses_tensorflow(x1, y1, b1).numpy()
print(tensorflow_function_value)
assert(regular_value == tensorflow_function_value)