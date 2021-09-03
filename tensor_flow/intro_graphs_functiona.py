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

# use of decorator

def inner_function(x, y, b):
    x = tf.matmul(x, y)
    x = x + b
    return x

# Use the decorator to make `outer_function` a `Function`.
@tf.function
def outer_function(x):
    y = tf.constant([ [2.], [3.] ])
    b = tf.constant(4.)
    return inner_function(x, y, b)

# Note that the callable will create a graph that
# includes `inner_function` as well as `outer_function`.
print(outer_function(tf.constant([[1.0, 2.0]])).numpy())

'''
Converting Python functions to graphs
Any function you write with TensorFlow will contain a mixture of built-in TF operations and Python logic, such as if-then clauses, 
loops, break, return, continue, and more. While TensorFlow operations are easily captured by a tf.Graph, 
Python-specific logic needs to undergo an extra step in order to become part of the graph. tf.function uses a library called AutoGraph (tf.autograph) 
to convert Python code into graph-generating code.
'''

def simple_relu(x):
    if tf.greater(x, 0):
        return x
    else:
        return 0

# `tf_simple_relu` is a TensorFlow `Function` that wraps `simple_relu`.
tf_simple_relu = tf.function(simple_relu)
print('First branch, with graph', tf_simple_relu(tf.constant(1)).numpy())
print('First branch, with graph', tf_simple_relu(tf.constant(-1)).numpy())

# This is the graph-generating output of AutoGraph.
print(tf.autograph.to_code(simple_relu))

# This is the graph itself.
print(tf_simple_relu.get_concrete_function(tf.constant(1)).graph.as_graph_def())