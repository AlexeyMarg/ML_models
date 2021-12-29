import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


'''
TensorFlow provides the tf.GradientTape API for automatic differentiation; that is, computing the gradient of a 
computation with respect to some inputs, usually tf.Variables. TensorFlow "records" relevant operations executed inside 
the context of a tf.GradientTape onto a "tape". TensorFlow then uses that tape to compute the gradients of a "recorded" 
computation using reverse mode differentiation.
'''
x = tf.Variable(3.0)
with tf.GradientTape() as tape:
    y = x ** 2

'''
Once you've recorded some operations, use GradientTape.gradient(target, sources) to calculate the gradient of some 
target (often a loss) relative to some source (often the model's variables):
'''

# dy = 2x * dx
dy_dx = tape.gradient(y, x)
print(dy_dx.numpy())



'''
#The above example uses scalars, but tf.GradientTape works as easily on any tensor:
'''
w = tf.Variable(tf.random.normal( (3, 2) ), name='w')
print(w)
b = tf.Variable(tf.zeros( (2) ), dtype=tf.float32, name='b')
print(b)
x = [[1, 2, 3]]

with tf.GradientTape(persistent=True) as tape:
    y = x @ w + b
    loss = tf.math.reduce_min(y ** 2)

[dl_dw, dl_db] = tape.gradient(loss, [w, b])
print(w.shape)
print(dl_dw.shape)
#or
my_vars = {
    'w': w,
    'b': b
}
grad = tape.gradient(loss, my_vars)
print(grad['b'])

'''
#Gradient with respect to a model
'''

layer = tf.keras.layers.Dense(2, activation='relu')
x = tf.constant( [ [1., 2., 3.] ])
with tf.GradientTape() as tape:
    #Forward pass
    y = layer(x)
    loss = tf.reduce_mean(y ** 2)

# Calculate gradients with respect to every trainable variable
grad = tape.gradient(loss, layer.trainable_variables)

for var, g in zip(layer.trainable_variables, grad):
    print(f'{var.name}, shape: {g.shape}')


'''
Controlling what the tape watches
The default behavior is to record all operations after accessing a trainable tf.Variable. The reasons for this are:

The tape needs to know which operations to record in the forward pass to calculate the gradients in the backwards pass.
The tape holds references to intermediate outputs, so you don't want to record unnecessary operations.
The most common use case involves calculating the gradient of a loss with respect to all a model's trainable variables.
For example, the following fails to calculate a gradient because the tf.Tensor is not "watched" by default, and the tf.Variable is not trainable:
'''
# A trainable variable
x0 = tf.Variable(3.0, name='x0')
# Not trainable
x1 = tf.Variable(3.0, name='x1', trainable=False)
# Not a Variable: A variable + tensor returns a tensor.
x2 = tf.Variable(2.0, name='x2') + 1.0
# Not a variable
x3 = tf.constant(3.0, name='x3')

with tf.GradientTape() as tape:
    y = (x0 ** 2) + (x1 ** 2) + (x2 ** 2)

grad = tape.gradient(y, [x0, x1, x2, x3])
for g in grad:
    print(g)

for g in [var.name for var in tape.watched_variables()]:
    print(g)

# To record gradients with respect to a tf.Tensor, you need to call GradientTape.watch(x):

x = tf.constant(3.0, name='x')
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 2
print(tape.gradient(y, x))

# Conversely, to disable the default behavior of watching all tf.Variables, set watch_accessed_variables=False
# when creating the gradient tape. This calculation uses two variables, but only connects the gradient for one of the variables:


x0 = tf.Variable(0.0)
x1 = tf.Variable(10.0)

with tf.GradientTape(watch_accessed_variables=False) as tape:
    tape.watch(x1)
    y0 = tf.math.sin(x0)
    y1 = tf.nn.softplus(x1)
    y = y0 + y1
    ys = tf.reduce_sum(y)

# Since GradientTape.watch was not called on x0, no gradient is computed with respect to it:
# dys/dx1 = exp(x1) / (1 + exp(x1)) = sigmoid(x1)
grad = tape.gradient(ys, {'x0': x0, 'x1': x1})

print('dy/dx0:', grad['x0'])
print('dy/dx1:', grad['x1'].numpy())

'''
Intermediate results
You can also request gradients of the output with respect to intermediate values computed inside the tf.GradientTape context.
'''

x = tf.constant(3.0, name='x')
with tf.GradientTape() as tape:
    tape.watch(x)
    y = x ** 2
    z = y ** 2

# Use the tape to compute the gradient of z with respect to the
# intermediate value y.
# dz_dy = 2 * y and y = x ** 2 = 9
print(tape.gradient(z, y).numpy())

#By default, the resources held by a GradientTape are released as soon as the GradientTape.gradient method is called.
# To compute multiple gradients over the same computation, create a gradient tape with persistent=True.
# This allows multiple calls to the gradient method as resources are released when the tape object is garbage collected. For example:

x = tf.constant([1, 3.0])
with tf.GradientTape(persistent=True) as tape:
    tape.watch(x)
    y = x ** 2
    z = y ** 2

print(tape.gradient(z, x).numpy())
print(tape.gradient(y, x).numpy())