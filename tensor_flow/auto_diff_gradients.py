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

