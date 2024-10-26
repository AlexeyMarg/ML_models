import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

'''
A Sequential model is appropriate for a plain stack of layers where each layer has exactly one input tensor and one output tensor.

A Sequential model is not appropriate when:

Your model has multiple inputs or multiple outputs
Any of your layers has multiple inputs or multiple outputs
You need to do layer sharing
You want non-linear topology (e.g. a residual connection, a multi-branch model)
'''

my_model = keras.Sequential(
    [
        layers.Dense(2, activation='relu', name='layer1'),
        layers.Dense(3, activation='relu', name='layer2'),
        layers.Dense(4, name='layer3')
    ]
)

x = tf.ones((3, 3))
y = my_model(x)

print(my_model.layers)

# Other way to create mutilayer sequential model
model = keras.Sequential(name="my_sequential")
model.add(layers.Dense(2, activation="relu"))
model.add(layers.Dense(3, activation="relu"))
model.add(layers.Dense(4))
print(model.layers)
model.pop()
print(len(model.layers))

'''
Specify input form
'''

model = keras.Sequential(
    [
        layers.Input(shape=(4,)),
        layers.Dense(2, activation='relu')
    ]
)
model.summary()

#Other way
model = keras.Sequential()
model.add(layers.Dense(2, activation="relu", input_shape=(4,)))
model.summary()

'''
See inputs and outputs of layers
'''

initial_model = keras.Sequential(
    [
        keras.Input(shape=(250, 250, 3)),
        layers.Conv2D(32, 5, strides=2, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
        layers.Conv2D(32, 3, activation="relu"),
    ]
)
feature_extractor = keras.Model(
    inputs=initial_model.inputs,
    outputs=[layer.output for layer in initial_model.layers],
)

# Call feature extractor on test input.
x = tf.ones((1, 250, 250, 3))
features = feature_extractor(x)
