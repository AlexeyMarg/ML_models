import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

inputs = keras.Input(shape=(784,))
print('Input layer: ', inputs.shape, inputs.dtype)

# You create a new node in the graph of layers by calling a layer on this inputs object:
dense = layers.Dense(64, activation='relu')
x = dense(inputs)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(10)(x)

model = keras.Model(inputs=inputs, outputs=outputs, name='mnist_model')
model.summary()

#keras.utils.plot_model(model, 'mnist_model.png')