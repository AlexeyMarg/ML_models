import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

'''
When passing data to the built-in training loops of a model, you should either use NumPy arrays 
(if your data is small and fits in memory) or tf.data Dataset objects. In the next few paragraphs, 
we'll use the MNIST dataset as NumPy arrays, in order to demonstrate how to use optimizers, losses, and metrics.
'''

def get_uncompiled_model():
    inputs = layers.Input(shape=(784,), name='digits')
    x = layers.Dense(64, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(64, activation='relu', name='dense_2')(inputs)
    outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

def get_compiled_model():
    model = get_uncompiled_model()
    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.SparseCategoricalAccuracy()]
    )
    return model

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocessing
x_train = x_train.reshape(60000, 784).astype('float32') /255
x_test = x_test.reshape(10000, 784).astype('float32') /255
y_train = y_train.astype('float32')
y_test = y_test.astype('float32')

# Reserve data for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]


model = get_compiled_model()

print("Fit model on training data")
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=2,
    # We pass some validation for
    # monitoring validation loss and metrics
    # at the end of each epoch
    validation_data=(x_val, y_val),
)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x_test)
print("predictions shape:", predictions.shape)
test_scores = model.evaluate(x_test, y_test)
print('Test scores: ', test_scores)

'''
Compile parameters:

Optimizers:

SGD() (with or without momentum)
RMSprop()
Adam()
etc.
Losses:

MeanSquaredError()
KLDivergence()
CosineSimilarity()
etc.
Metrics:

AUC()
Precision()
Recall()
etc.
'''

'''
Custom losses
If you need to create a custom loss, Keras provides two ways to do so.
The first method involves creating a function that accepts inputs y_true and y_pred. 
The following example shows a loss function that computes the mean squared error between the real data and the predictions:
'''

def custom_mean_squared_error(y_true, y_pred):
    return tf.math.reduce_mean(tf.square(y_true - y_pred))

model = get_uncompiled_model()
model.compile(optimizer=keras.optimizers.Adam(),
              loss=custom_mean_squared_error)
y_train_one_hot = tf.one_hot(y_train, depth=10)
model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)
