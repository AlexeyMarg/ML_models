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

'''
If you need a loss function that takes in parameters beside y_true and y_pred, you can subclass the tf.keras.losses.Loss class and implement the following two methods:

__init__(self): accept parameters to pass during the call of your loss function
call(self, y_true, y_pred): use the targets (y_true) and the model predictions (y_pred) to compute the model's loss
Let's say you want to use mean squared error, but with an added term that will de-incentivize prediction values far from 0.5 
(we assume that the categorical targets are one-hot encoded and take values between 0 and 1). 
This creates an incentive for the model not to be too confident, which may help reduce overfitting (we won't know if it works until we try!).

Here's how you would do it:
'''
class CustomMSE(keras.losses.Loss):
    def __init__(self, regularization_factor=0.1, name="custom_mse"):
        super().__init__(name=name)
        self.regularization_factor = regularization_factor

    def call(self, y_true, y_pred):
        mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
        reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
        return mse + reg * self.regularization_factor


model = get_uncompiled_model()
model.compile(optimizer=keras.optimizers.Adam(), loss=CustomMSE())

y_train_one_hot = tf.one_hot(y_train, depth=10)
model.fit(x_train, y_train_one_hot, batch_size=64, epochs=1)

'''
Custom metrics
If you need a metric that isn't part of the API, you can easily create custom metrics by subclassing the tf.keras.metrics.Metric class. You will need to implement 4 methods:

__init__(self), in which you will create state variables for your metric.
update_state(self, y_true, y_pred, sample_weight=None), which uses the targets y_true and the model predictions y_pred to update the state variables.
result(self), which uses the state variables to compute the final results.
reset_states(self), which reinitializes the state of the metric.
State update and results computation are kept separate (in update_state() and result(), respectively) because in some cases, the results computation might be very expensive and would only be done periodically.

Here's a simple example showing how to implement a CategoricalTruePositives metric that counts how many samples were correctly classified as belonging to a given class:
'''
class CategoricalTruePositives(keras.metrics.Metric):
    def __init__(self, name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        values = tf.cast(values, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)


model = get_uncompiled_model()
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[CategoricalTruePositives()],
)
model.fit(x_train, y_train, batch_size=64, epochs=3)
