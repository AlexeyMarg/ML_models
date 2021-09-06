import tensorflow as tf
import matplotlib.pyplot as plt

'''
Linear noised data preparation
'''

TRUE_W = 3.
TRUE_B = 2.

N_SAMPLES = 1000

# A vector of random x values
x = tf.random.normal(shape=[N_SAMPLES])
# Generate some noise
noise = tf.random.normal(shape=[N_SAMPLES])

# Calculate y
y = x * TRUE_W + TRUE_B + noise

plt.scatter(x, y, c='b')
#plt.show()

class MyModel(tf.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Initialize the weights to `5.0` and the bias to `0.0`
        # In practice, these should be randomly initialized
        self.w = tf.Variable(5.)
        self.b = tf.Variable(0.)

    def __call__(self, x):
        return x * self.w + self.b

model = MyModel()

# List the variables tf.modules's built-in variable aggregation.
print("Variables:", model.variables)

# Verify the model works
assert model(3.0).numpy() == 15.0

'''
Define a loss function
'''

def loss(y_targer, y_predicted):
    return tf.reduce_mean(tf.square(y_targer - y_predicted))

print("Current loss: %1.6f" % loss(y, model(x)).numpy())

plt.scatter(x, y, c="b")
plt.scatter(x, model(x), c="r")
#plt.show()

'''
Train loop
'''
def train(model, x, y, learning_rate):
    with tf.GradientTape() as tape:
        current_loss = loss(y, model(x))

    dw, db = tape.gradient(current_loss, [model.w, model.b])

    model.w.assign_sub(learning_rate * dw)
    model.b.assign_sub(learning_rate * db)

w_story, b_story = [], []
n_epochs = 10

def training_loop(model, x, y):
    for epoch in range(n_epochs):
        train(model, x, y, learning_rate=0.1)
        w_story.append(model.w.numpy())
        b_story.append((model.b.numpy()))
        current_loss = loss(y, model(x))
        print("Epoch %2d: W=%1.2f b=%1.2f, loss=%2.5f" %
              (epoch, w_story[-1], b_story[-1], current_loss))

print("Starting: W=%1.2f b=%1.2f, loss=%2.5f" % (model.w, model.b, loss(y, model(x))))
model = MyModel()
training_loop(model, x, y)

plt.plot(range(n_epochs), w_story, "r",
         range(n_epochs), b_story, "b")

plt.plot([TRUE_W] * n_epochs, "r--",
         [TRUE_B] * n_epochs, "b--")

plt.legend(["w", "b", "True W", "True b"])
#plt.show()

plt.scatter(x, y, c="b")
plt.scatter(x, model(x), c="r")
plt.show()

print("Current loss: %1.6f" % loss(model(x), y).numpy())


