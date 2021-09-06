import tensorflow as tf
from datetime import datetime

'''
Most models are made of layers. Layers are functions with a known mathematical structure that can be reused and have trainable variables. 
In TensorFlow, most high-level implementations of layers and models, such as Keras or Sonnet, are built on the same foundational class: tf.Module.
Here's an example of a very simple tf.Module that operates on a scalar tensor:
'''

class SimpleModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)
        self.a_variable = tf.Variable(5.0, name="train_me")
        self.non_trainable_variable = tf.Variable(5.0, trainable=False, name="do_not_train_me")
    def __call__(self, x):
        return self.a_variable * x + self.non_trainable_variable

simple_module = SimpleModule(name='simple')
print(simple_module(tf.constant(5.)))
# All trainable variables
print("trainable variables:", simple_module.trainable_variables)
# Every variable
print("all variables:", simple_module.variables)

'''
Example of a two-layer linear layer model made out of modules.
'''
#First a dense (linear) layer:
class Dense(tf.Module):
    def __init__(self, in_features, out_features, name=None):
        super().__init__(name=name)
        self.w = tf.Variable( tf.random.normal([in_features, out_features]), name="w")
        self.b = tf.Variable( tf.zeros([out_features]), name="b")
    def __call__(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)

class SequentialModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = Dense(in_features=3, out_features=3)
        self.dense_2 = Dense(in_features=3, out_features=2)

    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)


my_model = SequentialModule(name="my_module")
print('Model results: ', my_model(tf.constant([ [2., 2., 2.] ])))
print('Submodules: ', my_model.submodules)
for var in my_model.variables:
    print(var)

'''
Example of a two-layer linear layer model made out of modules with unknown input dimension.
'''
print('\n\n\nExample of a two-layer linear layer model made out of modules with unknown input dimension')
class FlexibleDenseModule(tf.Module):
    def __init__(self, out_features, name=None):
        super().__init__(name=name)
        self.is_built = False
        self.out_features = out_features

    def __call__(self, x):
        if not self.is_built == True:
            self.w = tf.Variable(tf.random.normal( [x.shape[-1], self.out_features] ), name='w')
            self.b = tf.Variable(tf.zeros([self.out_features]), name='b')
            self.is_built = True

        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)

# Used in a module

class MyFlexibleSequentialModule(tf.Module):
    def __init__(self, name=None):
        super().__init__(name=name)

        self.dense_1 = FlexibleDenseModule(out_features=3)
        self.dense_2 = FlexibleDenseModule(out_features=2)

    def __call__(self, x):
        x = self.dense_1(x)
        return self.dense_2(x)

my_model = MyFlexibleSequentialModule(name="the_model")
print("Flexible Model results:", my_model(tf.constant([[2.0, 2.0, 2.0]])))

'''
Saving weights
You can save a tf.Module as both a checkpoint and a SavedModel.
Checkpoints are just the weights (that is, the values of the set of variables inside the module and its submodules):
'''
chk_path = 'my_checkpoint'
checkpoint = tf.train.Checkpoint(model=my_model)
checkpoint.write(chk_path)
print(tf.train.list_variables(chk_path))

new_model = MyFlexibleSequentialModule()
new_checkpoint = tf.train.Checkpoint(new_model)
new_checkpoint.restore('my_checkpoint')
print(new_model(tf.constant( [ [2.0, 2.0, 2.0] ] )))

'''
Saved model
'''

tf.saved_model.save(my_model, "the_saved_model")
new_model = tf.saved_model.load("the_saved_model")

'''
Keras layers
tf.keras.layers.Layer is the base class of all Keras layers, and it inherits from tf.Module.
You can convert a module into a Keras layer just by swapping out the parent and then changing __call__ to call:
'''

class MyDense(tf.keras.layers.Layer):
    # Adding **kwargs to support base Keras layer arguments
    def __init__(self, in_features, out_features, **kwargs):
        super().__init__(**kwargs)

        self.w = tf.Variable(tf.random.normal([in_features, out_features]), name='w')
        self.b = tf.Variable(tf.zeros([out_features]), name='b')

    def call(self, x):
        y = tf.matmul(x, self.w) + self.b
        return tf.nn.relu(y)

simple_layer = MyDense(name='simple', in_features=3, out_features=2)
print('Use of keras layer')
print(simple_layer([[2.0, 2.0, 2.0]]))
