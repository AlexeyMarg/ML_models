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