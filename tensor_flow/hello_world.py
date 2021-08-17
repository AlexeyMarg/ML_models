import tensorflow as tf
tf.compat.v1.disable_eager_execution()

x = tf.compat.v1.Variable(3, name='x')
y = tf.compat.v1.Variable(4, name='x')
f = x * x * y + y + 2

with tf.compat.v1.Session() as sess:
    x.initializer.run()
    y.initializer.run()
    result = sess.run(f)
    print(result)

init = tf.compat.v1.global_variables_initializer()

with tf.compat.v1.Session() as sess:
    init.run()
    result = sess.run(f)
    print(result)