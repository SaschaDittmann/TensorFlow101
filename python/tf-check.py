# Verify that Tensorflow is working
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# print version
print("Tensorflow version is " + str(tf.__version__))

# verify session works
hello = tf.constant('Hello from Tensorflow')
sess = tf.Session()
print(sess.run(hello))
