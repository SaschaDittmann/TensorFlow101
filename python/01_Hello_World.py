# Verify that Tensorflow is working
import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"

# print version
print("Tensorflow version is " + str(tf.__version__))

ph = "Hello"
pw = " World!"
phw = ph + pw

print(phw)

h = tf.constant("Hello")
w = tf.constant(" World!")
hw = h + w

print(hw)

with tf.Session() as sess:
    ans = sess.run(hw)

print(ans)