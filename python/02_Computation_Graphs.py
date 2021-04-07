# Computation Graphs

import tensorflow as tf
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]="2"
print("Tensorflow version is " + str(tf.__version__))

# *********************
# ***    Example    ***
# *********************
a = tf.constant(5) 
b = tf.constant(2)
c = tf.constant(3)

d = tf.multiply(a,b) 
e = tf.add(c,b) 
f = tf.subtract(d,e)

print("f = {}".format(f))

# **********************
# ***  Exercise 1-A  ***
# **********************

a = tf.constant(5.0) 
b = tf.constant(2.0)

# ADD YOUR CODE HERE

print("a = %.2f\nb = %.2f\nc = %.2f\nd = %.2f\ne = %.2f\nf = %.2f\ng = %.2f" % (a, b, c, d, e, f, g))


# **********************
# ***  Exercise 1-B  ***
# **********************

a = tf.constant(5.0) 
b = tf.constant(2.0)

# ADD YOUR CODE HERE

print("a = %.2f\nb = %.2f\nc = %.2f\nd = %.2f\ne = %.2f" % (a, b, c, d, e))
