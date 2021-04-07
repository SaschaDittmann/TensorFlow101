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

# BEGIN - ADD YOUR CODE HERE
d = tf.add(a,b) # 7
c = tf.multiply(a,b) # 10

f = tf.add(c,d) # 17
e = tf.subtract(c,d) # 3

g = tf.divide(f,e) # 5,66666
# END - ADD YOUR CODE HERE

print("a = %.2f\nb = %.2f\nc = %.2f\nd = %.2f\ne = %.2f\nf = %.2f\ng = %.2f" % (a, b, c, d, e, f, g))


# **********************
# ***  Exercise 1-B  ***
# **********************

a = tf.constant(5.0) 
b = tf.constant(2.0)

# BEGIN - ADD YOUR CODE HERE
c = tf.multiply(a,b) # 10
d = tf.sin(c) # -0.54402111088

e = tf.divide(d,b) # -0.272010564804
# END - ADD YOUR CODE HERE

print("a = %.2f\nb = %.2f\nc = %.2f\nd = %.2f\ne = %.2f" % (a, b, c, d, e))
