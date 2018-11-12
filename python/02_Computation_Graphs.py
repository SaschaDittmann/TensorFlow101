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

sess = tf.Session() 
outs = sess.run(f) 
sess.close() 
print("outs = {}".format(outs))

# **********************
# ***  Exercise 1-A  ***
# **********************

a = tf.constant(5.0) 
b = tf.constant(2.0)

# ADD YOUR CODE HERE

with tf.Session() as sess:
   fetches = [a,b,c,d,e,f,g]
   outs = sess.run(fetches) 

print("outs = {}".format(outs))
print(type(outs[0]))


# **********************
# ***  Exercise 1-B  ***
# **********************

a = tf.constant(5.0) 
b = tf.constant(2.0)

# ADD YOUR CODE HERE

with tf.Session() as sess:
   fetches = [a,b,c,d,e]
   outs = sess.run(fetches) 

print("outs = {}".format(outs))
print(type(outs[0]))