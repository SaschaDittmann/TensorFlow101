from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import os
import math
import matplotlib.pyplot as plt

# print version
print("Tensorflow version is " + str(tf.__version__))

project_dir = './outputs/mnist-with-summaries'
os.makedirs(project_dir, exist_ok=True)

from tensorflow.examples.tutorials.mnist import input_data

# Supress warning and informational messages
old_v = tf.logging.get_verbosity()
tf.logging.set_verbosity(tf.logging.ERROR)

mnist = input_data.read_data_sets("./MNIST_data/")

# Restore warning and informational messages
tf.logging.set_verbosity(old_v)

print('images shape:', mnist.train.images.shape)
print('labels shape:', mnist.train.labels)

def weight_variable(shape):
    """Create a weight variable with appropriate initialization."""
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    """Create a bias variable with appropriate initialization."""
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

def nn_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.relu):
    """Reusable code for making a simple neural net layer.

    It does a matrix multiply, bias add, and then uses ReLU to nonlinearize.
    It also sets up name scoping so that the resultant graph is easy to read,
    and adds a number of summary ops.
    """
    # Adding a name scope ensures logical grouping of the layers in the graph.
    with tf.name_scope(layer_name):
        # This Variable will hold the state of the weights for the layer
        with tf.name_scope('weights'):
            weights = weight_variable([input_dim, output_dim])
            variable_summaries(weights)
        with tf.name_scope('biases'):
            biases = bias_variable([output_dim])
            variable_summaries(biases)
        with tf.name_scope('Wx_plus_b'):
            preactivate = tf.matmul(input_tensor, weights) + biases
            tf.summary.histogram('pre_activations', preactivate)
        activations = act(preactivate, name='activation')
        tf.summary.histogram('activations', activations)
        return activations

learning_rate = 0.001

# Input placeholders
with tf.name_scope('input'):
    x = tf.placeholder(tf.float32, [None, 784], name='x-input')
    y_ = tf.placeholder(tf.int64, [None], name='y-input')

with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, [-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10)

hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('dropout_keep_probability', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

# Do not apply softmax activation yet, see below.
y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

with tf.name_scope('cross_entropy'):
    # The raw formulation of cross-entropy,
    #
    # tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.softmax(y)),
    #                               reduction_indices=[1]))
    #
    # can be numerically unstable.
    #
    # So here we use tf.losses.sparse_softmax_cross_entropy on the
    # raw logit outputs of the nn_layer above, and then average across
    # the batch.
    with tf.name_scope('total'):
        cross_entropy = tf.losses.sparse_softmax_cross_entropy(labels=y_, logits=y)

tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('train'):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), y_)
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
tf.summary.scalar('accuracy', accuracy)

# Merge all the summaries and write them out to
merged = tf.summary.merge_all()

# Train the model, and also write summaries.
# Every 10th step, measure test-set accuracy, and write test summaries
# All other steps, run train_step on training data, & add training summaries
max_steps = 1000
logs_dir = os.path.join(project_dir, "logs")
os.environ["LOGS_DIR"] = logs_dir

if tf.gfile.Exists(logs_dir):
    tf.gfile.DeleteRecursively(logs_dir)
tf.gfile.MakeDirs(logs_dir)

# initialize the global variables
init = tf.global_variables_initializer()

with tf.Session() as sess:
    train_writer = tf.summary.FileWriter(logs_dir + '/train', sess.graph)
    test_writer = tf.summary.FileWriter(logs_dir + '/test')
    
    # perform the initialization which is only the initialization of all global variables
    sess.run(init)
    
    for i in range(max_steps):
        if i % 10 == 0:  # Record summaries and test-set accuracy
            xs, ys = mnist.train.next_batch(100)
            summary, acc = sess.run(
                [merged, accuracy], 
                feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}
            )
            test_writer.add_summary(summary, i)
            print('Accuracy at step %s: %s' % (i, acc))
        else:  # Record train set summaries, and train
            batch_xs, batch_ys = mnist.train.next_batch(100)
            if i % 100 == 99:  # Record execution stats
                run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                summary, _ = sess.run([merged, train_step],
                    feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.9},
                    options=run_options,
                    run_metadata=run_metadata)
                train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
                train_writer.add_summary(summary, i)
                print('Adding run metadata for', i)
            else:  # Record a summary
                summary, _ = sess.run(
                    [merged, train_step], 
                    feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.9}
                )
                train_writer.add_summary(summary, i)
    train_writer.close()
    test_writer.close()
