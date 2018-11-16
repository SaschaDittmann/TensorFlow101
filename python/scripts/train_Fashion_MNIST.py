from __future__ import print_function
import keras
from keras.datasets import fashion_mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow as tf
from azureml.core.run import Run
import os
import shutil

os.makedirs('./outputs', exist_ok=True)

run = Run.get_context()

# Number of classes - do not change unless the data changes
num_classes = 10

# sizes of batch and # of epochs of data
batch_size = 128
epochs = 24

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

#   Deal with format issues between different backends.  Some put the # of channels in the image before the width and height of image.
if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = X_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

#   Type convert and scale the test and training data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape (after reshape):', x_train.shape)
print('x_test shape (after reshape):', x_test.shape)

print("Before:\n{}".format(y_train[:4]))
# convert class vectors to binary class matrices.  One-hot encoding
#  3 => 0 0 0 1 0 0 0 0 0 0 and 1 => 0 1 0 0 0 0 0 0 0 0 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("After:\n{}".format(y_train[:4]))  # verify one-hot encoding

# Define the model
image_input = tf.keras.Input(shape=input_shape, name='input_layer')

# Some convolutional layers
conv_1 = tf.keras.layers.Conv2D(32,
                                kernel_size=(3, 3),
                                activation='relu')(image_input)
pool_1 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_1)
conv_2 = tf.keras.layers.Conv2D(64,
                                kernel_size=(3, 3),
                                activation='relu')(pool_1)
pool_2 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(conv_2)

# Flatten the output of the convolutional layers
drop_1 = tf.keras.layers.Dropout(0.25)(pool_2)
conv_flat = tf.keras.layers.Flatten()(drop_1)

# Some dense layers
fc_1 = tf.keras.layers.Dense(128, activation='relu')(conv_flat)
drop_2 = tf.keras.layers.Dropout(0.5)(fc_1)
classes_output = tf.keras.layers.Dense(num_classes, 
                             activation='softmax',
                             name='classes')(drop_2)

model = tf.keras.Model(inputs=image_input, outputs=[classes_output])

# Take a look at the model summary
model.summary()

#   define compile to minimize categorical loss, use ada delta optimized, and optimize to maximizing accuracy
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer='adam',
              metrics=['accuracy'])

#   Define callbacks
my_callbacks = [
    keras.callbacks.EarlyStopping(monitor='val_acc', patience=5, mode='max'), 
    keras.callbacks.ModelCheckpoint('./outputs/model.h5', verbose=1)
]

#   Train the model and test/validate the mode with the test data after each cycle (epoch) through the training data
#   Return history of loss and accuracy for each epoch
hist = model.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            callbacks=my_callbacks,
            validation_data=(x_test, y_test))
run.log_list('Training Loss', hist.history['loss'])
run.log_list('Training Accuracy', hist.history['acc'])
run.log_list('Validation Accuracy', hist.history['val_acc'])

#   Evaluate the model with the test data to get the scores on "real" data.
score = model.evaluate(x_test, y_test, verbose=0)
run.log('loss', score[0])
run.log('accuracy', score[1])
print('loss is {0:.2f}, and accuracy is {1:0.2f}'.format(score[0], score[1]))

#   Plot data to see relationships in training and validation data
import numpy as np
import matplotlib.pyplot as plt
epoch_list = list(range(1, len(hist.history['acc']) + 1))  # values for x axis [1, 2, ..., # of epochs]
plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
run.log_image(name='Accuracy', plot=plt)

# Use this only for export of the model.  
# This must come before the instantiation of ResNet50
K._LEARNING_PHASE = tf.constant(0)
K.set_learning_phase(0)

# The export path contains the name and the version of the model
model = tf.keras.models.load_model('./outputs/model.h5')
export_path = 'outputs/saved_model'
print('Exporting trained model to', export_path)
with K.get_session() as sess:
    # Import the libraries needed for saving models
    from tensorflow.python.saved_model import builder as saved_model_builder
    from tensorflow.python.saved_model import tag_constants, signature_constants, signature_def_utils_impl

    prediction_signature = tf.saved_model.signature_def_utils.predict_signature_def(
        {"image": model.input}, 
        {"prediction":model.output}
    )

    # export_path is a directory in which the model will be created
    builder = saved_model_builder.SavedModelBuilder(export_path)
    legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

    # Initialize global variables and the model
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    sess.run(init_op)

    # Add the meta_graph and the variables to the builder
    builder.add_meta_graph_and_variables(
        sess,
        [tag_constants.SERVING],
        signature_def_map={
            signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY: prediction_signature,
        },
    legacy_init_op=legacy_init_op)

    # save the graph
    builder.save()
    
    shutil.make_archive(export_path, 'zip', export_path)
    shutil.rmtree(export_path, ignore_errors=True)