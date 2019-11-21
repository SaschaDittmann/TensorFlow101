from __future__ import print_function
import warnings
warnings.filterwarnings('ignore')

import os
import numpy as np
from functools import partial
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K

import azureml.core
from azureml.core.run import Run

print("TensorFlow version:", tf.__version__)
print("Using GPU build:", tf.test.is_built_with_cuda())
print("Is GPU available:", tf.test.is_gpu_available())
print("Azure ML SDK version:", azureml.core.VERSION)

outputs_folder = './outputs'
os.makedirs(outputs_folder, exist_ok=True)

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
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
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

img_index = 1
plt.imsave('fashion.png', 1-x_train[img_index][:, :, 0], cmap='gray')
run.log_image('Fashion Sample', path='fashion.png')

print("Before:\n{}".format(y_train[:4]))
# convert class vectors to binary class matrices.  One-hot encoding
#  3 => 0 0 0 1 0 0 0 0 0 0 and 1 => 0 1 0 0 0 0 0 0 0 0 
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
print("After:\n{}".format(y_train[:4]))  # verify one-hot encoding

# Define the model
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Take a look at the model summary
model.summary()

#   define compile to minimize categorical loss, use ada delta optimized, and optimize to maximizing accuracy
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

#   Define callbacks
my_callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=5, mode='max'), 
    ModelCheckpoint('./outputs/checkpoint.h5', verbose=1)
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
run.log_list('Training Accuracy', hist.history['accuracy'])
run.log_list('Validation Accuracy', hist.history['val_accuracy'])

#   Evaluate the model with the test data to get the scores on "real" data.
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
run.log('loss', score[0])
run.log('accuracy', score[1])

#   Plot data to see relationships in training and validation data
epoch_list = list(range(1, len(hist.history['accuracy']) + 1))  # values for x axis [1, 2, ..., # of epochs]
plt.plot(epoch_list, hist.history['accuracy'], epoch_list, hist.history['val_accuracy'])
plt.legend(('Training Accuracy', 'Validation Accuracy'))
run.log_image(name='Accuracy', plot=plt)

keras_path = os.path.join(outputs_folder, "keras")
os.makedirs(keras_path, exist_ok=True)

print("Exporting Keras models to", keras_path)
with open(os.path.join(keras_path, "model.json"), 'w') as f:
    f.write(model.to_json())
model.save_weights(os.path.join(keras_path, 'model.h5'))

model.save(os.path.join(keras_path, 'full_model.h5'))
