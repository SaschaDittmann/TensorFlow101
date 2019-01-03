from __future__ import print_function
import os
import numpy as np
import random
from functools import partial
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import backend as K

from azureml.core import Experiment, Run, Workspace
import azureml.core

# Check core SDK version number
print("SDK version:", azureml.core.VERSION)

# Create project directory
outputs_folder = './tmp/fashion_mnist_local'
os.makedirs(outputs_folder, exist_ok=True)

ws = Workspace.from_config()
print('Workspace name: ' + ws.name, 
      'Azure region: ' + ws.location, 
      'Resource group: ' + ws.resource_group, sep='\n')

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

# Label Description 
label_dict = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

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
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

experiment_name = 'fashion-mnist'
experiment = Experiment(workspace=ws, name=experiment_name)

#   Define early stopping callback
my_callbacks = [EarlyStopping(monitor='val_acc', patience=5, mode='max')]

os.makedirs('./outputs', exist_ok=True)

try:
    run = experiment.start_logging()
    run.tag("Description","Locally trained Fashion MNIST model")

    #   Train the model and test/validate the mode with the test data after each cycle (epoch) through the training data
    #   Return history of loss and accuracy for each epoch
    hist = model.fit(x_train, y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        callbacks=my_callbacks,
        validation_data=(x_test, y_test))
    run.log_list('Training Accuracy', hist.history['acc'])
    run.log_list('Validation Accuracy', hist.history['val_acc'])

    #   Evaluate the model with the test data to get the scores on "real" data.
    score = model.evaluate(x_test, y_test, verbose=0)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])
    run.log('loss', score[0])
    run.log('accuracy', score[1])

    #   Plot data to see relationships in training and validation data
    import numpy as np
    import matplotlib.pyplot as plt
    epoch_list = list(range(1, len(hist.history['acc']) + 1))  # values for x axis [1, 2, ..., # of epochs]
    plt.plot(epoch_list, hist.history['acc'], epoch_list, hist.history['val_acc'])

    plt.legend(('Training Accuracy', 'Validation Accuracy'))
    run.log_image(name='Accuracy', plot=plt)

    run.complete()
except Exception as e:
    run.fail()
    print(str(e))

# Look at some predictions
def run_prediction(idx):
    result = np.argmax(model.predict(x_test[idx:idx+1]))
    label = np.argmax(y_test[idx])
    print('Prediction: {} ({})'.format(result, label_dict[result]))
    print('Label: {} ({})'.format(label, label_dict[label]))
    #plt.imshow(1-x_test[idx][:, :, 0], cmap='gray')

for _ in range(1,10):
    idx = random.randint(0, 47-1)
    run_prediction(idx)

# Export model to TF SavedModel for CMLE Prediction
# https://www.tensorflow.org/api_docs/python/tf/keras/estimator/model_to_estimator
model_dir = os.path.join(outputs_folder, "estimator_model")

# First, convert Keras Model to TensorFlow Estimator
model_input_name = model.input_names[0]
estimator_model = keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir)

# Export the TensorFlow Estimator to SavedModel
def serving_input_receiver_fn():
    input_ph = tf.placeholder(tf.string, shape=[None], name='image_binary')
    base64 = tf.io.decode_base64(input_ph)
    images = tf.map_fn(partial(tf.image.decode_image, channels=1), base64, dtype=tf.uint8)
    images = tf.cast(images, tf.float32) / 255.
    images.set_shape([None, 28, 28, 1])

    # the first key is the name of first layer of the (keras) model. 
    # The second key is the name of the key that will be passed in the prediction request
    return tf.estimator.export.ServingInputReceiver({model_input_name: images}, {'bytes': input_ph})

print("Exporting TensorFlow model...")
export_path = estimator_model.export_savedmodel(
    os.path.join(outputs_folder, "saved_model"), 
    serving_input_receiver_fn=serving_input_receiver_fn
)

# Keras exports
keras_path = os.path.join(outputs_folder, "keras")
os.makedirs(keras_path, exist_ok=True)

print("Exporting Keras model...")
keras_model_json = os.path.join(keras_path, "model.json")
keras_model_weights = os.path.join(keras_path, 'model.h5')
keras_full_model = os.path.join(keras_path, 'full_model.h5')
with open(keras_model_json, 'w') as f:
    f.write(model.to_json())
model.save_weights(keras_model_weights)
model.save(keras_full_model)

print("Uploading output...")
for root, dirs, files in os.walk(outputs_folder, topdown=False):
    for filename in files:
        source = os.path.join(root, filename)
        name = source.replace(outputs_folder, "./outputs")
        print("Uploading " + filename)
        run.upload_file(name, source)
