# Based on MNIST CNN from Keras' examples: https://github.com/keras-team/keras/blob/master/examples/mnist_cnn.py (MIT License)

import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras import backend as K
import numpy as np
import os

RANDOM_SEED = 12345
batch_size = 128
NUM_CLASSES = 10
epochs = 12

# input image dimensions
img_rows, img_cols = 28, 28
IMAGE_SHAPE = (img_rows, img_cols)
if K.image_data_format() == 'channels_first':
    INPUT_SHAPE = (1, img_rows, img_cols)
else:
    INPUT_SHAPE = (img_rows, img_cols, 1)


def reset_seed(seed=RANDOM_SEED):
    np.random.seed(seed)
    tf.compat.v1.set_random_seed(seed)


def load_dataset(basedir=None):
    def load(f):
        if basedir:
            f = os.path.join(basedir, f)
        return np.load(f)['arr_0']

    # Load the data
    name = 'kmnist'
    x_train = load(name + '-train-imgs.npz')
    x_test = load(name + '-test-imgs.npz')
    y_train = load(name + '-train-labels.npz')
    y_test = load(name + '-test-labels.npz')

    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255

    print('{} train samples, {} test samples'.format(len(x_train), len(x_test)))

    # # Convert class vectors to binary class matrices
    # y_train = keras.utils.to_categorical(y_train, num_classes)
    # y_test = keras.utils.to_categorical(y_test, num_classes)

    return x_train, y_train, x_test, y_test


def create_model(num_classes=NUM_CLASSES):
    if True:
        # ======== Sample model 1 ========
        # seed=12345, train_acc=0.99825, test_acc=0.9169
        layers = [
            tf.keras.layers.Flatten(input_shape=IMAGE_SHAPE),
            tf.keras.layers.Dense(512, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax),
        ]
    elif True:
        # ======== Sample model 2 ========
        # seed=12345, train_acc=0.99935, test_acc=0.9529
        layers = [
            tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation=tf.nn.relu, input_shape=INPUT_SHAPE),
            tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation=tf.nn.relu),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
            tf.keras.layers.Dropout(0.25),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation=tf.nn.relu),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(num_classes, activation=tf.nn.softmax),
        ]
    model = tf.keras.models.Sequential(layers)
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )
    return model


def main():
    reset_seed()

    x_train, y_train, x_test, y_test = load_dataset()

    model = create_model()

    input_shape = model.input_shape[1:]
    if x_train.shape[1:] != input_shape:
        x_train = x_train.reshape((-1, *input_shape))
        x_test = x_test.reshape((-1, *input_shape))

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    cp_basedir = 'ckpts-k10-{}-{}'.format(timestamp, os.getpid())
    os.makedirs(cp_basedir)
    cp_path = os.path.join(cp_basedir, 'weights.{epoch:02d}-{val_loss:.2f}.hdf5')
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=cp_path,
        monitor='val_loss',
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        period=1,
    )

    model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        verbose=1,
        validation_data=(x_test, y_test),
        callbacks=[cp_callback],
    )

    train_score = model.evaluate(x_train, y_train, verbose=0)
    test_score = model.evaluate(x_test, y_test, verbose=0)
    print('Train loss:', train_score[0])
    print('Train accuracy:', train_score[1])
    print('Test loss:', test_score[0])
    print('Test accuracy:', test_score[1])


if __name__ == '__main__':
    main()
