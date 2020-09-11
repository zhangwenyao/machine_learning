# -*- coding: utf-8 -*-

import os
import logging
import gzip
import _pickle as pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import sys

sys.path.append("..")
from common.iofile import loginit, PATH_R, PATH_DATA, PATH_LOG

xy_name = 'mnist'
model_name = 'mnist_svm_LinearSVC'
# model_name = 'mnist_svm_SVC_rbf_C5_gamma0.05'
# mnist_dir = 'MNIST_xor_mid_data'
train_ratio = 1
test_ratio = 1
num_classes = 10
input_shape = (28, 28, 1)

CURNAME = os.path.abspath(__file__)
PATH_CUR = os.path.dirname(os.path.abspath(__file__))
PATH_DATA = os.path.join(PATH_R, 'data')
PATH_SRC = os.path.join(PATH_R, 'src')
PATH_LOG = os.path.join(PATH_R, 'log')
filename_xy = os.path.join(PATH_DATA, xy_name)
filename_model = os.path.join(PATH_DATA, model_name)

x0_train = None
y0_train = None
x0_test = None
y0_test = None
model = None
x_train = None
y_train = None
x_test = None
y_test = None

tf.get_logger().setLevel('ERROR')
# tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# logging.basicConfig(level=logging.INFO,
#                     filename=os.path.join(PATH_LOG, 'log.log'),
#                     filemode='a',
#                     format='%(asctime)s %(pathname)s %(module)s %(funcName)s'
#                            + '(%(lineno)s) %(levelname)s: %(message)s',
#                     datefmt='%Y-%m-%dT%H:%M:%S')
# debug = logging.debug
# info = logging.info
# warning = logging.warning
# error = logging.error
# critical = logging.critical
logger = logging.getLogger()
logger.setLevel(logging.INFO)
file_handle = logging.FileHandler(os.path.join(PATH_LOG, 'log.log'),
                                  encoding="UTF-8")
stream_handle = logging.StreamHandler()
fmt = logging.Formatter('%(asctime)s %(pathname)s %(module)s %(funcName)s'
                        + '(%(lineno)s) %(levelname)s: %(message)s',
                        datefmt='%Y-%m-%dT%H:%M:%S')
file_handle.setFormatter(fmt)
stream_handle.setFormatter(fmt)
logger.addHandler(file_handle)
logger.addHandler(stream_handle)
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical


def get_xy0_mnist():
    global x0_train, y0_train, x0_test, y0_test
    (x0_train, y0_train), (x0_test, y0_test) = keras.datasets.mnist.load_data()
    x0_train = x0_train.astype('float32') / 255
    x0_test = x0_test.astype('float32') / 255
    x0_train = np.expand_dims(x0_train, -1)
    x0_test = np.expand_dims(x0_test, -1)
    y0_train = keras.utils.to_categorical(y0_train, num_classes)
    y0_test = keras.utils.to_categorical(y0_test, num_classes)


def get_xy(train_ratio, test_ratio):
    global x0_train, y0_train, x0_test, y0_test, x_train, y_train, x_test, y_test
    train_num = int(len(y0_train) * train_ratio)
    test_num = int(len(y0_test) * test_ratio)
    x_train = x0_train[:train_num]
    y_train = y0_train[:train_num]
    x_test = x0_test[:test_num]
    y_test = y0_test[:test_num]


def train():
    global model, x_train, y_train, x_test, y_test
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.summary()

    batch_size = 128
    epochs = 15
    model.compile(loss="categorical_crossentropy", optimizer="adam",
                  metrics=["accuracy"])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
              validation_split=0.1)


def verify():
    global x_test, y_test, model
    score = model.evaluate(x_test, y_test, verbose=0)
    info("Test loss: %s, Test accuracy: %s" % (score[0], score[1]))


def save(file=os.path.join(PATH_DATA, 'tmp'), xy=False):
    save_model(file)
    if xy:
        save_xy(file)


def save_model(file):
    global model
    with gzip.open(file + '_model.pkl.gz', 'wb') as f:
        pickle.dump(model, f)
    info('save model to ' + file)


def save_xy(file):
    global x_train, y_train, x_test, y_test
    with gzip.open(file + '_x.pkl.gz', 'wb') as f:
        pickle.dump(x_train, f)
    with gzip.open(file + '_y.pkl.gz', 'wb') as f:
        pickle.dump(y_train, f)
    with gzip.open(file + '_testx.pkl.gz', 'wb') as f:
        pickle.dump(x_test, f)
    with gzip.open(file + '_testy.pkl.gz', 'wb') as f:
        pickle.dump(y_test, f)
    info('save xy to ' + file)


def read(file=os.path.join(PATH_DATA, 'tmp'), xy=False):
    read_model(file)
    if xy:
        read_xy(file)


def read_model(file):
    global model
    with gzip.open(file + '_model.pkl.gz', 'rb') as f:
        model = pickle.load(f)
    info('read model from ' + file)


def read_xy(file):
    global x0_train, y0_train, x0_test, y0_test, x_train, y_train, x_test, y_test
    with gzip.open(file + '_x.pkl.gz', 'rb') as f:
        x_train = x0_train = pickle.load(f)
    with gzip.open(file + '_y.pkl.gz', 'rb') as f:
        y_train = y0_train = pickle.load(f)
    with gzip.open(file + '_testx.pkl.gz', 'rb') as f:
        x_test = x0_test = pickle.load(f)
    with gzip.open(file + '_testy.pkl.gz', 'rb') as f:
        y_test = y0_test = pickle.load(f)
    info('read xy from ' + file)


def do(file_model=filename_model, file_xy=filename_xy,
       train_ratio=1, test_ratio=1,
       sm=True, sxy=False, rm=False, rxy=False):
    global filename, filename_read
    if rxy:
        read_xy(file_xy)
    else:
        get_xy0_mnist()
        get_xy(train_ratio, test_ratio)
        if sxy:
            save_xy(file_xy)
    if rm:
        read_model(file_model)
    else:
        train()
    verify()
    if sm:
        save_model(file_model)


if __name__ == '__main__':
    info(filename_model)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    do(file_xy=filename_xy, file_model=filename_model,
       train_ratio=train_ratio, test_ratio=test_ratio,
       rxy=False, rm=False, sxy=False, sm=False)
    info('exit')
