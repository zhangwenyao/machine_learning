import os
import logging
import numpy
import pandas
import matplotlib.pyplot as plt
import tensorflow as tf

tf.get_logger().setLevel('ERROR')
# tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
print("Tensorflow version " + tf.__version__)
KAGGLE = None

shape = (28, 28, 1)
if KAGGLE:
    mnist_dir = '/kaggle/input/digit-recognizer'
    res_file = 'submission.csv'
    CURNAME = 'kaggle'
    logger = logging.getLogger(CURNAME)
    logfile = 'log.log'

else:
    import sys

    sys.path.append('..')
    from common.iofile import loginit, PATH_R, PATH_DATA, PATH_LOG, save_model
    from common.image_count import count1, count2
    from common.mnist_base import read_mnist_data

    model_name = 'mnist_keras'
    mnist_dir = 'MNIST_data'
    res_file = 'submission.csv'

    PATH_CUR = os.path.dirname(os.path.abspath(__file__))
    CURNAME = os.path.splitext(os.path.split(PATH_CUR)[0])[0]
    logger = logging.getLogger(CURNAME)
    logfile = os.path.join(PATH_LOG, 'kaggle.log')

    mnist_dir = os.path.join(PATH_DATA, mnist_dir)
    PATH_DATA = os.path.join(PATH_DATA, 'kaggle')
    res_file = os.path.join(PATH_DATA, 'mnist_' + res_file)

logger.setLevel(logging.INFO)
file_handle = logging.FileHandler(logfile, encoding="UTF-8")
stream_handle = logging.StreamHandler()
fmt = logging.Formatter(
    '%(asctime)s %(pathname)s %(module)s %(funcName)s'
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

# load data
if KAGGLE:
    train = pandas.read_csv(os.path.join(mnist_dir, 'train.csv'))
    test = pandas.read_csv(os.path.join(mnist_dir, 'test.csv'))
else:
    import zipfile

    with zipfile.ZipFile(os.path.join(mnist_dir, 'train.csv.zip')) as zf:
        with zf.open('train.csv') as f:
            train = pandas.read_csv(f)
    with zipfile.ZipFile(os.path.join(mnist_dir, 'test.csv.zip')) as zf:
        with zf.open('test.csv') as f:
            test = pandas.read_csv(f)
# train.head()
y = train['label'].values.astype('float32')
x = train.drop(labels=['label'], axis=1)
x = x.astype('float32') / 255
x = x.values.reshape(len(x), 28, 28, 1)
x.shape
plt.imshow(x[1][:, :, 0])
test = test.astype('float32') / 255
test = test.values.reshape(len(test), 28, 28, 1)
test.shape


# Model
def create_model():
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                     input_shape=(28, 28, 1)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(10, activation="softmax"))
    model.compile(loss="sparse_categorical_crossentropy", optimizer="Adam",
                  metrics=['accuracy'])
    return model


if KAGGLE:
    # enable TPU
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Running on TPU ', tpu.master())
    except ValueError:
        tpu = None

    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        strategy = tf.distribute.get_strategy()
    print("REPLICAS: ", strategy.num_replicas_in_sync)
    with strategy.scope():
        model = create_model()
else:
    model = create_model()

# train
model.fit(x, y, epochs=10, steps_per_epoch=40, verbose=2)

# Prediction
test_predictions = model.predict(test)
results = numpy.argmax(test_predictions, axis=1)
results = pandas.Series(results, name="Label")

submission = pandas.concat(
    [pandas.Series(range(1, 1 + len(test)), name="ImageId"), results],
    axis=1)
submission.to_csv(res_file, index=False)
submission.head()
