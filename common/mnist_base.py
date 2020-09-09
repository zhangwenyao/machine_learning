# -*- coding: utf-8 -*-

import os
import logging
import numpy
import gzip
from common.iofile import loginit, read32, PATH_DATA

PATH_CUR = os.path.dirname(os.path.abspath(__file__))
CURNAME = os.path.splitext(os.path.split(PATH_CUR)[0])[0]
logger = logging.getLogger(CURNAME)
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical


def extract_mnist(file, reshape=False, shape=None):
    print('Extracting', file)
    with gzip.open(file, 'rb') as bytestream:
        x = read32(bytestream)
        if (x & 0xffff0000) != 0:
            raise ValueError(
                'Invalid magic number %d in MNIST image file: %s' % (x, file))
        tp = x >> 8
        if tp == 8:
            dtype = numpy.uint8
            l = 1
        elif tp == 9:
            dtype = numpy.int8
            l = 1
        elif tp == 0xB:
            dtype = numpy.dtype(numpy.int16).newbyteorder('>')
            l = 2
        elif tp == 0xC:
            dtype = numpy.dtype(numpy.int32).newbyteorder('>')
            l = 4
        elif tp == 0xD:
            dtype = numpy.float32
            l = 4
        elif tp == 0xE:
            dtype = numpy.float64
            l = 8
        else:
            raise ValueError(
                'Invalid type %s in MNIST image file: %s' % (tp, file))
        dim = x & 0xff
        if dim == 0:
            raise ValueError(
                'Invalid dim %d in MNIST image file: %s' % (dim, file))
        dims = [read32(bytestream) for _ in range(dim)]
        size = 1
        for _ in dims:
            size *= _
        buf = bytestream.read(size * l)
        data = numpy.frombuffer(buf, count=size, dtype=dtype)
        data = data.reshape(dims)
    if reshape:
        f = True
        sp = list(shape)
        if len(dims) != len(shape):
            f = False
        else:
            for i in range(len(shape)):
                if dims[i] != -1 and shape[i] != -1 and dims[i] != shape[i]:
                    f = False
                    break
        if f is False:
            id = -1
            for i in range(len(shape)):
                if shape[i] == -1:
                    sp[i] = dims[i]
                    size = size // dims[i]
                    id = i
            sp[id] *= size
            data = data.reshape(tuple(sp))
    return data


def read_mnist_data(train_ratio=1, test_ratio=1, normalize=False,
                    reshape=False, dir=os.path.join(PATH_DATA, 'MNIST_data')):
    TRAIN_IMAGES = os.path.join(dir, 'train-images-idx3-ubyte.gz')
    TRAIN_LABELS = os.path.join(dir, 'train-labels-idx1-ubyte.gz')
    TEST_IMAGES = os.path.join(dir, 't10k-images-idx3-ubyte.gz')
    TEST_LABELS = os.path.join(dir, 't10k-labels-idx1-ubyte.gz')
    train_images = extract_mnist(TRAIN_IMAGES, reshape=reshape, shape=(-1, -1))
    train_labels = extract_mnist(TRAIN_LABELS, reshape=reshape, shape=(-1,))
    test_images = extract_mnist(TEST_IMAGES, reshape=reshape, shape=(-1, -1))
    test_labels = extract_mnist(TEST_LABELS, reshape=reshape, shape=(-1,))
    train_num = round(len(train_labels) * train_ratio)
    test_num = round(len(test_labels) * test_ratio)
    x = train_images[:train_num]
    y = train_labels[:train_num]
    tx = test_images[:test_num]
    ty = test_labels[:test_num]
    if normalize:
        x = x.astype(numpy.float32)
        tx = tx.astype(numpy.float32)
        for i in range(train_num):
            x[i] = numpy.multiply(x[i], 1.0 / 255.0)
            # ma = max(x[i])
            # mi = min(x[i)
            # x[i] = (x[i]  - mi) / (ma - mi)
        for i in range(test_num):
            tx[i] = numpy.multiply(tx[i], 1.0 / 255.0)
            # ma = max(x[i])
            # mi = min(x[i)
            # tx[i] = (tx[i]  - mi) / (ma - mi)
    return x, y, tx, ty


def main():
    x, y, tx, ty = read_mnist_data()
    print(len(y), len(ty))


if __name__ == '__main__':
    loginit(CURNAME)
    info(os.getcwd())
    main()
    info('exit')
