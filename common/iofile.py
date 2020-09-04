#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import numpy
import gzip
import _pickle as pickle

PATH_CUR = os.path.dirname(os.path.abspath(__file__))
PATH_R = os.path.abspath(os.path.join(PATH_CUR, os.path.pardir))
PATH_DATA = os.path.join(PATH_R, 'data')
PATH_LOG = os.path.join(PATH_R, 'log')
CURNAME = os.path.splitext(os.path.split(PATH_CUR)[0])[0]

logger = logging.getLogger(CURNAME)
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical


# logging.basicConfig(level=logging.INFO,
#                     format='%(asctime)s %(pathname)s %(module)s %(funcName)s'
#                            + '(%(lineno)s) %(levelname)s: %(message)s',
#                     datefmt='%Y-%m-%dT%H:%M:%S')
# debug = logging.debug
# info = logging.info
# warning = logging.warning
# error = logging.error
# critical = logging.critical


def loginit(name=CURNAME, filename=os.path.join(PATH_LOG, 'log.log'),
            level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    file_handle = logging.FileHandler(filename, encoding="UTF-8")
    stream_handle = logging.StreamHandler()
    fmt = logging.Formatter(
        '%(asctime)s %(pathname)s %(module)s %(funcName)s'
        + '(%(lineno)s) %(levelname)s: %(message)s',
        datefmt='%Y-%m-%dT%H:%M:%S')
    file_handle.setFormatter(fmt)
    stream_handle.setFormatter(fmt)
    logger.addHandler(file_handle)
    logger.addHandler(stream_handle)


def savefile(file, data):
    with gzip.open(file + '.pkl.gz', 'wb') as f:
        pickle.dump(data, f)
    info('save to ' + file)


def readfile(file):
    with gzip.open(file + '.pkl.gz', 'rb') as f:
        data = pickle.load(f)
    info('read from ' + file)
    return data


def save(file, model=None, x=None, y=None,
         tx=None, ty=None, xy=False):
    save_model(file=file, model=model)
    if xy:
        save_xy(file=file, x=x, y=y, tx=tx, ty=ty)


def save_model(file, model):
    with gzip.open(file + '_model.pkl.gz', 'wb') as f:
        pickle.dump(model, f)
    info('save model to ' + file)


def save_xy(file, x, y, tx, ty):
    with gzip.open(file + '_x.pkl.gz', 'wb') as f:
        pickle.dump(x, f)
    with gzip.open(file + '_y.pkl.gz', 'wb') as f:
        pickle.dump(y, f)
    with gzip.open(file + '_tx.pkl.gz', 'wb') as f:
        pickle.dump(tx, f)
    with gzip.open(file + '_ty.pkl.gz', 'wb') as f:
        pickle.dump(ty, f)
    info('save xy to ' + file)


def read(file, xy=False):
    model = read_model(file)
    if xy:
        x, y, tx, ty = read_xy(file)
        return model, x, y, tx, ty
    else:
        return model


def read_model(file):
    with gzip.open(file + '_model.pkl.gz', 'rb') as f:
        model = pickle.load(f)
    info('read model from ' + file)
    return model


def read_xy(file):
    with gzip.open(file + '_x.pkl.gz', 'rb') as f:
        x = pickle.load(f)
    with gzip.open(file + '_y.pkl.gz', 'rb') as f:
        y = pickle.load(f)
    with gzip.open(file + '_testx.pkl.gz', 'rb') as f:
        tx = pickle.load(f)
    with gzip.open(file + '_testy.pkl.gz', 'rb') as f:
        ty = pickle.load(f)
    info('read xy from ' + file)
    return x, y, tx, ty


def readbyte(bytestream, n=1, t=numpy.uint8):
    dt = numpy.dtype(t).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(n), dtype=dt)[0]


def read32(bytestream):
    dt = numpy.dtype(numpy.uint32).newbyteorder('>')
    return numpy.frombuffer(bytestream.read(4), dtype=dt)[0]


def main():
    pass


if __name__ == "__main__":
    loginit(CURNAME)
    info(os.getcwd())
    main()
    info('exit')
