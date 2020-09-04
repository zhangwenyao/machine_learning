# -*- coding: utf-8 -*-

import os
import logging
import numpy
from sklearn import svm
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import tensorflow as tf
import sys

sys.path.append('..')
from common.iofile import loginit, PATH_R, PATH_DATA, PATH_LOG, save_model
from common.image_count import count1, count2

tf.get_logger().setLevel('ERROR')
# tf.logging.set_verbosity(tf.logging.ERROR)
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
PATH_CUR = os.path.dirname(os.path.abspath(__file__))
CURNAME = os.path.splitext(os.path.split(PATH_CUR)[0])[0]
logger = logging.getLogger(CURNAME)
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical

model = None
x = None
y = None
tx = None
ty = None


def train():
    global model, x, y, tx, ty
    model = svm.LinearSVC()
    # model = svm.SVC(kernel='linear')
    # model = svm.SVC(kernel='rbf')
    # model = svm.SVC(C=10, gamma=0.001, kernel="rbf")
    # model = svm.SVC(C=2.82842712475, gamma=0.00728932024638, kernel="rbf")
    model.fit(x, y)
    # from sklearn.model_selection import GridSearchCV
    # model_cv = GridSearchCV(estimator=model,
    #                         param_grid=hyper_params,
    #                         scoring='accuracy',
    #                         cv=folds,
    #                         verbose=1,
    #                         return_train_score=True)
    #
    # model_cv.fit(X_train, y_train) # fit the model


def verify():
    global x, y
    z = model.predict(tx)
    info('准确率: %s' % (numpy.sum(z == ty) / z.size))


def digits_do():
    global model, x, y, tx, ty
    # 1797, 8 * 8
    train_ratio = 0.8
    test_ratio = 0.2
    mnist = load_digits()
    x, tx, y, ty = train_test_split(
        mnist.data, mnist.target, train_size=train_ratio, test_size=test_ratio,
        random_state=1)
    train()
    name = 'digits_do'
    # save_model(os.join(PATH_DATA, 'mnist_svm_LinearSVC_%s_%s_%s' %
    #                    (name, train_ratio, test_ratio)))
    info('%s, train_ratio=%s, tratio=%s' % (name, train_ratio, test_ratio))
    verify()


def digits_count2():
    global model, x, y, tx, ty
    train_ratio = 0.8
    test_ratio = 0.2
    mnist = load_digits()
    x, tx, y, ty = train_test_split(
        mnist.data, mnist.target, train_size=train_ratio, test_size=test_ratio,
        random_state=1)
    m = 8
    n = 8
    x = [count2(_, m, n) for _ in x]
    tx = [count2(_, m, n) for _ in tx]
    train()
    name = 'digits_count2'
    # save_model(os.join(PATH_DATA, 'mnist_svm_LinearSVC_%s_%s_%s' %
    #                    (name, train_ratio, test_ratio)))
    info('%s, train_ratio=%s, tratio=%s' % (name, train_ratio, test_ratio))
    verify()


def test():
    pass


def main():
    # digits_do()
    # digits_count2()
    test()


if __name__ == '__main__':
    loginit(CURNAME, filename=os.path.join(PATH_LOG, 'digits.log'))
    # info(os.getcwd())
    main()
    # info('exit')
