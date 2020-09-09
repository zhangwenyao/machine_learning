# -*- coding: utf-8 -*-

import os
import logging
import numpy
from sklearn import svm
import sys

sys.path.append('..')
from common.iofile import loginit, PATH_R, PATH_DATA, PATH_LOG, save_model
from common.image_count import count1, count2, filter
from common.mnist_base import read_mnist_data

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


def mnist_origin():
    global model, x, y, tx, ty
    # 60000+10000, 28 * 28
    train_ratio = 1
    test_ratio = 1
    x, y, tx, ty = read_mnist_data(test_ratio, test_ratio, normalize=True,
                                   reshape=True)
    for i in range(len(x)):
        x[i] = filter(x[i], 0.1)
    for i in range(len(tx)):
        tx[i] = filter(tx[i], 0.1)
    train()
    name = sys._getframe().f_code.co_name
    # save_model(os.join(PATH_DATA, 'mnist_svm_LinearSVC_%s_%s' %
    #                    (train_ratio, test_ratio)))
    info('%s, train_ratio=%s, test_ratio=%s' % (name, train_ratio, test_ratio))
    verify()


def mnist_filter():
    global model, x, y, tx, ty
    # 60000+10000, 28 * 28
    train_ratio = 1
    test_ratio = 1
    x, y, tx, ty = read_mnist_data(test_ratio, test_ratio, normalize=True,
                                   reshape=True)
    for i in range(len(x)):
        x[i] = filter(x[i], 0.1)
    for i in range(len(tx)):
        tx[i] = filter(tx[i], 0.1)
    train()
    name = sys._getframe().f_code.co_name
    # save_model(os.join(PATH_DATA, 'mnist_svm_LinearSVC_%s_%s' %
    #                    (train_ratio, test_ratio)))
    info('%s, train_ratio=%s, test_ratio=%s' % (name, train_ratio, test_ratio))
    verify()


def mnist_count1():
    global model, x, y, tx, ty
    train_ratio = 0.01
    test_ratio = 0.01
    x, y, tx, ty = read_mnist_data(test_ratio, test_ratio, normalize=False,
                                   reshape=True)
    m = 28
    n = 28
    cc = [0]
    x = [count1(_, m, n, cc) for _ in x]
    cc[0] = 0
    tx = [count1(_, m, n, cc) for _ in tx]
    train()
    name = sys._getframe().f_code.co_name
    # save_model(os.join(PATH_DATA, 'mnist_svm_LinearSVC_%s_%s_%s' %
    #                    (name, train_ratio, test_ratio)))
    info('%s, train_ratio=%s, test_ratio=%s' % (name, train_ratio, test_ratio))
    verify()


def mnist_count2():
    global model, x, y, tx, ty
    train_ratio = 0.01
    test_ratio = 0.01
    x, y, tx, ty = read_mnist_data(test_ratio, test_ratio, normalize=False,
                                   reshape=True)
    m = 28
    n = 28
    cc = [0]
    x = [count2(_, m, n, cc) for _ in x]
    cc[0] = 0
    tx = [count2(_, m, n, cc) for _ in tx]
    train()
    name = sys._getframe().f_code.co_name
    # save_model(os.join(PATH_DATA, 'mnist_svm_LinearSVC_%s_%s_%s' %
    #                    (name, train_ratio, test_ratio)))
    info('%s, train_ratio=%s, test_ratio=%s' % (name, train_ratio, test_ratio))
    verify()


def mnist_count8():
    global model, x, y, tx, ty
    train_ratio = 1
    test_ratio = 1
    x, y, tx, ty = read_mnist_data(
        test_ratio, test_ratio, reshape=True,  # normalize=True,
        dir=os.path.join(PATH_DATA, 'MNIST_count8_data'))
    # print(x.shape, y.shape, tx.shape, ty.shape)
    # exit(0)
    train()
    name = sys._getframe().f_code.co_name
    # save_model(os.join(PATH_DATA, 'mnist_svm_LinearSVC_%s_%s_%s' %
    #                    (name, train_ratio, test_ratio)))
    info('%s, train_ratio=%s, test_ratio=%s' % (name, train_ratio, test_ratio))
    verify()


def mnist_count80():
    global model, x, y, tx, ty
    train_ratio = 1
    test_ratio = 1
    x, y, tx, ty = read_mnist_data(
        test_ratio, test_ratio, reshape=True,  # normalize=True,
        dir=os.path.join(PATH_DATA, 'MNIST_count80_data'))
    # print(x.shape, y.shape, tx.shape, ty.shape)
    # exit(0)
    train()
    name = sys._getframe().f_code.co_name
    # save_model(os.join(PATH_DATA, 'mnist_svm_LinearSVC_%s_%s_%s' %
    #                    (name, train_ratio, test_ratio)))
    info('%s, train_ratio=%s, test_ratio=%s' % (name, train_ratio, test_ratio))
    verify()


def main():
    # mnist_origin()
    # mnist_filter()
    # mnist_count1()
    # mnist_count2()
    mnist_count80()


if __name__ == '__main__':
    loginit(CURNAME, filename=os.path.join(PATH_LOG, 'mnist.log'))
    info(os.getcwd())
    main()
    info('exit')
