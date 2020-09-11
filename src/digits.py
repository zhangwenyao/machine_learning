# -*- coding: utf-8 -*-

import os
import logging
import numpy
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import sys

sys.path.append('..')
from common.iofile import loginit, PATH_R, PATH_DATA, PATH_LOG, save_model
from common.mnist_base import read_digits_data

CURNAME = os.path.abspath(__file__)
PATH_CUR = os.path.dirname(os.path.abspath(__file__))
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
model_name = None


def train(max_iter=1000):
    global model, x, y, tx, ty, model_name
    model_name = 'LinearSVC'
    model = svm.LinearSVC(max_iter=max_iter)
    # model = svm.SVC(kernel='linear')
    # model_name = 'SVC_rbf'
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


def digits_export_data():
    digits = load_digits()
    data, target = digits.data, digits.target
    print(data.shape, target.shape, target[0].dtype)
    data = data.astype(int)
    target = target.astype(int)
    numpy.savetxt(os.path.join(PATH_DATA, 'digits_data', 'data.txt'),
                  data, delimiter="\t")
    numpy.savetxt(os.path.join(PATH_DATA, 'digits_data', 'target.txt'),
                  target, delimiter="\t")


def digits_origin(train_size=None, test_size=None, max_iter=None):
    global x, y, tx, ty
    # 1797, 8 * 8
    # train_size:0.75 test_size:0.25
    digits = load_digits()
    x = digits.data
    y = digits.target
    x = MinMaxScaler().fit_transform(x)
    if train_size is not None or test_size is not None:
        x, tx, y, ty = train_test_split(x, y, random_state=1,
                                        train_size=train_size,
                                        test_size=test_size)
    else:
        tx = x
        ty = y
    # print(x.shape, y.shape, tx.shape, ty.shape)
    # print(x[0], y[0])
    # print(x[-1], y[-1])
    # print(tx[0], ty[0])
    # print(tx[-1], ty[-1])
    train(max_iter)
    name = sys._getframe().f_code.co_name
    # save_model(os.join(PATH_DATA, 'mnist_svm_LinearSVC_%s_%s_%s' %
    #                    (name, train_ratio, test_ratio)))
    info('%s, %s, train_size=%s, test_size=%s' % (
        name, model_name, train_size, test_size))
    verify()


def digits_count8_do(name, train_size=None, test_size=None, max_iter=None,
                     dtype=None, normalize=True):
    global x, y, tx, ty
    # 1797, 8 * 8
    x, y = read_digits_data(reshape=True, normalize=normalize, dtype=dtype,
                            dir=os.path.join(PATH_DATA, name))
    if train_size is not None or test_size is not None:
        x, tx, y, ty = train_test_split(x, y, random_state=1,
                                        train_size=train_size,
                                        test_size=test_size)
    else:
        tx = x
        ty = y
    # print(x.shape, y.shape, tx.shape, ty.shape)
    # print(x[0], y[0])
    # print(x[-1], y[-1])
    # print(tx[0], ty[0])
    # print(tx[-1], ty[-1])
    train(max_iter)
    # name = sys._getframe().f_code.co_name
    # save_model(os.join(PATH_DATA, 'mnist_svm_LinearSVC_%s_%s_%s' %
    #                    (name, train_ratio, test_ratio)))
    info('%s, %s, train_size=%s, test_size=%s' % (
        name, model_name, train_size, test_size))
    verify()


def main():
    max_iter = 1000
    # digits_origin(0.75, 0.25, max_iter=max_iter)
    digits_origin(max_iter=max_iter)
    # digits_count2(0.75, 0.25, max_iter=max_iter)
    # digits_export_data()
    # digits_count8_do('digits_count8_data', max_iter=max_iter)
    # digits_count8_do('digits_count80n_data', 0.75, 0.25, max_iter=max_iter)
    digits_count8_do('digits_count80n_data', max_iter=max_iter)
    # digits_count8_do('digits_count80n_data', max_iter=max_iter)
    # digits_count8_do('digits_count87_data', max_iter=max_iter)
    # digits_count8_do('digits_count80n4_data', max_iter=max_iter)
    # digits_count8_do('digits_count870_data', max_iter=max_iter)
    # digits_count8_do('digits_count870n_data', max_iter=max_iter)


if __name__ == '__main__':
    loginit(CURNAME, filename=os.path.join(PATH_LOG, 'digits.log'))
    info(os.getcwd())
    main()
    info('exit')
