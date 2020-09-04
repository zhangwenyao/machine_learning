# -*- coding: utf-8 -*-

import os
import logging
import numpy
from common.iofile import loginit

PATH_CUR = os.path.dirname(os.path.abspath(__file__))
CURNAME = os.path.splitext(os.path.split(PATH_CUR)[0])[0]
logger = logging.getLogger(CURNAME)
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical

mnist2 = None
model = None
x = None
y = None
test_x = None
test_y = None


# 每一方向对应：旋转角度 theta 小于 45°，伸缩因子 ratio，幅度 h

def xor1(data, rows, cols, _cc=None):
    if _cc is not None:
        if _cc[0] % 100 == 0:
            print(_cc[0])
        _cc[0] += 1

    res = data.reshape(rows, cols)
    for r in range(rows):
        for c in range(cols - 1):
            res[r][c + 1] ^= res[r][c]
    for r in range(rows - 1):
        res[r][cols - 1] ^= res[r + 1][- 1]
    for c in range(cols):
        for r in range(rows - 1):
            res[r + 1][c] ^= res[r][c]
    return res


def xor2(data, rows, cols, _cc=None):
    if _cc is not None:
        if _cc[0] % 100 == 0:
            print(_cc[0])
        _cc[0] += 1
    # data = data.reshape(rows, cols)
    return data


def main():
    pass


if __name__ == '__main__':
    loginit(CURNAME)
    info(os.getcwd())
    main()
    info('exit')
