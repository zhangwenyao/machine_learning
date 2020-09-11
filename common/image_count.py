# -*- coding: utf-8 -*-

import os
import logging
import numpy as np
from common.iofile import loginit

CURNAME = os.path.abspath(__file__)
PATH_CUR = os.path.dirname(os.path.abspath(__file__))
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

def count1(data, m, n, _cc=None):
    if _cc is not None:
        if _cc[0] % 100 == 0:
            print(_cc[0])
        _cc[0] += 1
    size = int(max(m, n) / 2)
    valmax = max(data)
    valmin = min(data)
    valmid = (valmax + valmin) / 2
    cnt = np.zeros([8 * size + 1], dtype=np.float32)
    cntn = np.zeros([8 * size + 1], dtype=np.uint32)
    dirs = [[1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, -1, 0],
            [-1, 0, 0, 1],
            [-1, 0, 0, -1],
            [0, -1, -1, 0],
            [0, -1, 1, 0],
            [1, 0, 0, -1]]
    datasumrow = np.zeros((m + 1, n + 1), dtype=np.uint32)
    datasumcol = np.zeros((m + 1, n + 1), dtype=np.uint32)
    datasumrow2 = np.zeros((m + 1, n + 1), dtype=np.uint32)
    datasumcol2 = np.zeros((m + 1, n + 1), dtype=np.uint32)
    data2 = data * data
    for i in range(m):
        for j in range(n):
            datasumcol[i + 1][j] = datasumcol[i][j] + data[i * n + j]
            datasumrow[i][j + 1] = datasumrow[i][j] + data[i * n + j]
            datasumcol2[i + 1][j] = datasumcol2[i][j] + data2[i * n + j]
            datasumrow2[i][j + 1] = datasumrow2[i][j] + data2[i * n + j]
    for i in range(m):
        for j in range(n):
            dij = data[i * n + j]
            # if dij < valmid:
            #     continue
            cnt[0] += dij ** 2
            cntn[0] += 1
            smax = min(size, max(m - 1 - i, i, n - 1 - j, j))
            for d in range(8):
                for s in range(1, smax + 1):
                    x = i + dirs[d][0] * s + dirs[d][2] * (
                        0 if d % 2 == 0 else 1)
                    y = j + dirs[d][1] * s + dirs[d][3] * (
                        0 if d % 2 == 0 else 1)
                    if not (0 <= x < m and 0 <= y < n):
                        break
                    # print(i, j, d, dirs[d], x, y, s)
                    if dirs[d][2] != 0:
                        if dirs[d][2] == 1:
                            dd = min(m - x, s)
                            ds = datasumcol[x + dd - 1][y] - datasumcol[x][y]
                            ds2 = datasumcol2[x + dd - 1][y] - \
                                  datasumcol2[x][y]
                            cnt[s * 8 - 7 + d] += dij ** 2 * dd - \
                                                  2 * dij * ds + ds2
                            cntn[s * 8 - 7 + d] += dd
                        else:
                            dd = min(x + 1, s)
                            ds = datasumcol[x + 1][y] - \
                                 datasumcol[x + 1 - dd][y]
                            ds2 = datasumcol2[x + 1][y] - \
                                  datasumcol2[x + 1 - dd][y]
                            cnt[s * 8 - 7 + d] += dij ** 2 * dd - \
                                                  2 * dij * ds + ds2
                            cntn[s * 8 - 7 + d] += dd
                    else:
                        if dirs[d][3] == 1:
                            dd = min(m - y, s)
                            ds = datasumrow[x][y + dd - 1] - datasumrow[x][y]
                            ds2 = datasumrow2[x][y + dd - 1] - \
                                  datasumrow2[x][y]
                            cnt[s * 8 - 7 + d] += dij ** 2 * dd - \
                                                  2 * dij * ds + ds2
                            cntn[s * 8 - 7 + d] += dd
                        else:
                            dd = min(y + 1, s)
                            ds = datasumrow[x][y + 1] - \
                                 datasumrow[x][y + 1 - dd]
                            ds2 = datasumrow2[x][y + 1] - \
                                  datasumrow2[x][y + 1 - dd]
                            cnt[s * 8 - 7 + d] += dij ** 2 * dd - \
                                                  2 * dij * ds + ds2
                            cntn[s * 8 - 7 + d] += dd
    return cnt / cntn


def count2(data, m: int, n: int, _cc=None, size=None):
    if _cc is not None:
        if _cc[0] % 100 == 0:
            print(_cc[0])
        _cc[0] += 1
    if size is None:
        size = int(max(m, n) / 2)
        # size = min(m, n) - 1
    # valmax = max(data)
    # valmin = min(data)
    # valmax = 0
    # valmin = 0
    # for i in range(m * n):
    #     if data[i] == 0:
    #         continue
    #     if valmax == 0 or data[i] > valmax:
    #         valmax = data[i]
    #     if valmin == 0 or data[i] < valmin:
    #         valmin = data[i]
    # valmid = (valmax + valmin) / 2
    cnt = np.zeros([8 * size + 1], dtype=float)
    cntn = np.ones([8 * size + 1], dtype=int)
    dirs = [[1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, -1, 0],
            [-1, 0, 0, 1],
            [-1, 0, 0, -1],
            [0, -1, -1, 0],
            [0, -1, 1, 0],
            [1, 0, 0, -1]]
    datasumrow = np.zeros((m + 1, n + 1), dtype=float)
    datasumcol = np.zeros((m + 1, n + 1), dtype=float)
    for i in range(m):
        for j in range(n):
            datasumcol[i + 1][j] = datasumcol[i][j] + data[i * n + j]
            datasumrow[i][j + 1] = datasumrow[i][j] + data[i * n + j]
    for i in range(m):
        for j in range(n):
            dij = data[i * n + j]
            # if dij < valmid:
            #     continue
            # if dij == 0:
            #     continue
            cnt[0] += dij
            cntn[0] += 1
            smax = min(size, max(m - 1 - i, i, n - 1 - j, j))
            for d in range(8):
                for s in range(1, smax + 1):
                    x = i + dirs[d][0] * s + dirs[d][2] * (
                        0 if d % 2 == 0 else 1)
                    y = j + dirs[d][1] * s + dirs[d][3] * (
                        0 if d % 2 == 0 else 1)
                    if not (0 <= x < m and 0 <= y < n):
                        break
                    if dirs[d][2] != 0:
                        if dirs[d][2] == 1:
                            dd = min(m - x, s)
                            ds = datasumcol[x + dd - 1][y] - datasumcol[x][y]
                            cnt[s * 8 - 7 + d] += ds
                            cntn[s * 8 - 7 + d] += dd
                        else:
                            dd = min(x + 1, s)
                            ds = datasumcol[x + 1][y] - \
                                 datasumcol[x + 1 - dd][y]
                            cnt[s * 8 - 7 + d] += ds
                            cntn[s * 8 - 7 + d] += dd
                    else:
                        if dirs[d][3] == 1:
                            dd = min(m - y, s)
                            ds = datasumrow[x][y + dd - 1] - datasumrow[x][y]
                            cnt[s * 8 - 7 + d] += ds
                            cntn[s * 8 - 7 + d] += dd
                        else:
                            dd = min(y + 1, s)
                            ds = datasumrow[x][y + 1] - \
                                 datasumrow[x][y + 1 - dd]
                            cnt[s * 8 - 7 + d] += ds
                            cntn[s * 8 - 7 + d] += dd
    return cnt / cntn


def filter(x, r=0.5):
    ma = max(x)
    mi = min(x)
    mid = (ma + mi) / 2
    for i in range(len(x)):
        if x[i] < mid:
            x[i] = 0
    return x


def main():
    pass


if __name__ == '__main__':
    loginit(CURNAME)
    info(os.getcwd())
    main()
    info('exit')
