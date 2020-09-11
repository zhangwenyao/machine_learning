#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import sys
sys.path.append("..")
from common.iofile import loginit, PATH_R,PATH_DATA,PATH_LOG

CURNAME = os.path.abspath(__file__)
PATH_CUR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(CURNAME)
debug = logger.debug
info = logger.info
warning = logger.warning
error = logger.error
critical = logger.critical

def main():
    print(CURNAME)
    pass


if __name__ == "__main__":
    loginit(CURNAME)
    info(os.getcwd())
    main()
    info('exit')
