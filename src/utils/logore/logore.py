# -*- coding: utf-8 -*-
"""
@Time    : 2026/4/30
@Author  : chen
@Software: PyCharm
"""

import logging
import colorlog

logger = colorlog.getLogger('credit_risk_model')
logger.setLevel(logging.DEBUG)
logger.handlers.clear()

handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        #'%(log_color)s%(asctime)s - [%(levelname)s] - %(name)s - %(module)s:%(lineno)d - %(message)s',
        '%(log_color)s%(asctime)s - [%(levelname)s] - %(name)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        log_colors={
            'DEBUG': 'cyan',
            'INFO': 'white',
            'WARNING': 'yellow',
            'ERROR': 'red',
            'CRITICAL': 'red,bg_white'
        }
    )
)

logger.addHandler(handler)
logger.propagate = False