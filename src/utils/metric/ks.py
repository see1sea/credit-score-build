# -*- coding: utf-8 -*-
"""
@Time    : 2026/4/30
@Author  : chen
@Software: PyCharm
"""

from src.base.lib import np
from src.base.lib import roc_curve

def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
        计算KS统计量
    """
    fpr, tpr, _ = roc_curve(y_true, y_score)
    ks_stat = max(tpr - fpr)
    return ks_stat



