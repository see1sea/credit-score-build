# -*- coding: utf-8 -*-
"""
@Time    : 2026/4/30
@Author  : chen
@Software: PyCharm
"""

from src.base.lib import Protocol, pd

class DataSource(Protocol):
    """
        定义统一的数据加载接口
    """

    def load_data(self) -> pd.DataFrame:
        """
            加载数据
        """


