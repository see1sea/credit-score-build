# -*- coding: utf-8 -*-
"""
@Time    : 2026/4/30
@Author  : chen
@Software: PyCharm
"""

from typing import Dict, Any
from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    """抽象基类，定义评估接口"""

    @abstractmethod
    def evaluate(self, *args, **kwargs) -> Dict[str, Any]:
        pass

    @abstractmethod
    def save(self, *args, **kwargs) :
        pass