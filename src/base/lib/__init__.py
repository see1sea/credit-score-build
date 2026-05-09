# -*- coding: utf-8 -*-
"""
@Time    : 2026/4/30
@Author  : chen
@Software: PyCharm
"""

import os
import yaml
import json
import pickle
import numpy as np
import pandas as pd
import xgboost as xgb

from typing import Protocol
from sklearn.metrics import roc_curve
from typing import Tuple, Dict, Any, Optional, List

from skopt import BayesSearchCV
from skopt.space import Integer, Real, Categorical

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split,StratifiedKFold
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, accuracy_score,precision_score,f1_score,confusion_matrix

from src.utils.logore import logger
from src.utils.metric.ks import ks_statistic
from src.utils.metric.model import create_param_space_from_config,compute_classification_metrics
from src.tools.sources import FileDataSource
