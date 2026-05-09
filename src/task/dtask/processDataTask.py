# -*- coding: utf-8 -*-
"""
@Time    : 2026/4/30
@Author  : chen
@Software: PyCharm
"""

from src.base.lib import BaseEstimator, TransformerMixin
from src.base.lib import pickle, np, pd, Dict, List, logger


class NumericalPreprocessor(BaseEstimator, TransformerMixin):
    """
        数值列预处理器：选择数值列 + 异常值替换
    """
    def __init__(self, suspicious_values: List = [-999999]):
        self.suspicious_values = suspicious_values
        self.numeric_cols_ = None

    def fit(self, X: pd.DataFrame, y=None):
        """
            识别数值列（排除目标列和日期列）
        """
        numeric_cols = []
        for col in X.columns:
            if pd.api.types.is_numeric_dtype(X[col]):
                try:
                    pd.to_numeric(X[col], errors='raise')
                    numeric_cols.append(col)
                except (ValueError, TypeError):
                    continue

        self.numeric_cols_ = numeric_cols
        logger.info(f"Selected {len(self.numeric_cols_)} true numeric columns: {self.numeric_cols_}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
            转换：替换-999999为NA，仅保留数值列
        """
        cols_to_process = [col for col in self.numeric_cols_ if col in X.columns]
        X_transformed = X[cols_to_process].copy()

        for val in self.suspicious_values:
            X_transformed = X_transformed.replace(val, pd.NA)

        return X_transformed


class WOEBinTransformer(BaseEstimator, TransformerMixin):
    """
        WOE分箱转换器
    """
    def __init__(self, n_bins: int = 10, target_col: str = "target"):
        self.n_bins = n_bins
        self.target_col = target_col
        self.bins_map_ = {}
        self.iv_values_ = {}

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
            训练分箱规则
        """
        for col in X.columns:

            if not pd.api.types.is_numeric_dtype(X[col]):
                logger.info(f"Skipping non-numeric column: {col}")
                continue

            x_clean = X[col].dropna()
            if len(x_clean) == 0 or len(x_clean.unique()) < 2:
                logger.info(f"Skipping column {col} due to insufficient data")
                continue

            try:

                x_clean = pd.to_numeric(x_clean, errors='coerce').dropna()
                if len(x_clean) == 0 or len(x_clean.unique()) < 2:
                    logger.info(f"Skipping column {col} after numeric conversion")
                    continue

                q = np.quantile(x_clean, np.linspace(0, 1, self.n_bins + 1), method='inverted_cdf')
                q = np.unique(q)

                if len(q) < 2:
                    logger.info(f"Skipping column {col} due to insufficient unique values after binning")
                    continue

                bins = pd.cut(x_clean, bins=q, include_lowest=True, duplicates='drop')

                df_bin = pd.DataFrame({col: x_clean, 'bin': bins, 'target': y[x_clean.index]})
                agg = df_bin.groupby('bin',observed=False).agg({col: 'count', 'target': 'sum'}).rename(columns={col: 'total',
                                                                                                  'target': 'bad'})
                agg['good'] = agg['total'] - agg['bad']

                agg['dist_good'] = (agg['good'] + 0.5) / (agg['good'].sum() + 0.5 * len(agg))
                agg['dist_bad'] = (agg['bad'] + 0.5) / (agg['bad'].sum() + 0.5 * len(agg))
                agg['woe'] = np.log(agg['dist_good'] / agg['dist_bad'])
                agg['iv'] = (agg['dist_good'] - agg['dist_bad']) * agg['woe']

                iv = agg['iv'].sum()
                self.iv_values_[col] = iv
                self.bins_map_[col] = {'breaks': q, 'woe_map': dict(zip(agg.index, agg['woe']))}
            except Exception as e:
                logger.info(f"Error processing column {col}: {e}")
                continue

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
            应用分箱规则转换
        """
        X_transformed = X.copy()

        for col, bin_info in self.bins_map_.items():
            if col not in X.columns:
                continue

            breaks = bin_info['breaks']
            woe_map = bin_info['woe_map']


            x_series = pd.to_numeric(X[col], errors='coerce')

            bins = pd.cut(x_series, bins=breaks, include_lowest=True, duplicates='drop')


            bins = bins.astype(object)


            woe_values = bins.map(woe_map)

            woe_values = woe_values.fillna(0)

            X_transformed[col] = woe_values

        return X_transformed

    def save_bins(self, filepath: str):
        """
            保存分箱规则
        """
        with open(filepath, 'wb') as f:
            pickle.dump({'bins_map': self.bins_map_, 'iv_values': self.iv_values_}, f)
        logger.info(f"Bins saved to {filepath}")

    def load_bins(self, filepath: str):
        """
            加载分箱规则
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.bins_map_ = data['bins_map']
        self.iv_values_ = data['iv_values']
        logger.info(f"Bins loaded from {filepath}")


def select_features_by_iv(iv_values: Dict[str, float], threshold: float = 0.02) -> List[str]:
    """
        根据IV值选择特征
    """
    selected_features = [k for k, v in iv_values.items() if pd.notna(v) and v >= threshold]
    logger.info(f"Selected {len(selected_features)} features based on IV >= {threshold}: {selected_features}")
    return selected_features
