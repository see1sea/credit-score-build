# -*- coding: utf-8 -*-
"""
@Time    : 2026/4/30
@Author  : chen
@Software: PyCharm
"""
from src.base.lib import pd, logger
from src.base.source import DataSource

class FileDataSource(DataSource):
    """
        支持多种文件格式
    """

    def __init__(self, filepath: str, file_format: str = 'csv', **kwargs):
        """
            初始化文件数据源
        """
        self.filepath = filepath
        self.file_format = file_format.lower()
        self.kwargs = kwargs




    def load_data(self) -> pd.DataFrame:
        """
            从文件加载数据
        """

        logger.info(f"Loading data from {self.filepath} (format: {self.file_format})")

        if self.file_format == 'csv':
            df = pd.read_csv(self.filepath, **self.kwargs)
        elif self.file_format == 'excel':
            df = pd.read_excel(self.filepath, **self.kwargs)
        elif self.file_format == 'json':
            df = pd.read_json(self.filepath, **self.kwargs)
        elif self.file_format == 'parquet':
            df = pd.read_parquet(self.filepath, **self.kwargs)
        elif self.file_format == 'feather':
            df = pd.read_feather(self.filepath, **self.kwargs)
        else:
            raise ValueError(f"Unsupported file format: {self.file_format}")

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns.")
        return df