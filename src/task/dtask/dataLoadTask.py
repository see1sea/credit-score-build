# -*- coding: utf-8 -*-
"""
@Time    : 2026/4/30
@Author  : chen
@Software: PyCharm
"""
from src.base.source import DataSource

from src.task.dtask.processDataTask import NumericalPreprocessor
from src.base.lib import pd, os, Tuple, Dict ,Any, Optional, yaml, logger, train_test_split

class DataLoadTask:
    """
        支持多种数据源的数据加载、预处理和拆分
    """

    def __init__(self, config_path: str = ''):
        """
            初始化数据任务
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.preprocessor = None
        self.data_source = None

    def _load_config(self) -> Dict[str, Any]:
        """
            加载配置文件
        """
        with open(self.config_path) as f:
            config = yaml.safe_load(f)
        return config

    def set_data_source(self, data_source: DataSource):
        """
            设置数据源
        """
        self.data_source = data_source

    def create_directories(self):
        """
            设置工作目录
        """
        paths_to_create = [
            os.path.dirname(self.config['paths']['model_output']),
            os.path.dirname(self.config['paths']['artifacts']),
            self.config['paths']['report_dir'],
        ]
        for path in paths_to_create:
            os.makedirs(path, exist_ok=True)

    def load_data(self) -> pd.DataFrame:
        """
            从当前数据源加载数据
        """
        if self.data_source is None:
            raise ValueError("No data source has been set. Please call set_data_source() first.")

        return self.data_source.load_data()

    def split_data_by_ratio(self, df: pd.DataFrame, target_col: str, date_col: str, test_size: float = 0.2,
                            random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
            按比例划分
        """
        X = df.drop(columns=[target_col, date_col])
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        logger.info(f"OOS Split - Train: {len(X_train)}, Test: {len(X_test)}")
        return X_train, X_test, y_train, y_test

    def split_data_by_date(
            self,
            df: pd.DataFrame,
            target_col: str,
            date_col: str,
            split_month: str,
            random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
            按日期拆分
        """
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in dataframe")

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])

        split_date = pd.to_datetime(split_month)

        train_mask = df[date_col] < split_date
        test_mask = df[date_col] >= split_date

        if train_mask.sum() == 0 or test_mask.sum() == 0:
            logger.info(f"Warning: No data found for {'train' if train_mask.sum() == 0 else 'test'} set.")
            logger.info(f"Data date range: {df[date_col].min()} to {df[date_col].max()}")
            logger.info(f"Split date: {split_date}")

        X_train = df[train_mask].drop(columns=[target_col, date_col])
        y_train = df[train_mask][target_col]

        X_test = df[test_mask].drop(columns=[target_col, date_col])
        y_test = df[test_mask][target_col]

        logger.info(f"OOT Split - Train: {len(X_train)}, Test: {len(X_test)}")

        if len(X_train) > 0:
            logger.info(f"Train date range: {df[train_mask][date_col].min()} to {df[train_mask][date_col].max()}")
        if len(X_test) > 0:
            logger.info(f"Test date range: {df[test_mask][date_col].min()} to {df[test_mask][date_col].max()}")

        return X_train, X_test, y_train, y_test

    def split_data_by_ratio_oot(
            self,
            df: pd.DataFrame,
            target_col: str,
            date_col: str,
            oot_ratio: float = 0.2,
            random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
            按比例拆分OOT
        """
        if date_col not in df.columns:
            raise ValueError(f"Date column '{date_col}' not found in dataframe")

        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df = df.dropna(subset=[date_col])

        df_sorted = df.sort_values(by=date_col).reset_index(drop=True)
        n_oot = max(1, int(len(df_sorted) * oot_ratio))

        X_train = df_sorted.iloc[:n_oot].drop(columns=[target_col, date_col])
        y_train = df_sorted.iloc[:n_oot][target_col]

        X_test = df_sorted.iloc[n_oot:].drop(columns=[target_col, date_col])
        y_test = df_sorted.iloc[n_oot:][target_col]

        logger.info(f"OOT Ratio Split - Train: {len(X_train)}, Test: {len(X_test)}")

        if len(X_train) > 0:
            logger.info(
                f"Train date range: {df_sorted.iloc[:n_oot][date_col].min()} to {df_sorted.iloc[:n_oot][date_col].max()}")
        if len(X_test) > 0:
            logger.info(
                f"Test date range: {df_sorted.iloc[n_oot:][date_col].min()} to {df_sorted.iloc[n_oot:][date_col].max()}")

        return X_train, X_test, y_train, y_test

    def build_data(self, df: pd.DataFrame,
                   target_col: str,
                   date_col: str) -> pd.DataFrame:
        """
            数据预处理
        """
        X = df.drop(columns=[target_col])
        y = df[target_col]

        self.preprocessor = NumericalPreprocessor()
        X_processed = self.preprocessor.fit_transform(X)

        # 重新组合数据框
        df_processed = pd.concat([X_processed, y], axis=1)
        if date_col in df.columns:
            df_processed = pd.concat([df_processed, df[date_col]], axis=1)

        logger.info(
            f"After preprocessing: {len(df_processed)} rows, {len(df_processed.columns)} columns, Columns: {list(df_processed.columns)}")
        return df_processed

    def setup_dataset(
            self,
            config_path: Optional[str] = None
    ) -> Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
            执行完整的数据集设置流程
        """
        if config_path:
            self.config = self._load_config()

        logger.info(f"Starting Credit Model Pipeline with {self.config['params']['model_type']}:")

        self.create_directories()
        df = self.load_data()

        target_col = self.config['params']['target_col']
        date_col = self.config['params']['date_col']

        # 数据预处理
        df_processed = self.build_data(df, target_col, date_col)

        use_oot = self.config['params'].get('use_oot', False)

        if use_oot:
            oot_method = self.config['params']['oot_method']
            if oot_method == 'month':

                try:
                    X_train, X_test, y_train, y_test = self.split_data_by_date(
                        df_processed, target_col, date_col, self.config['params']['oot_split_month']
                    )
                except ValueError as e:
                    logger.info(f"OOT month split failed: {e}")
                    logger.info("Falling back to ratio split...")
                    X_train, X_test, y_train, y_test = self.split_data_by_ratio_oot(
                        df_processed, target_col, date_col, self.config['params']['oot_ratio']
                    )

            elif oot_method == 'ratio':
                X_train, X_test, y_train, y_test = self.split_data_by_ratio_oot(
                    df_processed, target_col, date_col, self.config['params']['oot_ratio']
                )

            else:
                raise ValueError(f"Invalid oot_method: {oot_method}")

        else:
            X_train, X_test, y_train, y_test = self.split_data_by_ratio(
                df_processed, target_col, date_col,
                self.config['params']['test_size'],
                self.config['params']['random_state']
            )

        logger.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
        return self.config, X_train, X_test, y_train, y_test




