# -*- coding: utf-8 -*-
"""
@Time    : 2026/4/30
@Author  : chen
@Software: PyCharm
"""

from src.base.lib import Integer, Real, Categorical
from src.base.lib import os, json, pickle, pd, np, xgb, Dict, Any, logger
from src.base.lib import BayesSearchCV, Pipeline, StratifiedKFold, compute_classification_metrics, create_param_space_from_config


class CreditModel:
    """
        训练模型
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.cv_models = []
        self.overall_model = None
        self.best_params_ = None
        self.feature_names_ = None
        self.model_type = config['params'].get('model_type', 'xgboost')
        self.cv_folds = config['params'].get('cv_folds', 5)
        self.pipeline = None
        self.oot_metrics = None
        self.overall_train_metrics = None
        self.overall_oot_metrics = None
        self.overall_cv_metrics = None
        self.threshold = config['params'].get('classification_threshold', 0.5)
        self.model_addr = config['paths'].get('model_output', None)

    def build_pipeline(self):
        """
            构建模型管道（不包含特征选择）
        """

        steps = []
        # 根据配置选择模型
        if self.model_type == 'XGBoost':
            # 获取XGBoost的基础参数
            xgb_kwargs = self.config['params'].get('xgb_kwargs', {})
            # 添加默认的XGBoost参数
            default_xgb_kwargs = {
                'objective': 'binary:logistic',
                'importance_type': 'gain',
                'tree_method': 'hist',
                'verbosity': 0
            }
            # 更新为配置中的参数
            default_xgb_kwargs.update(xgb_kwargs)
            steps.append(('classifier', xgb.XGBClassifier(**default_xgb_kwargs)))

        elif self.model_type == 'LightGBM':
            # 获取 LGB 的基础参数（从配置文件读取）
            lgb_kwargs = self.config['params'].get('lgb_kwargs', {})
            # LGB 默认参数（适配二分类任务）
            default_lgb_kwargs = {
                'objective': 'binary',
                'metric': 'auc',
                'boosting_type': 'gbdt',
                'verbosity': -1,  # 关闭冗余日志
                'n_jobs': 1  # 避免并行冲突（和 BayesSearchCV 兼容）
            }
            # 配置文件参数覆盖默认参数
            default_lgb_kwargs.update(lgb_kwargs)
            steps.append(('classifier', lgb.LGBMClassifier(**default_lgb_kwargs)))

        # 未知模型类型容错
        else:
            raise ValueError(f"不支持的模型类型：{self.model_type}，仅支持 'XGBoost'/'LightGBM'")

        pipeline = Pipeline(steps)
        return pipeline

    def load_saved_params(self):
        """
            从文件加载保存的最佳参数
        """
        saved_params_file = self.config['params'].get('saved_params_file', 'artifacts/best_params.json')

        if os.path.exists(saved_params_file):
            with open(saved_params_file) as f:
                saved_params = json.load(f)
            logger.info(f"Loaded saved best parameters from {saved_params_file}")
            return saved_params
        else:
            logger.info(f"Saved parameters file {saved_params_file} not found. Will perform Bayesian optimization.")
            return None

    def save_best_params(self):
        """
            保存最佳参数到文件
        """
        saved_params_file = self.config['params'].get('saved_params_file', 'artifacts/best_params.json')

        if self.best_params_:
            # 确保目录存在
            os.makedirs(os.path.dirname(saved_params_file), exist_ok=True)

            with open(saved_params_file, 'w') as f:
                json.dump(self.best_params_, f, indent=2, default=str)
            logger.info(f"Best parameters saved to {saved_params_file}")

    def bayesian_tune(self, X_train: pd.DataFrame, y_train: pd.Series, X_oot: pd.DataFrame, y_oot: pd.Series):
        """
            贝叶斯调参，如果配置允许则使用保存的参数
        """
        use_saved_params = self.config['params'].get('use_saved_params', False)

        if use_saved_params:
            # 尝试加载保存的最佳参数
            saved_params = self.load_saved_params()
            if saved_params:
                logger.info("Using saved best parameters for quick training.")
                self.best_params_ = saved_params

                self.pipeline = self.build_pipeline()
                self.pipeline.set_params(**saved_params)
                self.pipeline.fit(X_train, y_train)

                self.overall_model = self.pipeline

                if len(X_train) > 0:
                    y_pred_proba = self.pipeline.predict_proba(X_train)[:, 1]

                    self.overall_train_metrics = compute_classification_metrics(y_train, y_pred_proba, self.threshold)

                else:
                    logger.info("No Train set available for evaluation with saved params.")
                    
                if len(X_oot) > 0:
                    y_pred_proba = self.pipeline.predict_proba(X_oot)[:, 1]

                    self.overall_oot_metrics = compute_classification_metrics(y_oot, y_pred_proba, self.threshold)

                else:
                    logger.info("No OOT set available for evaluation with saved params.")

                return self.overall_train_metrics,self.overall_oot_metrics

        self.pipeline = self.build_pipeline()
        param_space = create_param_space_from_config(self.config, self.model_type)

        search = BayesSearchCV(
            self.pipeline,
            param_space,
            n_iter=self.config['params']['bayes_n_iter'],
            scoring='roc_auc',
            cv=self.config['params']['bayes_cv_folds'],
            random_state=self.config['params']['bayes_random_state'],
            n_jobs=self.config['params']['bayes_n_jobs'],
            verbose=self.config['params']['bayes_verbose']
        )

        logger.info(f"Starting Bayesian optimization for {self.model_type}...")
        search.fit(X_train, y_train)

        self.best_params_ = search.best_params_

        self.pipeline.set_params(**search.best_params_)

        self.pipeline.fit(X_train, y_train)
        self.overall_model = self.pipeline

        self.save_best_params()

        y_train_pred_proba = self.pipeline.predict_proba(X_train)[:, 1]
        self.overall_train_metrics = compute_classification_metrics(y_train, y_train_pred_proba, self.threshold)
        y_oot_pred_proba = self.pipeline.predict_proba(X_oot)[:, 1]
        self.overall_oot_metrics = compute_classification_metrics(y_oot, y_oot_pred_proba, self.threshold)

        return self.overall_train_metrics,self.overall_oot_metrics

    def cross_validate(self, X_train: pd.DataFrame, y_train: pd.Series) -> Dict[str, Any]:
        if self.overall_model is None:
            raise ValueError("self.overall_model is None. Please train or load a model first.")

        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True,
                              random_state=self.config['params']['cv_random_state'])


        fold_details = []
        for fold_idx, (train_idx, valid_idx) in enumerate(skf.split(X_train, y_train)):
            x_val_fold = X_train.iloc[valid_idx]
            y_val_fold = y_train.iloc[valid_idx]
            y_val_pred_proba = self.pipeline.predict_proba(x_val_fold)[:, 1]

            # 使用新函数计算所有指标
            fold_metrics = compute_classification_metrics(y_val_fold, y_val_pred_proba, self.threshold)
            fold_result = {
                'fold_index': fold_idx + 1,
                **fold_metrics  # 展开所有指标
            }
            fold_details.append(fold_result)

        # 提取各指标用于汇总
        metric_keys = ['auc', 'ks', 'recall', 'precision', 'f1', 'fp_rate', 'fn_rate', 'tn_rate']
        summary_stats = {}
        for key in metric_keys:
            scores = [f[key] for f in fold_details]
            summary_stats[f'train_{key}_mean'] = np.mean(scores)
            summary_stats[f'train_{key}_std'] = np.std(scores)

        return {
            'fold_details': fold_details,
            'summary_stats': summary_stats
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
            使用整体模型进行预测
        """
        if self.overall_model is not None:
            return self.overall_model.predict(X)
        else:
            raise ValueError("No model available. Run bayesian_tune first.")



    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
            使用整体模型进行概率预测
        """
        if self.overall_model is not None:
            return self.overall_model.predict_proba(X)[:, 1]
        else:
            raise ValueError("No model available. Run bayesian_tune first.")


    def save_model(self):
        """
            仅保存整体训练模型到单一文件
        """
        model_data = {
            'config': self.config,
            'best_params_': self.best_params_,
            'feature_names_': self.feature_names_,
            'model_type': self.model_type,
            'overall_model': self.overall_model,
        }

        with open(self.model_addr, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"Overall model saved to {self.model_addr}")


