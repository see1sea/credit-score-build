


from src.base.lib import Integer, Real, Categorical, Dict, Any
from src.base.lib import  ks_statistic
from src.base.lib import roc_auc_score,confusion_matrix,f1_score


def compute_classification_metrics(y_true, y_proba, threshold=0.5):
    """
    计算分类指标
    """
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def safe_div(a, b):
        return a / b if b != 0 else 0.0

    metrics = {
        'auc': roc_auc_score(y_true, y_proba),
        'ks': ks_statistic(y_true, y_proba),
        'recall': safe_div(tp, tp + fn),
        'precision': safe_div(tp, tp + fp),
        'f1': f1_score(y_true, y_pred),
        'fp_rate': safe_div(fp, fp + tn),
        'fn_rate': safe_div(fn, fn + tp),
        'tn_rate': safe_div(tn, tn + fp),
        'tp_rate': safe_div(tp, tp + fn),
    }
    return metrics


def create_param_space_from_config(config: Dict[str, Any], model_type: str):
    """
    从配置文件创建参数空间（支持 LightGBM/XGBoost）
    """
    LGB_DISCRETE_INT_PARAMS = {
        'num_leaves', 'max_depth', 'n_estimators', 'min_child_samples',
        'min_child_weight', 'subsample_freq', 'reg_alpha', 'reg_lambda'
    }
    LGB_DISCRETE_REAL_PARAMS = {
        'learning_rate', 'subsample', 'colsample_bytree', 'colsample_bynode',
        'bagging_fraction', 'feature_fraction', 'lambda_l1', 'lambda_l2'
    }
    XGB_DISCRETE_INT_PARAMS = {
        'max_depth', 'n_estimators', 'min_child_weight', 'subsample_freq'
    }
    XGB_DISCRETE_REAL_PARAMS = {
        'learning_rate', 'subsample', 'colsample_bytree'
    }

    param_space = {}
    model_config = {
        'LightGBM': {
            'config_key': 'lgb_params',
            'discrete_int': LGB_DISCRETE_INT_PARAMS,
            'discrete_real': LGB_DISCRETE_REAL_PARAMS
        },
        'XGBoost': {
            'config_key': 'xgb_params',
            'discrete_int': XGB_DISCRETE_INT_PARAMS,
            'discrete_real': XGB_DISCRETE_REAL_PARAMS
        }
    }

    if model_type not in model_config:
        raise ValueError(
            f"Unsupported model type: {model_type}. Supported: {list(model_config.keys())}"
        )

    current_model = model_config[model_type]
    config_key = current_model['config_key']
    discrete_int_params = current_model['discrete_int']
    discrete_real_params = current_model['discrete_real']

    try:
        param_config = config['params'][config_key]
    except KeyError as e:
        raise KeyError(
            f"Missing '{config_key}' under 'params' in config"
        ) from e

    for param_name, param_def in param_config.items():
        if 'type' not in param_def:
            raise KeyError(f"Parameter '{param_name}' missing 'type' field")

        param_type = param_def['type']

        if param_type == 'Integer':
            for req_field in ['low', 'high']:
                if req_field not in param_def:
                    raise KeyError(f"Integer parameter '{param_name}' missing '{req_field}'")

            low = int(param_def['low'])
            high = int(param_def['high'])
            step = int(param_def.get('step', 1))

            if param_name in discrete_int_params:
                if step <= 0:
                    raise ValueError(f"Step must be > 0 for {param_name}, got {step}")
                choices = list(range(low, high + 1, step))
                if not choices:
                    raise ValueError(f"Invalid range for {param_name}: low={low} > high={high}")
                param_space[f'classifier__{param_name}'] = Categorical(choices)
            else:
                param_space[f'classifier__{param_name}'] = Integer(low, high)

        elif param_type == 'Real':
            for req_field in ['low', 'high']:
                if req_field not in param_def:
                    raise KeyError(f"Real parameter '{param_name}' missing '{req_field}'")

            low = float(param_def['low'])
            high = float(param_def['high'])
            prior = param_def.get('prior', 'uniform')
            step = float(param_def.get('step', 0.1))

            if param_name in discrete_real_params:
                if step <= 0:
                    raise ValueError(f"Step must be > 0 for {param_name}, got {step}")
                choices = []
                current = low
                while current <= high + 1e-8:
                    choices.append(round(current, 4))
                    current += step
                if not choices:
                    raise ValueError(f"Invalid range for {param_name}: low={low} > high={high}")
                param_space[f'classifier__{param_name}'] = Categorical(choices)
            else:
                param_space[f'classifier__{param_name}'] = Real(low, high, prior=prior)

        elif param_type == 'Categorical':
            if 'choices' not in param_def:
                raise KeyError(f"Categorical parameter '{param_name}' missing 'choices'")

            choices = param_def['choices']
            if not isinstance(choices, list) or len(choices) == 0:
                raise ValueError(f"'choices' must be non-empty list for {param_name}")

            if model_type.lower() == 'lightgbm':
                if param_name == 'boosting_type':
                    valid = {'gbdt', 'dart', 'goss', 'rf'}
                    choices = [c for c in choices if c in valid]
                    if not choices:
                        raise ValueError(f"boosting_type only supports {valid}, got {param_def['choices']}")
                elif param_name == 'metric':
                    valid = {'auc', 'binary_logloss', 'multi_logloss', 'mae'}
                    choices = [c for c in choices if c in valid]

            param_space[f'classifier__{param_name}'] = Categorical(choices)

        else:
            raise ValueError(f"Unsupported param type: {param_type} for {param_name}")

    return param_space