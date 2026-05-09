# -*- coding: utf-8 -*-
"""
@Time    : 2026/4/30
@Author  : chen
@Software: PyCharm
"""

# train.py

import click
from src.base.lib import logger
from src.task.dtask import DataLoadTask
from src.task.mtask import CreditModel, ModelEvaluator
from src.tools.sources.fileSource import FileDataSource



def run(config_path: str, data_path: str):
    """
        RUN
    """
    # 1. 加载数据
    logger.info("Step 01. Start.")
    task = DataLoadTask(config_path=config_path)
    task.set_data_source(FileDataSource(data_path))

    # 2. 无特征工程/无变量筛选, 仅训练
    config, X_train, X_test, y_train, y_test = task.setup_dataset()

    # 3. 自动调参
    logger.info("Step 02. TRAIN.")
    model = CreditModel(config)
    train_metrics, oot_metrics = model.bayesian_tune(X_train, y_train, X_test, y_test)
    cv_metrics = model.cross_validate(X_train, y_train)

    # 4. 模型评估
    logger.info("Step 03. EVALUATE.")
    evaluator = ModelEvaluator()
    evaluator.evaluate(train_metrics, oot_metrics, cv_metrics)

    # 保存模型
    model.save_model()

    logger.info("Step 04. FINISHED.")


@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default="config/config.yaml",
    show_default=True,
    help="Path to the configuration YAML file."
)
@click.option(
    "--data",
    "-d",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    default="data/raw/train.csv",
    show_default=True,
    help="Path to the training CSV file."
)

def cmd(config: str, data: str):
    """
    Run the credit risk model training pipeline.
    """


    try:
        run(config_path=config, data_path=data)
        click.secho("Training completed successfully!", fg="green")
    except Exception as e:
        logger.exception("Training failed!")
        click.secho(f"Error: {e}", fg="red", err=True)
        raise click.Abort()


