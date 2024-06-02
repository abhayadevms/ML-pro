import logging
import pandas as pd 
from zenml import step 
from src.model_dev import LinearRegressionModel
from sklearn.base import RegressorMixin
from .config import ModelNameConfig

@step
def train_model(X_train: pd.DataFrame,
                X_test: pd.DataFrame,
                  y_train: pd.Series,
                  y_test: pd.Series,
                  config: ModelNameConfig) -> RegressorMixin:
    """
    Purpose: Train the model
    Args:
        df (pd.DataFrame): The dataframe to train the model on
    """
    try:
        model=None
        if config.model_name=='LinearRegression':
            model = LinearRegressionModel()
            train_model = model.train(X_train, y_train)
            return train_model
        else:
            raise Exception('Invalid model name')
    except Exception as e:
        logging.error(e)
        logging.error("Training model failed")
        raise e
    