import logging
import pandas as pd 
from zenml import step 
from src.evaluation import MSE, R2, RMSE
from sklearn.base import RegressorMixin
from typing_extensions import Annotated
from typing import Tuple 
from zenml.client import Client
import mlflow

experimental_tracker = Client().active_stack.experiment_tracker
logging.info(experimental_tracker)
@step(experiment_tracker=experimental_tracker.name)
def evaluate_model(model: RegressorMixin, 
    X_test: pd.DataFrame, y_test: pd.Series) -> Tuple[Annotated[float, "r2"],
                                                       Annotated[float, "rmse"]]:
    """
    Purpose: 
    """
    logging.info("Evaluating model")
    try:
        y_pred = model.predict(X_test)
        mse = MSE()
        mse = mse.calculate_score(y_test, y_pred)
        mlflow.log_metric("mse", mse)
        r2 = R2()
        r2 = r2.calculate_score(y_test, y_pred)
        mlflow.log_metric("r2", r2)
        rmse = RMSE()
        rmse = rmse.calculate_score(y_test, y_pred)
        mlflow.log_metric("rmse", rmse)

        logging.info(f"MSE: {mse}")
        logging.info(f"R2: {r2}")
        logging.info(f"RMSE: {rmse}")
        return r2, rmse
    
    except Exception as e:
        logging.error(e)
        logging.error("Evaluating model failed")
        raise e