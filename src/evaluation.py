import logging
from abc import ABC, abstractmethod
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
class Evaluation(ABC):

    @abstractmethod
    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):

        pass

class MSE(Evaluation):
    """
    Purpose: mean squared error"""

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            mse = mean_squared_error(y_true, y_pred)
            logging.info(f"MSE: {mse}")
            return mse
        except Exception as e: 
            logging.error("Error in calculating MSE: {}".format(e))#, e.args[0], e.args[1], e)
            raise e
        
class R2(Evaluation):

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            r2 = r2_score(y_true, y_pred)
            logging.info(f"R2: {r2}")
            return r2
        except Exception as e:
            logging.error("Error in calculating R2: {}".format(e)) 
            raise e
class RMSE(Evaluation):

    def calculate_score(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            rmse = mean_squared_error(y_true, y_pred, squared=False)
            logging.info(f"RMSE: {rmse}")
            return rmse
        except Exception as e:
            logging.error("Error in calculating RMSE: {}".format(e))
            raise e       