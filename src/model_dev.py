import logging
from abc import ABC, abstractmethod
from sklearn.linear_model import LinearRegression



class Model(ABC):

    @abstractmethod
    def train(self, x_train, y_train):
        """
        Train the model
        Args:
            x_train (pd.DataFrame): Training data
            y_train (pd.Series): Training labels
            """
        pass

class LinearRegressionModel(Model):
    def train(self, x_train, y_train, **kwargs):
        try:
            reg=LinearRegression(**kwargs)
            reg.fit(x_train, y_train)
            logging.info("Training model completed")
            return reg
        except  Exception as e:
            logging.error(e)
            logging.error("Training model failed")
            raise e
