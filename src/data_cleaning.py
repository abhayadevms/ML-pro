import logging
from abc import ABC, abstractmethod
from typing import Union
import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):

    @abstractmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:

        pass

class DataPreProcessStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:

            data = data.dropna(subset=["order_purchase_timestamp",
                                 "order_approved_at", 
                                 "order_delivered_carrier_date", 
                                 "order_delivered_customer_date", 
                                 "order_estimated_delivery_date"])
            data['product_weight_g'].fillna(data['product_weight_g'].median(),inplace=True)
            data['product_length_cm'].fillna(data['product_length_cm'].median(),inplace=True)
            data['product_height_cm'].fillna(data['product_height_cm'].median(),inplace=True)
            data['product_width_cm'].fillna(data['product_width_cm'].median(),inplace=True)
            data['review_score'].fillna(data['review_score'].median(),inplace=True)
            data = data.select_dtypes(include=[np.number])

            cols_to_drop = ["customer_zip_code_prefix","order_item_id"]
            data = data.drop(cols_to_drop, axis=1)
            return data
        except Exception as e:
            logging.error(e)
            raise e
        
class DataDivideStrategy(DataStrategy):

    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        try:
            X = data.drop(["review_score"], axis=1)

            y = data["review_score"]


            X_train, X_test,y_train, y_test  = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test,y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e
class DataCleaning:

    def __init__(self, data: pd.DataFrame, data_strategy: DataStrategy):
        self.data = data
        self.data_strategy = data_strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        return self.data_strategy.handle_data(self.data)
    
# if __name__ == "__main__":
#     data = pd.read_csv("data.csv")
#     data_cleaning = DataCleaning(data, DataPreProcessStrategy())
#     data_cleaning.handle_data()