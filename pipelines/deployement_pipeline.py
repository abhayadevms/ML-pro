import numpy as np 
import pandas as pd 
import logging

from zenml import pipeline, step
from zenml.config import DockerSettings
from zenml.integrations.mlflow import MLFLOW
from zenml.config import DockerSettings
from zenml.constants import DEFAULT_SERVICE_START_STOP_TIMEOUT
# from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer
# from zenml.integrations.mlflow.services import MLFlowDeploymentService
from zenml.integrations.mlflow.steps import mlflow_model_deployer_step
from zenml.steps import BaseParameters, Output

from steps.clean_data import clean_data
from steps.ingest_data import ingest_df
from steps.model_train import train_model
from steps.evaluation import evaluate_model



docker_setting = DockerSettings(required_integrations=[MLFLOW])

class DeploymentTriggerConfig(BaseParameters):
    min_accuracy: float
@step
def deployment_trigger(
    accuracy: float,
):
    """Implements a simple model deployment trigger that looks at the
    input model accuracy and decides if it is good enough to deploy"""

    return accuracy < 0.52

@pipeline(enable_cache=False, settings={"docker": docker_setting})
def continious_deployement_pipeline(data_path: str,
                                    min_accuaracy: float=0.92,
                                    workers: int=1, 
                                    timeout:int = DEFAULT_SERVICE_START_STOP_TIMEOUT):
    

    df = ingest_df(data_path)
    X_train, X_test, y_train, y_test = clean_data(df)
    model =  train_model(X_train, X_test, y_train, y_test)
     
    r2, rmse = evaluate_model(model, X_test, y_test)
    logging.info(f"R2: {r2}, RMSE: {rmse}")
    
    deploy_desicion = deployment_trigger(accuracy=r2)
    mlflow_model_deployer_step(model=model,
                                deploy_decision=deploy_desicion,
                              
                               workers=workers,
                               timeout=timeout)

