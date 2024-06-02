from pipelines.deployement_pipeline import continious_deployement_pipeline
import click
from rich import print
from zenml.integrations.mlflow import MLFLOW
from zenml.integrations.mlflow.mlflow_utils import get_tracking_uri
from zenml.integrations.mlflow.model_deployers import MLFlowModelDeployer                                                                            

from zenml.integrations.mlflow.services import MLFlowDeploymentService

DEPLOY="deploy"
PREDICT="predict"
DEPLOY_AND_PREDICT="deploy_and_predict"

@click.command()
@click.option(
    "--config",
    "-c",
    type=click.Choice([DEPLOY, PREDICT, DEPLOY_AND_PREDICT]),
    default=DEPLOY_AND_PREDICT,
    help="Choose between deploy, predict or deploy_and_predict",
)
@click.option(
    
    "--min-accuracy",
    default=0.92,
    help="Minimum accuracy for deployment",
)
def main(config: str, min_accuracy: float):
    mlflow_model_deployer_component = MLFlowModelDeployer.get_active_model_deployer()
    deploy = config == DEPLOY or config == DEPLOY_AND_PREDICT
    predict = config == PREDICT or config == DEPLOY_AND_PREDICT
    if deploy:
        continious_deployement_pipeline(data_path="data/olist_customers_dataset.csv",
                                            min_accuaracy=0.92,
                                            workers=3,
                                            timeout=60)
        
        

    # if predict:
    #     inference_pipeline()

    # print(
    #     "You can run:\n "
    #     f"[italic green]    mlflow ui --backend-store-uri '{get_tracking_uri()}"
    #     "[/italic green]\n ...to inspect your experiment runs within the MLflow"
    #     " UI.\nYou can find your runs tracked within the "
    #     "`mlflow_example_pipeline` experiment. There you'll also be able to "
    #     "compare two or more runs.\n\n"
    # )

    # # fetch existing services with same pipeline name, step name and model name
    # existing_services = mlflow_model_deployer_component.find_model_server(
    #     pipeline_name="continuous_deployment_pipeline",
    #     pipeline_step_name="mlflow_model_deployer_step",
    #     model_name="model",
    # )

    # if existing_services:
    #     service = cast(MLFlowDeploymentService, existing_services[0])
    #     if service.is_running:
    #         print(
    #             f"The MLflow prediction server is running locally as a daemon "
    #             f"process service and accepts inference requests at:\n"
    #             f"    {service.prediction_url}\n"
    #             f"To stop the service, run "
    #             f"[italic green]`zenml model-deployer models delete "
    #             f"{str(service.uuid)}`[/italic green]."
    #         )
    #     elif service.is_failed:
    #         print(
    #             f"The MLflow prediction server is in a failed state:\n"
    #             f" Last state: '{service.status.state.value}'\n"
    #             f" Last error: '{service.status.last_error}'"
    #         )
    # else:
    #     print(
    #         "No MLflow prediction server is currently running. The deployment "
    #         "pipeline must run first to train a model and deploy it. Execute "
    #         "the same command with the `--deploy` argument to deploy a model."
    #     )
if __name__ == "__main__":
    main()