from pipelines.training_pipeline import training_pipeline
from zenml.client import Client
if __name__ == "__main__":
    stack = Client().active_stack
    experiment_tracker = stack.experiment_tracker
    

    # Get the experiment tracker component
    if experiment_tracker.flavor == "mlflow":
        a= experiment_tracker.get_tracking_uri()
        print(a)

        # tracking_uri = experiment_tracker.config.uri
        # print(f"MLflow Tracking URL: {tracking_uri}")
    
        training_pipeline(data_path="data/olist_customers_dataset.csv")
    