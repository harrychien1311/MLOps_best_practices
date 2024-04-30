import mlflow
from mlflow import MlflowClient
import argparse

mlflow.set_tracking_uri("http://localhost:5000")
client = MlflowClient()

def test_model_registry(args):
    if args.test_func == "download-runid": # Dowload a model specified by run id from the model registry
        logged_model = f"runs:/{args.run_id}/model"
        logged_model = mlflow.pytorch.load_model(logged_model)
        print(logged_model)
    if args.test_func == "download-version": # Dowload a model specified by run id from the model registry
        logged_model = f"models:/{args.model_name}/{args.model_version}"
        logged_model = mlflow.pytorch.load_model(logged_model)
        print(logged_model)    
    if args.test_func == "delete": # Delete a version of a logged from the model registry
        client.delete_model_version(name=args.model_name, version=args.model_version)
        # Load the deleted version of the model to check if it's still exist
        try:
            mlflow.pytorch.load_model(model_uri=f"models:/{args.model_name}/{args.model_version}")
        except Exception as e:
            print(e)
    if args.test_func == "get_experiment": # List information of a experiment specified by name
        experiment = client.get_experiment_by_name(args.experiment_name)
        # Show experiment info
        print(f"Name: {experiment.name}")
        print(f"Experiment ID: {experiment.experiment_id}")
        print(f"Artifact Location: {experiment.artifact_location}")
        print(f"Lifecycle_stage: {experiment.lifecycle_stage}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Mlflow test suite')
    parser.add_argument('--test_func', type=str, help="Feature type of model registry")
    parser.add_argument('--run_id', type=str, help="run id of the model we want to download" )
    parser.add_argument('--model_name', type=str, help="model name of the registered model")
    parser.add_argument('--model_version', type=str, help="modle version of the registered model")
    parser.add_argument('--experiment_name', type=str, help="Experiment name of a experiment in the model registry")
    args = parser.parse_args()
    test_model_registry(args)