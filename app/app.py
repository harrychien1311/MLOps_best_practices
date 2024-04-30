import torch
import mlflow
from torchvision import transforms, datasets
import yaml
import torch.optim as optim
from mlflow.models.signature import infer_signature
from flask import (
    Flask,
    jsonify,
    request,
)
from model import Net
from utils import train, val, process_data, predict

app = Flask("model-store-interaction")
# Specify the experiment name where your model runs are logged
experiment_name = "MNIST training"
# Load all runs for the specified experiment
runs = mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name(experiment_name).experiment_id)
# Find the run with the smallest val loss metric
best_run = runs.loc[runs['metrics.val_loss'].idxmin()]
# Get the run ID of the best run
best_run_id = best_run.run_id
# Retrieve the best model from model registry
logged_model = f"runs:/{best_run_id}/model"
logged_model = mlflow.pytorch.load_model(logged_model)
print(logged_model)

transform = transforms.Compose([
                           transforms.Resize((28, 28)),
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,)) # Transformation for the input image in inference
                       ])

@app.route("/predict", methods=["POST"])
def run_inference():
    """Run inference on the provided image.

    Returns:
        dict: A dictionary containing the output of the inference.
    """
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    try:
        image = request.files['image'].read() # Read image from the request
        image = process_data(image, transform)
        print(logged_model)
        output = predict(image, logged_model)
    except Exception as e: # If there is any error, the handler function will thrown a staus code of 500 with the error message
        error_message = 'An error occurred: {}'.format(str(e))
        return jsonify({'error': error_message}), 500
    
    result = {"output": output.item()}
    return jsonify(result)

@app.route("/train", methods=["POST"])
def run_train():
    
    if 'config_file' not in request.files:
        return jsonify({'error': 'No config file provided'}), 400
    config_file = request.files["config_file"]
    try:
        config = yaml.safe_load(config_file)

        # Set random seed
        seed = config["seed"] if "seed" in config.keys() else 42
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        device = torch.device("cpu")

        # Initialize train and test loader
        train_dataset = datasets.MNIST('data', train=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ]))
        test_dataset = datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]))
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config["batch_size"], shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config["batch_size"], shuffle=True)
        
        mlflow.set_experiment("MNIST training")
        if "model_version" in config.keys():
            # if run id of a registered model is specified we train the model from a pre-trained model logged
            # in mlflow model registry
            pretrained_model = "models:/net/{}".format(config["model_version"])
            model = mlflow.pytorch.load_model(pretrained_model)
        else:
            model = Net()
        optimizer = optim.Adam(model.parameters(), lr = config["lr"])
        loss_fn = torch.nn.CrossEntropyLoss()
        
        # Infer the signature of the model to log the model in mlflow model registry
        model.to(device)
        sample_input, _ = train_dataset.__getitem__(0)
        model.eval()
        with torch.no_grad():
            sample_output = model(sample_input)
        signature = infer_signature(sample_input.numpy(), sample_output.numpy())
        
        
        # Start training with mlflow tracking
        with mlflow.start_run() as run:
            for epoch in range(1, config["max_epochs"] + 1):
                train_loss = train(model, device, train_loader, optimizer, epoch, loss_fn)
                val_loss = val(model, device, test_loader, loss_fn)
                if epoch == 1:
                    best_loss = val_loss
                if val_loss <= best_loss: # Only save the best loss checkpoint
                    best_loss = val_loss
                    mlflow.pytorch.log_model(model, registered_model_name="net", artifact_path="model", signature=signature)
                mlflow.log_metric("train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_param("training_config", config)
    except Exception as e: # If there is any error, the handler function will thrown a staus code of 500 with the error message
        error_message = 'An error occurred: {}'.format(str(e))
        return jsonify({'error': error_message}), 500
    result = {"status": "success",
              "experiment_run_id": run.info.run_id}
    return jsonify(result)

if __name__ == "__main__":
   app.run(debug=True, host="0.0.0.0", port=9696)