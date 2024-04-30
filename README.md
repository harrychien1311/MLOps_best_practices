# Lunit assignment
This source code is ML pipline for MNIST project. It includes a training script to train
a simple CNN model and 2 docker files to build the MLflow-based model registry and a service app
to interact with the model registry

lunit_assignment/
├── App_Dockerfile # Docker file for API service
├── Ml_flow_Dockerfile # Docker file for MLflow model registry
├── README.md
├── app # Source code for API service
│   ├── app.py
│   ├── app_requirements.txt
│   ├── model.py
│   └── utils.py
├── config # training config 
│   ├── initial_experiment.yaml
│   └── new_experiment.yaml
├── data # MNIST data
├── docker-compose.yaml
├── mlartifacts # volume folder for mlflow container
├── requirements.txt
├── test_suites # test suites of model registry
│   └── test_suite.py
├── train.py # training script

## Prequisites:
- Follow [here](https://docs.docker.com/engine/install/ubuntu/) to install Docker engine
- Follow [here](https://docs.anaconda.com/free/miniconda/miniconda-install/) to install Miniconda
- After installing conda successfully, create a virtual environment. Run:
```bash
conda create -n assignment python=3.11
``` 
- Run 
```bash
pip install -r requirements.txt
``` 
to install required requiremnts

## How to run
We have to start the docker compose service of mlflow first because the mlflow tracking is intergrated in training script
1. Start the docker compose service of mlflow-based model registry
    - First, build the contaniner of MLFlow server. Run:
    ```bash
    sudo docker build -t mlflow-server -f Ml_flow_Dockerfile .
    ``` 
    - Second, build the container of the API service that used to interact with the MLflow-based model registry. Run:
    ```bash
    sudo docker build -t app -f App_Dockerfile .
    ```
    - Then, start the docker compose service including mlflow server and the API service. Run:
    ```bash
    sudo docker compose -f docker-compose.yaml up -d
    ```
After start the docker compose service, you can view the UI of the model registry at http://127.0.0.1:5000

2. To run the training script with initial experiment:
```bash
python3 training.py --config config/initial_experiment.yaml
```
3. Test suites for the model registry:
We can do many things with the MLFlow model registry via python SDK or UI
Here, I wrote a script that python SDK of MLFLow to test some basic features of the model registry
    -  Retrieve a logged model from a run id:
```bash
python3 test_suites/test_suite.py --test_func download-runi --run_id <a run id>
``` 
Replace the run id with the run id you want to get the model. Find all of the run id at the model registry UI at http://127.0.0.1:5000
    - Retrieve a logged model from a specified version of this model
```bash
python3 test_suites/test_suite.py --test_func download-version --model_name net --model_version <number of version>
``` 
    - Delete a version of a logged from the model registry
```bash
python3 test_suites/test_suite.py --test_func delete --model_name net --model_version <number of version>
```
    - List information of a experiment specified by name
```bash
python3 test_suites/test_suite.py --test_func get_experiment --model_name net
```
Here, I only have one model name net with different versions in the model registry

4. To work with the API:
The API includes 2 handler functions to handle a request of training with a given configuration file and a request of inference wih a given of image
    - To send a post request of inference with a given image. Run:
    ```bash
    curl -X POST -F "image=@7.png" http://localhost:9696/predict
    ```
    You can replace "7.png" with another image file. The handler function will retrieve the best performance model registred in the model registry to run the inference
    - To send a post request of training with a given configuration file:
    ```bash
    curl -X POST -F "config_file=@config/new_experiment.yaml" http://localhost:9696/train
    ```
        In the new_experiment.yaml file there are 4 parameters: 
        - "max_epochs": number of max epochs
        - "lr": learning rate
        - "batch_size": batch size of each training step
        - "model_version": (Optional) the version of the logged model in the model registry if you want to train from a pretrained model
        
        You can adjsut different values to train a new model. The handler function will run the training code and save the trained model to the model registry. If the training is successfull, the handler will return a respone:
        {"status": "success", "model_url": the model uri of the logged model} 