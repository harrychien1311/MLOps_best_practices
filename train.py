# MLFLow Pytorch mnist training integration

import argparse
import os
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import mlflow.pytorch
from mlflow.models.signature import infer_signature

# Setup mlflow tracking server uri 
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
mlflow.set_experiment("MNIST training")

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train(model, device, train_loader, optimizer, epoch, loss_fn):
    model.train()
    epoch_train_loss = 0 # train loss for each epoch
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        
        epoch_train_loss += loss
        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
    # Calculate average train loss for each epoch
    epoch_train_loss /= len(train_loader.dataset)
    return epoch_train_loss


def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    accuracy = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += loss_fn(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            accuracy += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, accuracy, len(test_loader.dataset),
        100. * accuracy / len(test_loader.dataset)))
    return test_loss
    
def get_next_run_folder(params_folder):
    cur_run_folders = os.listdir(params_folder)
    last_run = 0
    for folder in cur_run_folders:
        if folder[:3] != 'run':
            continue
        if int(folder[3:]) > last_run:
            last_run = int(folder[3:])
    return os.path.join(params_folder, 'run{}'.format(last_run + 1))

def main(args):

    # Parse the training config file to set up the training job
    with open(args.config, "r") as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Set random seed
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    device = torch.device("cpu")

    # Initialize train and test loader
    train_dataset = datasets.MNIST('data', train=True, download=True,
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

    model = Net().to(device)
    optimizer = optim.Adam(model.parameters(), lr = config["lr"])
    loss_fn = torch.nn.CrossEntropyLoss()
    
    os.makedirs('result', exist_ok=True) # Make the save folder to save the model
    # For each training, we save the model file to a different folder name run
    run_folder = get_next_run_folder("result")
    
    # Infer the signature of the model
    sample_input, _ = train_dataset.__getitem__(0)
    model.eval()
    with torch.no_grad():
        sample_output = model(sample_input.to(device))
    signature = infer_signature(sample_input.cpu().numpy(), sample_output.cpu().numpy())
    
    # Start training with mlflow tracking
    with mlflow.start_run() as run:
        for epoch in range(1, config["max_epochs"] + 1):
            try:
                train_loss = train(model, device, train_loader, optimizer, epoch, loss_fn)
                val_loss = test(model, device, test_loader, loss_fn)
                if epoch == 1:
                    best_loss = val_loss
                if val_loss <= best_loss:
                    best_loss = val_loss
                    os.makedirs(run_folder, exist_ok=True)
                    torch.save(model.state_dict(), os.path.join(run_folder, "mnist_cnn.pt"))
                    mlflow.pytorch.log_model(model, registered_model_name="net",artifact_path="model",signature=signature)
                mlflow.log_metric(f"train_loss", train_loss, step=epoch)
                mlflow.log_metric("val_loss", val_loss, step=epoch)
            except:
                raise
        mlflow.log_param("training_config", config)
                

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch MNIST')
    parser.add_argument('--config', type=str, help="training config file")
    args = parser.parse_args()
    main(args)