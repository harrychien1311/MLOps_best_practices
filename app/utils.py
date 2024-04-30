import torch
import io
from PIL import Image

def train(model, device, train_loader, optimizer, epoch, loss_fn):
    """ Train the model for one epoch on the training dataset.

        Args:
            model (torch.nn.Module): The neural network model to be trained.
            device (torch.device): The device to run the training on (e.g., 'cuda' or 'cpu').
            train_loader (torch.utils.data.DataLoader): The DataLoader containing the training dataset.
            optimizer (torch.optim.Optimizer): The optimizer used to update the model parameters.
            epoch (int): The current epoch number.
            loss_fn (torch.nn.Module): The loss function used to compute the training loss.

        Returns:
            torch.Tensor: The average training loss for the epoch
    """
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


def val(model, device, test_loader, loss_fn):
    """Evaluate the model for one epoch on the validation dataset.
    
    Args:
        model (torch.nn.Module): The trained neural network model to be evaluated.
        device (torch.device): The device to run the evaluation on (e.g., 'cuda' or 'cpu').
        test_loader (torch.utils.data.DataLoader): The DataLoader containing the validation dataset.
        loss_fn (torch.nn.Module): The loss function used to compute the validation loss.
    Returns:
        torch.Tensor: The average validation loss.
    """
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

def process_data(image, transform):
    """Process the input image using the specified transformation.

    Args:
        image (bytes): The input image data.
        transform (torchvision.transforms.Transform): The transformation to be applied to the image.

    Returns:
        PIL.Image.Image: The processed image.
    """
    image = Image.open(io.BytesIO(image))
    image = image.convert('L') # The required input is a gray scale image
    image = transform(image)
    return image


def predict(image, model):
    """Perform inference on the input image using the trained model

    Args:
        image (torch.Tensor): The input image tensor.
        model (torch.nn.Module): The trained neural network model used for inference.

    Returns:
        torch.Tensor: The predicted class label.
    """
    image = image.unsqueeze(0)
    model.eval()
    with torch.no_grad():
        output = model(image)
        output = torch.argmax(output, dim=1)
    return output 
