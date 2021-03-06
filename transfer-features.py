# %%
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import logging


# setup logging
LOG = logging.getLogger("CIFAR-FEATURES")
logging.basicConfig(filename="CIFAR-FEATURES.log", level=logging.DEBUG)


# Download the CIFAR dataset - these are PIL images
ROOT = './data'

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_data = datasets.CIFAR10(
    root=ROOT, train=True, download=True, transform=transform)
test_data = datasets.CIFAR10(
    root=ROOT, train=False, download=True, transform=transform)

LOG.info(f"Train data: {train_data}")
LOG.info(f"Test data: {test_data}")

# data loaders
BATCH_SIZE = 100
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# get the device
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
LOG.info(f"Using device: {DEVICE}")


def calculate_accuracy(y_pred, y):
    """Helper function to calculate the accuracy of our predictions"""
    prediction = y_pred.argmax(1, keepdim=True)
    correct = prediction.eq(y.view_as(prediction)).sum()
    acc = correct.float() / y.shape[0]
    return acc


# Load a pretrained model (ResNet18) and reset final fully connected layer.
net = models.resnet18(pretrained=True)

# Freeze all the network except the final layer.
for param in net.parameters():
    param.requires_grad = False

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = net.fc.in_features
net.fc = nn.Linear(num_ftrs, len(classes))

# %%
# instantiate the loss function and optimiser
criterion = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters())
LOG.info(f"Model: {net}")

# send to the device
net.to(DEVICE)
criterion.to(DEVICE)

# %%
for epoch in range(1, 50):  # loop over the dataset multiple times
    LOG.info(f"Epoch: {epoch}")

    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # log statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            msg = f'[{epoch}, {i + 1:5d}] loss: {running_loss / 100:.3f}'
            LOG.info(f"Loss: {msg}")
            running_loss = 0.0

# Log the final loss/accuracy
epoch_loss = 0
epoch_acc = 0
net.eval()

# no grad for evaluation
with torch.no_grad():
    for x, y in testloader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_pred = net(x)
        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)
        epoch_loss += loss.item()
        epoch_acc += acc.item()

final_loss = epoch_loss / len(testloader)
final_acc = epoch_acc / len(testloader)
LOG.info(f"Test loss: {final_loss:.3f}")
LOG.info(f"Test accuracy: {final_acc:.3f}")
