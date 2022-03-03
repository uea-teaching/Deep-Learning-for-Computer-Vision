import torch
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import logging


# setup logging
LOG = logging.getLogger("CIFAR-CNN")
logging.basicConfig(filename="CIFAR-CNN.log", level=logging.DEBUG)


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
LOG.info(f"Test data: {train_data}")

# data loaders
BATCH_SIZE = 16
trainloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
testloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# get the device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG.info(f"Using device: {DEVICE}")


# define the model
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        # The images in CIFAR-10 are of size 3x32x32
        # 3 input image channel, 6 output channels, 5x5 square convolution
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # feature image is now 5 x 5 
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def calculate_accuracy(y_pred, y):
    """Helper function to calculate the accuracy of our predictions"""
    prediction = y_pred.argmax(1, keepdim=True)
    correct = prediction.eq(y.view_as(prediction)).sum()
    acc = correct.float() / y.shape[0]
    return acc

# instantiate the model, loss function and optimiser
net = Model()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(net.parameters())
LOG.info(f"Model: {net}")

# send to the device
net.to(DEVICE)
criterion.to(DEVICE)


for epoch in range(1, 4):  # loop over the dataset multiple times
    LOG.info(f"Epoch: {epoch}")

    running_loss = 0.0
    for i, (inputs, labels) in enumerate(trainloader):

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
        y_pred = net(x)
        loss = criterion(y_pred, y)
        acc = calculate_accuracy(y_pred, y)
        epoch_loss += loss.item()
        epoch_acc += acc.item()

final_loss = epoch_loss / len(testloader)
final_acc = epoch_acc / len(testloader)
LOG.info(f"Test loss: {final_loss:.3f}")
LOG.info(f"Test accuracy: {final_acc:.3f}")
