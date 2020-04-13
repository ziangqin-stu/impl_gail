import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# import tensorboardX
# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

import torchvision
from torchvision import transforms


def get_num_correct(preds, labels):
    return preds.argmax(dim=1).eq(labels).sum().item()

class ConvTestNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=12, kernel_size=5)
        self.fc1 = nn.Linear(in_features=12*4*4, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=60)
        self.out = nn.Linear(in_features=60, out_features=10)

    def forward(self, state):
        x = F.relu(self.conv1(state))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.out(x)
        return x

# load image data from torchvision dataset
train_set = torchvision.datasets.FashionMNIST(
        root='./data',
        train=True,
        download=True,
        transform=transforms.Compose([transforms.ToTensor()])
    )

def testSummaryWriter():
    # variables prepare
    tb = SummaryWriter()
    network = ConvTestNetwork()
    # parse & load data
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)
    tb.add_image('images', grid)
    tb.add_graph(network, images)
    tb.close()

def testTensorBoardCNN(epoch_num):
    # create RL utils
    network = ConvTestNetwork()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=100, shuffle=True)
    optimizer = optim.Adam(network.parameters(), lr=0.01)
    # prepare data
    images, labels = next(iter(train_loader))
    grid = torchvision.utils.make_grid(images)
    # prepare tb utils
    tb = SummaryWriter()
    tb.add_image('images', grid)
    tb.add_graph(network, images)
    # training loop
    for epoch in range(epoch_num):
        total_loss = 0
        total_correct = 0
        for batch in train_loader:  # Get Batch
            images, labels = batch
            preds = network(images)
            loss = F.cross_entropy(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_correct += get_num_correct(preds, labels)
        # record data for tensorboard
        tb.add_scalar('Loss', total_loss, epoch)
        tb.add_scalar('Number Correct', total_correct, epoch)
        tb.add_scalar('Accuracy', total_correct / len(train_set), epoch)
        tb.add_histogram('conv1.bias', network.conv1.bias, epoch)
        tb.add_histogram('conv1.weight', network.conv1.weight, epoch)
        tb.add_histogram('conv1.weight.grad', network.conv1.weight.grad, epoch)
        # print training results
        print('epoch', epoch, 'total_correct', total_correct, 'loss', total_loss)

adas

# testSummaryWriter()
testTensorBoardCNN(50)
