import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import numpy as np
import scipy

# Reproducibility
reproduce = True
if reproduce:
    FIXED_SEED = 5
    random.seed(FIXED_SEED)
    torch.manual_seed(FIXED_SEED)
    torch.cuda.manual_seed(FIXED_SEED)
    np.random.seed(FIXED_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

IMAGE_SIZE = 28
batch_size = 64
test_batch_size = 100

# Transformations
mnist_transformations = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,)),
])

# Data Source
mnist_train = datasets.MNIST('../data', train=True, download=True,
                             transform=mnist_transformations)
mnist_test = datasets.MNIST('../data', train=False,
                            transform=mnist_transformations)

# Data loaders
mnist_train_loader = DataLoader(mnist_train,
                                batch_size=batch_size, shuffle=True)
mnist_test_loader = DataLoader(mnist_test,
                               batch_size=test_batch_size, shuffle=True)

# Transformations
svhn_transformations = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    # transforms.Normalize((0.5,), (0.5,)),
    transforms.Normalize((0.1307,), (0.3081,)),

])

# Data Source

svhn_train = datasets.SVHN('../data', split='train', download=True,
                           transform=svhn_transformations)
svhn_test = datasets.SVHN('../data', split='test', download=True,
                          transform=svhn_transformations)

# Data loaders
svhn_train_loader = DataLoader(svhn_train,
                               batch_size=batch_size, shuffle=True)
svhn_test_loader = DataLoader(svhn_test,
                              batch_size=test_batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=7, padding=3, stride=1)
        self.bn1 = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=7, padding=3, stride=1)
        self.bn2 = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(7 * 7 * 20, 400)
        self.fc2 = nn.Linear(400, 10)

    def forward(self, x):
        # x = F.relu(self.bn1(F.max_pool2d(self.conv1(x), 2)))
        # x = F.relu(self.bn2(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2)))

        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 7 * 7 * 20)
        x = F.relu(self.fc1(x))

        # x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)


use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
model_cnn = Net().to(device)


def train(model, device, train_loader, optimizer, epoch, print_stat=True):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0 and print_stat:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader, name, print_stat=True):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    if print_stat:
        print('\n{} set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            name, test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


log_interval = 700


def run(epochs, lr):
    model = model_cnn
    optimizer = optim.Adam(model.parameters(), lr=lr)
    acc = 0
    for epoch in range(1, epochs + 1):
        train(model, device, svhn_train_loader, optimizer, epoch, print_stat=False)
        svhn_train = test(model, device, svhn_train_loader, name="SVHN-train", print_stat=False)
        svhn_test = test(model, device, svhn_test_loader, name="SVHN-test", print_stat=False)
        mnist_test = test(model, device, mnist_test_loader, name="MNIST-test", print_stat=False)

        acc = mnist_test

    torch.save(model.state_dict(), "RuslanSabirov.pt")
    return acc


def cv():
    fout = open("results.txt", "w")
    for epochs in [30]:
        for lr in [0.00001, 0.0001, 0.005, 0.001, 0.010, ]:
            print(f"RUN, lr={lr}, epochs={epochs}, ")
            print(f"RUN, lr={lr}, epochs={epochs}, ", file=fout)

            best_acc, best_epoch = run(epochs, lr)
            print(f"best accuracy={best_acc}, best_epoch={best_epoch}\n\n")
            print(f"best accuracy={best_acc}, best_epoch={best_epoch}\n\n", file=fout)

    fout.close()


# cv()
run(epochs=10, lr=0.0001)