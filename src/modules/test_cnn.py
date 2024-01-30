import torch
import matplotlib.pyplot as plt
import torch.optim as opt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Root path to datasets
PATH = "C:/Users/Ato/Documents/Programming/Python/catdog/src/datasets"

# Hyperparameters
batch_size = 32
lr = 0.001
momentum = 0.9
epochs = 2

# Transforms for training set
train_transform = transforms.Compose([
  transforms.Resize((384,256)),
  transforms.ToTensor(),
])

# Data Loader with transformations, batch size and shuffle
train_path = PATH + "/train"
train_set = datasets.ImageFolder(root=train_path, transform=train_transform)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

images, labels = next(iter(train_loader))

conv1 = nn.Conv2d(3,6,5)
pool = nn.MaxPool2d(2,2)
conv2 = nn.Conv2d(6,16,5)

print(images.shape)

x = conv1(images)
print(x.shape)
x = pool(x)
print(x.shape)
x = conv2(x)
print(x.shape)
x = pool(x)
print(x.shape)
x = torch.flatten(x, 1)
print(x.shape)



