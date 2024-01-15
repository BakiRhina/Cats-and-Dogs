import torch
from torchvision import datasets

PATH = "C:/Users/Ato/Documents/Programming/Python/catdog/src/datasets"

train_path = PATH + "/train"
train_set = datasets.ImageFolder(root=train_path)
train_loader = torch.utils.data.DataLoader()
