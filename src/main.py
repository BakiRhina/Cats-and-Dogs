import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets

PATH = "C:/Users/Ato/Documents/Programming/Python/catdog/src/datasets"

train_path = PATH + "/train"
train_set = datasets.ImageFolder(root=train_path)
train_loader = DataLoader(train_set, batch_size=4)



