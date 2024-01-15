import torch
import matplotlib.pyplot as plt
import torch.optim as opt
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from modules.cnn import CNN

# Root path to datasets
PATH = "C:/Users/Ato/Documents/Programming/Python/catdog/src/datasets"

# Select GPU if available (cuda toolkit <= 11.8 for pytorch)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Training on device: {device}")




# Transforms for training set
train_transform = transforms.Compose([
  transforms.Resize((256,256)),
  transforms.ToTensor(),
])

# Data Loader with transformations, batch size and shuffle
train_path = PATH + "/train"
train_set = datasets.ImageFolder(root=train_path, transform=train_transform)


# Hyperparameters
batch_size = 4
lr = 0.0001
momentum = 0.9
epochs = 2

# Images grouped in batch_sizes and sent to the network
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)

cnn = CNN(batch_size).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = opt.SGD(cnn.parameters(), lr, momentum)
loss_values = []

for epoch in range(epochs):
  print(f"Epoch: {epoch}")
  running_loss = 0.0
  for i, data in enumerate(train_loader, 0):
    imgs, labels = data
    imgs, labels = imgs.to(device), labels.to(device)

    # Forward + backward + optimize
    output = cnn(imgs)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    if i % 50 == 49:
      running_loss += loss.item()
      print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 50:.3f}')
      loss_values.append(running_loss)
      running_loss = 0.0

print('Finished Training')


model_filename = 'model_1b.pth'
model_path = f'models/{model_filename}'

torch.save(cnn.state_dict(), model_path)


