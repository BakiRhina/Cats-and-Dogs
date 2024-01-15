import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from modules.cnn import CNN

# Root path to datasets
PATH = "C:/Users/Ato/Documents/Programming/Python/catdog/src/datasets"


# Transforms for training set
train_transform = transforms.Compose([
  transforms.Resize((256,256)),
  transforms.ToTensor(),
])

# Data Loader with transformations, batch size and shuffle
train_path = PATH + "/train"
train_set = datasets.ImageFolder(root=train_path, transform=train_transform)
train_loader = DataLoader(train_set, batch_size=16, shuffle=True)

'''
# Image test #
# - Manually iterates through the batch and shows the first image.
# - permute() changes the order of dimensions so the Channel (RGB) is in the last position.
images, labels = next(iter(train_loader))
plt.imshow(images[0].permute(1,2,0).numpy())
plt.show() # No need if using notebooks

# Debug tensors and numpy image representation
print("TENSOR: ", images[1])
print("NUMPY: ", images[1].numpy())
'''