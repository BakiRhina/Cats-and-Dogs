import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

PATH = "C:/Users/Ato/Documents/Programming/Python/catdog/src/datasets"

train_transform = transforms.Compose([
  transforms.Resize((256,256)),
  transforms.ToTensor(),
])

train_path = PATH + "/train"
train_set = datasets.ImageFolder(root=train_path, transform=train_transform)
train_loader = DataLoader(train_set, batch_size=4, shuffle=True)

images, labels = next(iter(train_loader))
plt.imshow(images[0].permute(1,2,0).numpy())
plt.show()