# Cats and Dogs Binary Classification

## Overview
This project focuses on binary label prediction, specifically distinguishing between cats and dogs using Convolutional Neural Networks (CNNs). The readme provides essential information and insights to understand and implement the neural network architecture, with a specific emphasis on handling the shape of the fully connected layer (`fc1`).

## Neural Network Architecture
The neural network architecture consists of convolutional layers followed by max-pooling layers, leading to fully connected layers for classification. The key components are as follows:

1. **Convolutional Layers:**
   - The first convolutional layer (`conv1`) has 3 input channels (RGB), 6 output channels, and a kernel size of 5.
   - The second convolutional layer (`conv2`) takes the 6 channels from the previous layer and produces 16 output channels with a kernel size of 5.

```python
conv1 = nn.Conv2d(3, 6, 5)
conv2 = nn.Conv2d(6, 16, 5)
```

2. **Max-Pooling Layers:**
   - Max-pooling layers with a kernel size of 2 and a stride of 2 are applied after each convolutional layer.

```python
pool = nn.MaxPool2d(2, 2)
```

3. **Fully Connected Layers:**
   - The fully connected layers (`fc1`) come after flattening the output of the last pooling layer.
   - The shape of `fc1` input is crucial and should be carefully considered based on the CNN architecture and image size.

```python
x = torch.flatten(x, 1)  # Flatten the output for fully connected layers
print(x.shape)  # Shape of the input to fc1
```

## Important Note on `fc1` Shape
The shape of the input to `fc1` is critical and depends on the CNN architecture and image size. To determine this shape, you can run convolutions and pooling on a test file while visualizing the shapes of the tensors. Here's an example:

```python
# Assuming 'images' is your input tensor with shape (batch_size, channels, height, width)
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
print(x.shape)  # This is the shape required for fc1 input
```

By inspecting the shapes during each step, you can ensure that the input to `fc1` aligns with your network architecture and image dimensions.

## Data Organization Tips

### Using DataLoader from Torch

When working with PyTorch's DataLoader, organizing your data in separate folders for cats and dogs can simplify the labeling process. By placing cat images in one folder and dog images in another, DataLoader can automatically assign labels (0 for cats and 1 for dogs). This approach enhances data handling and streamlines the training process.

```plaintext
dataset/
|-- cats/
|   |-- cat_image1.jpg
|   |-- cat_image2.jpg
|   |-- ...
|-- dogs/
|   |-- dog_image1.jpg
|   |-- dog_image2.jpg
|   |-- ...
```

### Handling Unsorted Data with Descriptive Names

If your data is not pre-sorted but has descriptive filenames (e.g., `dog_image1222.jpg`, `cat_image593.jpg`, `cat_image9128.jpg`), you can use string parsing techniques to extract labels.It is recommended using regular expressions or string manipulation to identify and assign labels based on the filenames.

For example, you can extract the label from a filename using Python's string operations:

```python
filename = "cat_image593.jpg"
label = 0 if "cat" in filename else 1 if "dog" in filename else None
```

Adjust the parsing logic based on the naming conventions in your dataset. This manual approach is effective when automatic labeling via folder structure is not feasible.


## Conclusion
Understanding and carefully handling the shape of the fully connected layer input (`fc1`) is essential for the successful implementation of a CNN for binary cat vs. dog classification. Regularly visualize tensor shapes during development to ensure proper alignment in your network architecture.