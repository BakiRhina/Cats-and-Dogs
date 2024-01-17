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

## GPU Configuration for CUDA Support

### Checking GPU Availability

Before utilizing CUDA GPUs with PyTorch, it's crucial to ensure that your device supports CUDA and that it is available. Use the following commands to check for GPU availability:

```bash
import torch

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("CUDA Available:", cuda_available)

# Get the number of available GPUs
num_gpus = torch.cuda.device_count()
print("Number of GPUs:", num_gpus)
```

### Verifying CUDA Version

If CUDA is not available or you encounter compatibility issues, it's essential to check your CUDA version using the following command:

```bash
nvidia-smi
```

### Handling CUDA Version Compatibility Issues

1. **Check CUDA Version:**
   - Ensure that your CUDA version is compatible with PyTorch. Torch is compatible with CUDA versions under 11.8.

2. **Uninstall CUDA Version**
   - If your CUDA version is higher than 11.8, uninstall CUDA. In windows you can do that from "Add or remove programs" with ease. In Linux you can use:
```bash
sudo apt-get purge cuda-<your_version>
```

3. **Install CUDA 11.8:**
   - Install CUDA 11.8, as it is compatible with PyTorch. Refer to the NVIDIA website for installation instructions: [CUDA Toolkit Archive](https://developer.nvidia.com/cuda-11-8-0-download-archive)

### Installing PyTorch with CUDA Support

After ensuring CUDA compatibility, install PyTorch with CUDA support. CHAT GPT recommends following the official PyTorch installation guide, which provides detailed instructions tailored to your system configuration. You can find the guide at [PyTorch - Get Started Locally](https://pytorch.org/get-started/locally/).

By carefully managing CUDA compatibility and installing PyTorch correctly, you can leverage the power of GPU acceleration for your deep learning tasks.

# Notes and useful information

- Here I put some examples to delete rows in a pandas dataframe based on the columns' value:

```markdown
### Deleting Rows Based on Column Value in Pandas DataFrames

When working with pandas DataFrames in Python, it is often necessary to delete rows based on certain conditions in one or more columns. This can be achieved using various methods, and here are some quick examples:

#### Example 1: Using `drop()` to Delete Rows Based on Column Value

```python
df.drop(df[df['Fee'] >= 24000].index, inplace=True)
```

This method involves using the `drop()` function along with boolean indexing. Rows where the condition (`df['Fee'] >= 24000`) is met are identified using `df[df['Fee'] >= 24000].index` and then dropped from the original DataFrame.

#### Example 2: Remove Rows Based on Condition

```python
df2 = df[df.Fee >= 24000]
```

Here, a new DataFrame `df2` is created, containing only the rows where the condition (`df.Fee >= 24000`) is satisfied.

#### Example 3: Handling Column Names with Spaces

```python
df2 = df[df['column name']]
```

When the column name contains spaces, it should be enclosed in single quotes to avoid syntax errors.

#### Example 4: Using `loc` for Row Deletion

```python
df2 = df.loc[df["Fee"] >= 24000]
```

The `loc` method is utilized to select rows based on the specified condition (`df["Fee"] >= 24000`). This is similar to the second example but using `loc` explicitly.

#### Example 5: Deleting Rows Based on Multiple Column Values

```python
df2 = df[(df['Fee'] >= 22000) & (df['Discount'] == 2300)]
```

In this example, rows are selected based on the combination of conditions using the logical AND (`&`) operator.

#### Example 6: Drop Rows with None/NaN Values

```python
df2 = df[df.Discount.notnull()]
```

Rows containing None or NaN values in the 'Discount' column are dropped using the `notnull()` method.

