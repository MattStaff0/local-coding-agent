---
url: https://docs.pytorch.org/tutorials/beginner/basics/data_tutorial.html
fetched: 2026-07-02
---

Note

[Go to the end](#sphx-glr-download-beginner-basics-data-tutorial-py)
to download the full example code.

[Learn the Basics](intro.html) ||
[Quickstart](quickstart_tutorial.html) ||
[Tensors](tensorqs_tutorial.html) ||
**Datasets & DataLoaders** ||
[Transforms](transforms_tutorial.html) ||
[Build Model](buildmodel_tutorial.html) ||
[Autograd](autogradqs_tutorial.html) ||
[Optimization](optimization_tutorial.html) ||
[Save & Load Model](saveloadrun_tutorial.html)

# Datasets & DataLoaders

Created On: Feb 09, 2021 | Last Updated: May 07, 2026 | Last Verified: Nov 05, 2024

Code for processing data samples can get messy and hard to maintain; we ideally want our dataset code
to be decoupled from our model training code for better readability and modularity.
PyTorch provides two data primitives: `torch.utils.data.DataLoader` and `torch.utils.data.Dataset`
that allow you to use pre-loaded datasets as well as your own data.
`Dataset` stores the samples and their corresponding labels, and `DataLoader` wraps an iterable around
the `Dataset` to enable easy access to the samples.

PyTorch domain libraries provide a number of pre-loaded datasets (such as FashionMNIST) that
subclass `torch.utils.data.Dataset` and implement functions specific to the particular data.
They can be used to prototype and benchmark your model. You can find them
here: [Image Datasets](https://pytorch.org/vision/stable/datasets.html),
[Text Datasets](https://pytorch.org/text/stable/datasets.html), and
[Audio Datasets](https://pytorch.org/audio/stable/datasets.html)

## Loading a Dataset

Here is an example of how to load the [Fashion-MNIST](https://research.zalando.com/project/fashion_mnist/fashion_mnist/) dataset from TorchVision.
Fashion-MNIST is a dataset of Zalando’s article images consisting of 60,000 training examples and 10,000 test examples.
Each example comprises a 28×28 grayscale image and an associated label from one of 10 classes.

We load the [FashionMNIST Dataset](https://pytorch.org/vision/stable/datasets.html#fashion-mnist) with the following parameters:
:   * `root` is the path where the train/test data is stored,
    * `train` specifies training or test dataset,
    * `download=True` downloads the data from the internet if it’s not available at `root`.
    * `transform` and `target_transform` specify the feature and label transformations

```
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import v2
import matplotlib.pyplot as plt

training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)

test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
)
```

```
  0%|          | 0.00/26.4M [00:00<?, ?B/s]
  0%|          | 65.5k/26.4M [00:00<01:10, 375kB/s]
  1%|          | 229k/26.4M [00:00<00:37, 704kB/s]
  3%|▎         | 918k/26.4M [00:00<00:11, 2.17MB/s]
 14%|█▍        | 3.67M/26.4M [00:00<00:03, 7.50MB/s]
 32%|███▏      | 8.32M/26.4M [00:00<00:01, 14.3MB/s]
 51%|█████     | 13.4M/26.4M [00:01<00:00, 19.2MB/s]
 71%|███████   | 18.7M/26.4M [00:01<00:00, 22.8MB/s]
 91%|█████████▏| 24.2M/26.4M [00:01<00:00, 25.5MB/s]
100%|██████████| 26.4M/26.4M [00:01<00:00, 18.7MB/s]

  0%|          | 0.00/29.5k [00:00<?, ?B/s]
100%|██████████| 29.5k/29.5k [00:00<00:00, 327kB/s]

  0%|          | 0.00/4.42M [00:00<?, ?B/s]
  1%|▏         | 65.5k/4.42M [00:00<00:11, 363kB/s]
  4%|▍         | 197k/4.42M [00:00<00:07, 579kB/s]
 19%|█▊        | 819k/4.42M [00:00<00:01, 1.89MB/s]
 74%|███████▍  | 3.28M/4.42M [00:00<00:00, 6.54MB/s]
100%|██████████| 4.42M/4.42M [00:00<00:00, 6.12MB/s]

  0%|          | 0.00/5.15k [00:00<?, ?B/s]
100%|██████████| 5.15k/5.15k [00:00<00:00, 63.3MB/s]
```

## Iterating and Visualizing the Dataset

We can index `Datasets` manually like a list: `training_data[index]`.
We use `matplotlib` to visualize some samples in our training data.

```
labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}
figure = plt.figure(figsize=(8, 8))
cols, rows = 3, 3
for i in range(1, cols * rows + 1):
    sample_idx = torch.randint(len(training_data), size=(1,)).item()
    img, label = training_data[sample_idx]
    figure.add_subplot(rows, cols, i)
    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

---

## Creating a Custom Dataset for your files

A custom Dataset class must implement three functions: \_\_init\_\_, \_\_len\_\_, and \_\_getitem\_\_.
Take a look at this implementation; the FashionMNIST images are stored
in a directory `img_dir`, and their labels are stored separately in a CSV file `annotations_file`.

In the next sections, we’ll break down what’s happening in each of these functions.

```
import os
import pandas as pd
from torchvision.io import decode_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = decode_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```

### `__init__`

The \_\_init\_\_ function is run once when instantiating the Dataset object. We initialize
the directory containing the images, the annotations file, and both transforms (covered
in more detail in the next section).

The labels.csv file looks like:

```
tshirt1.jpg, 0
tshirt2.jpg, 0
......
ankleboot999.jpg, 9
```

```
def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
    self.img_labels = pd.read_csv(annotations_file)
    self.img_dir = img_dir
    self.transform = transform
    self.target_transform = target_transform
```

### `__len__`

The \_\_len\_\_ function returns the number of samples in our dataset.

Example:

```
def __len__(self):
    return len(self.img_labels)
```

### `__getitem__`

The \_\_getitem\_\_ function loads and returns a sample from the dataset at the given index `idx`.
Based on the index, it identifies the image’s location on disk, converts that to a tensor using `decode_image`, retrieves the
corresponding label from the csv data in `self.img_labels`, calls the transform functions on them (if applicable), and returns the
tensor image and corresponding label in a tuple.

```
def __getitem__(self, idx):
    img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
    image = decode_image(img_path)
    label = self.img_labels.iloc[idx, 1]
    if self.transform:
        image = self.transform(image)
    if self.target_transform:
        label = self.target_transform(label)
    return image, label
```

---

## Preparing your data for training with DataLoaders

The `Dataset` retrieves our dataset’s features and labels one sample at a time. While training a model, we typically want to
pass samples in “minibatches”, reshuffle the data at every epoch to reduce model overfitting, and use Python’s `multiprocessing` to
speed up data retrieval.

`DataLoader` is an iterable that abstracts this complexity for us in an easy API.

```
from torch.utils.data import DataLoader

train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)
```

## Iterate through the DataLoader

We have loaded that dataset into the `DataLoader` and can iterate through the dataset as needed.
Each iteration below returns a batch of `train_features` and `train_labels` (containing `batch_size=64` features and labels respectively).
Because we specified `shuffle=True`, after we iterate over all batches the data is shuffled (for finer-grained control over
the data loading order, take a look at [Samplers](https://pytorch.org/docs/stable/data.html#data-loading-order-and-sampler)).

```
# Display image and label.
train_features, train_labels = next(iter(train_dataloader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[0].squeeze()
label = train_labels[0]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
```

```
Feature batch shape: torch.Size([64, 1, 28, 28])
Labels batch shape: torch.Size([64])
Label: 4
```

---

## Further Reading

* [torch.utils.data API](https://pytorch.org/docs/stable/data.html)

**Total running time of the script:** (0 minutes 4.995 seconds)

[`Download Jupyter notebook: data_tutorial.ipynb`](../../_downloads/36608d2d57f623ba3a623e0c947a8c3e/data_tutorial.ipynb)

[`Download Python source code: data_tutorial.py`](../../_downloads/56e3f440fc204e02856f8889c226d2d1/data_tutorial.py)

[`Download zipped: data_tutorial.zip`](../../_downloads/89855d8fec84a240291d4492f4ece548/data_tutorial.zip)
