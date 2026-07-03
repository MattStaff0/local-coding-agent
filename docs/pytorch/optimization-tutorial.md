---
url: https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
fetched: 2026-07-02
---

Note

[Go to the end](#sphx-glr-download-beginner-basics-optimization-tutorial-py)
to download the full example code.

[Learn the Basics](intro.html) ||
[Quickstart](quickstart_tutorial.html) ||
[Tensors](tensorqs_tutorial.html) ||
[Datasets & DataLoaders](data_tutorial.html) ||
[Transforms](transforms_tutorial.html) ||
[Build Model](buildmodel_tutorial.html) ||
[Autograd](autogradqs_tutorial.html) ||
**Optimization** ||
[Save & Load Model](saveloadrun_tutorial.html)

# Optimizing Model Parameters

Created On: Feb 09, 2021 | Last Updated: May 07, 2026 | Last Verified: Nov 05, 2024

Now that we have a model and data it’s time to train, validate and test our model by optimizing its parameters on
our data. Training a model is an iterative process; in each iteration the model makes a guess about the output, calculates
the error in its guess (*loss*), collects the derivatives of the error with respect to its parameters (as we saw in
the [previous section](autogradqs_tutorial.html)), and **optimizes** these parameters using gradient descent. For a more
detailed walkthrough of this process, check out this video on [backpropagation from 3Blue1Brown](https://www.youtube.com/watch?v=tIeHLnjs5U8).

## Prerequisite Code

We load the code from the previous sections on [Datasets & DataLoaders](data_tutorial.html)
and [Build Model](buildmodel_tutorial.html).

```
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2

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

train_dataloader = DataLoader(training_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()
```

```
  0%|          | 0.00/26.4M [00:00<?, ?B/s]
  0%|          | 65.5k/26.4M [00:00<01:11, 369kB/s]
  1%|          | 229k/26.4M [00:00<00:37, 693kB/s]
  3%|▎         | 918k/26.4M [00:00<00:11, 2.15MB/s]
 14%|█▍        | 3.64M/26.4M [00:00<00:02, 8.19MB/s]
 26%|██▌       | 6.78M/26.4M [00:00<00:01, 13.0MB/s]
 47%|████▋     | 12.6M/26.4M [00:00<00:00, 22.6MB/s]
 60%|██████    | 16.0M/26.4M [00:01<00:00, 23.3MB/s]
 79%|███████▉  | 20.9M/26.4M [00:01<00:00, 27.8MB/s]
 95%|█████████▍| 25.1M/26.4M [00:01<00:00, 28.5MB/s]
100%|██████████| 26.4M/26.4M [00:01<00:00, 19.6MB/s]

  0%|          | 0.00/29.5k [00:00<?, ?B/s]
100%|██████████| 29.5k/29.5k [00:00<00:00, 337kB/s]

  0%|          | 0.00/4.42M [00:00<?, ?B/s]
  1%|▏         | 65.5k/4.42M [00:00<00:11, 374kB/s]
  4%|▍         | 197k/4.42M [00:00<00:07, 593kB/s]
 19%|█▉        | 852k/4.42M [00:00<00:01, 2.03MB/s]
 76%|███████▌  | 3.34M/4.42M [00:00<00:00, 6.83MB/s]
100%|██████████| 4.42M/4.42M [00:00<00:00, 6.28MB/s]

  0%|          | 0.00/5.15k [00:00<?, ?B/s]
100%|██████████| 5.15k/5.15k [00:00<00:00, 63.1MB/s]
```

## Hyperparameters

Hyperparameters are adjustable parameters that let you control the model optimization process.
Different hyperparameter values can impact model training and convergence rates
([read more](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html) about hyperparameter tuning)

We define the following hyperparameters for training:
:   * **Number of Epochs** - the number of times to iterate over the dataset
    * **Batch Size** - the number of data samples propagated through the network before the parameters are updated
    * **Learning Rate** - how much to update models parameters at each batch/epoch. Smaller values yield slow learning speed, while large values may result in unpredictable behavior during training.

```
learning_rate = 1e-3
batch_size = 64
epochs = 5
```

## Optimization Loop

Once we set our hyperparameters, we can then train and optimize our model with an optimization loop. Each
iteration of the optimization loop is called an **epoch**.

Each epoch consists of two main parts:
:   * **The Train Loop** - iterate over the training dataset and try to converge to optimal parameters.
    * **The Validation/Test Loop** - iterate over the test dataset to check if model performance is improving.

Let’s briefly familiarize ourselves with some of the concepts used in the training loop. Jump ahead to
see the [Full Implementation](#full-impl-label) of the optimization loop.

### Loss Function

When presented with some training data, our untrained network is likely not to give the correct
answer. **Loss function** measures the degree of dissimilarity of obtained result to the target value,
and it is the loss function that we want to minimize during training. To calculate the loss we make a
prediction using the inputs of our given data sample and compare it against the true data label value.

Common loss functions include [nn.MSELoss](https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html#torch.nn.MSELoss) (Mean Square Error) for regression tasks, and
[nn.NLLLoss](https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html#torch.nn.NLLLoss) (Negative Log Likelihood) for classification.
[nn.CrossEntropyLoss](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html#torch.nn.CrossEntropyLoss) combines `nn.LogSoftmax` and `nn.NLLLoss`.

We pass our model’s output logits to `nn.CrossEntropyLoss`, which will normalize the logits and compute the prediction error.

```
# Initialize the loss function
loss_fn = nn.CrossEntropyLoss()
```

### Optimizer

Optimization is the process of adjusting model parameters to reduce model error in each training step. **Optimization algorithms** define how this process is performed (in this example we use Stochastic Gradient Descent).
All optimization logic is encapsulated in the `optimizer` object. Here, we use the SGD optimizer; additionally, there are many [different optimizers](https://pytorch.org/docs/stable/optim.html)
available in PyTorch such as ADAM and RMSProp, that work better for different kinds of models and data.

We initialize the optimizer by registering the model’s parameters that need to be trained, and passing in the learning rate hyperparameter.

```
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```

Inside the training loop, optimization happens in three steps:
:   * Call `optimizer.zero_grad()` to reset the gradients of model parameters. Gradients by default add up; to prevent double-counting, we explicitly zero them at each iteration.
    * Backpropagate the prediction loss with a call to `loss.backward()`. PyTorch deposits the gradients of the loss w.r.t. each parameter.
    * Once we have our gradients, we call `optimizer.step()` to adjust the parameters by the gradients collected in the backward pass.

## Full Implementation

We define `train_loop` that loops over our optimization code, and `test_loop` that
evaluates the model’s performance against our test data.

```
def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
```

We initialize the loss function and optimizer, and pass it to `train_loop` and `test_loop`.
Feel free to increase the number of epochs to track the model’s improving performance.

```
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

epochs = 10
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")
```

```
Epoch 1
-------------------------------
loss: 2.315428  [   64/60000]
loss: 2.291904  [ 6464/60000]
loss: 2.281391  [12864/60000]
loss: 2.272638  [19264/60000]
loss: 2.242959  [25664/60000]
loss: 2.231485  [32064/60000]
loss: 2.230196  [38464/60000]
loss: 2.203037  [44864/60000]
loss: 2.201399  [51264/60000]
loss: 2.161943  [57664/60000]
Test Error:
 Accuracy: 44.4%, Avg loss: 2.154582

Epoch 2
-------------------------------
loss: 2.175072  [   64/60000]
loss: 2.153977  [ 6464/60000]
loss: 2.107854  [12864/60000]
loss: 2.116912  [19264/60000]
loss: 2.055709  [25664/60000]
loss: 2.017740  [32064/60000]
loss: 2.030155  [38464/60000]
loss: 1.962298  [44864/60000]
loss: 1.965580  [51264/60000]
loss: 1.882715  [57664/60000]
Test Error:
 Accuracy: 51.9%, Avg loss: 1.884362

Epoch 3
-------------------------------
loss: 1.928036  [   64/60000]
loss: 1.888328  [ 6464/60000]
loss: 1.784374  [12864/60000]
loss: 1.811178  [19264/60000]
loss: 1.694641  [25664/60000]
loss: 1.665603  [32064/60000]
loss: 1.668604  [38464/60000]
loss: 1.583889  [44864/60000]
loss: 1.604311  [51264/60000]
loss: 1.490338  [57664/60000]
Test Error:
 Accuracy: 59.3%, Avg loss: 1.515731

Epoch 4
-------------------------------
loss: 1.588369  [   64/60000]
loss: 1.549094  [ 6464/60000]
loss: 1.410087  [12864/60000]
loss: 1.469561  [19264/60000]
loss: 1.348885  [25664/60000]
loss: 1.355386  [32064/60000]
loss: 1.355367  [38464/60000]
loss: 1.291774  [44864/60000]
loss: 1.322926  [51264/60000]
loss: 1.219604  [57664/60000]
Test Error:
 Accuracy: 63.8%, Avg loss: 1.250584

Epoch 5
-------------------------------
loss: 1.329445  [   64/60000]
loss: 1.309708  [ 6464/60000]
loss: 1.151539  [12864/60000]
loss: 1.248222  [19264/60000]
loss: 1.122893  [25664/60000]
loss: 1.152281  [32064/60000]
loss: 1.163085  [38464/60000]
loss: 1.110159  [44864/60000]
loss: 1.145985  [51264/60000]
loss: 1.059882  [57664/60000]
Test Error:
 Accuracy: 65.3%, Avg loss: 1.085019

Epoch 6
-------------------------------
loss: 1.156160  [   64/60000]
loss: 1.158373  [ 6464/60000]
loss: 0.981780  [12864/60000]
loss: 1.109792  [19264/60000]
loss: 0.982584  [25664/60000]
loss: 1.015819  [32064/60000]
loss: 1.044260  [38464/60000]
loss: 0.994672  [44864/60000]
loss: 1.030202  [51264/60000]
loss: 0.959475  [57664/60000]
Test Error:
 Accuracy: 66.5%, Avg loss: 0.977957

Epoch 7
-------------------------------
loss: 1.035880  [   64/60000]
loss: 1.060211  [ 6464/60000]
loss: 0.866160  [12864/60000]
loss: 1.017686  [19264/60000]
loss: 0.894058  [25664/60000]
loss: 0.920709  [32064/60000]
loss: 0.966423  [38464/60000]
loss: 0.919621  [44864/60000]
loss: 0.950257  [51264/60000]
loss: 0.891869  [57664/60000]
Test Error:
 Accuracy: 67.8%, Avg loss: 0.905042

Epoch 8
-------------------------------
loss: 0.948257  [   64/60000]
loss: 0.992238  [ 6464/60000]
loss: 0.783880  [12864/60000]
loss: 0.952688  [19264/60000]
loss: 0.835064  [25664/60000]
loss: 0.851506  [32064/60000]
loss: 0.911388  [38464/60000]
loss: 0.869120  [44864/60000]
loss: 0.892699  [51264/60000]
loss: 0.842945  [57664/60000]
Test Error:
 Accuracy: 68.9%, Avg loss: 0.852478

Epoch 9
-------------------------------
loss: 0.880987  [   64/60000]
loss: 0.941406  [ 6464/60000]
loss: 0.722616  [12864/60000]
loss: 0.904412  [19264/60000]
loss: 0.793099  [25664/60000]
loss: 0.799766  [32064/60000]
loss: 0.869614  [38464/60000]
loss: 0.833595  [44864/60000]
loss: 0.849646  [51264/60000]
loss: 0.805401  [57664/60000]
Test Error:
 Accuracy: 70.3%, Avg loss: 0.812593

Epoch 10
-------------------------------
loss: 0.827004  [   64/60000]
loss: 0.900632  [ 6464/60000]
loss: 0.675203  [12864/60000]
loss: 0.867006  [19264/60000]
loss: 0.761228  [25664/60000]
loss: 0.760273  [32064/60000]
loss: 0.835847  [38464/60000]
loss: 0.807286  [44864/60000]
loss: 0.816242  [51264/60000]
loss: 0.775126  [57664/60000]
Test Error:
 Accuracy: 71.6%, Avg loss: 0.780901

Done!
```

## Further Reading

* [Loss Functions](https://pytorch.org/docs/stable/nn.html#loss-functions)
* [torch.optim](https://pytorch.org/docs/stable/optim.html)
* [Warmstart Training a Model](https://pytorch.org/tutorials/recipes/recipes/warmstarting_model_using_parameters_from_a_different_model.html)

**Total running time of the script:** (1 minutes 54.068 seconds)

[`Download Jupyter notebook: optimization_tutorial.ipynb`](../../_downloads/91d72708edab956d7293bb263e2ab53f/optimization_tutorial.ipynb)

[`Download Python source code: optimization_tutorial.py`](../../_downloads/0662a149d54bd776924742c96eb6282d/optimization_tutorial.py)

[`Download zipped: optimization_tutorial.zip`](../../_downloads/1667c8f6aca240f9985540ba73f6cd3d/optimization_tutorial.zip)
