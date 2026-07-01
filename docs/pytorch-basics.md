Build the Neural Network
Created On: Feb 09, 2021 | Last Updated: Jan 24, 2025 | Last Verified: Not Verified

Neural networks comprise of layers/modules that perform operations on data. The torch.nn namespace provides all the building blocks you need to build your own neural network. Every module in PyTorch subclasses the nn.Module. A neural network is a module itself that consists of other modules (layers). This nested structure allows for building and managing complex architectures easily.

In the following sections, we’ll build a neural network to classify images in the FashionMNIST dataset.

import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
Get Device for Training
We want to be able to train our model on an accelerator such as CUDA, MPS, MTIA, or XPU. If the current accelerator is available, we will use it. Otherwise, we use the CPU.

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
print(f"Using {device} device")
Using cuda device
Define the Class
We define our neural network by subclassing nn.Module, and initialize the neural network layers in __init__. Every nn.Module subclass implements the operations on input data in the forward method.

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
We create an instance of NeuralNetwork, and move it to the device, and print its structure.

model = NeuralNetwork().to(device)
print(model)
NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)
To use the model, we pass it the input data. This executes the model’s forward, along with some background operations. Do not call model.forward() directly!

Calling the model on the input returns a 2-dimensional tensor with dim=0 corresponding to each output of 10 raw predicted values for each class, and dim=1 corresponding to the individual values of each output. We get the prediction probabilities by passing it through an instance of the nn.Softmax module.

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")
Predicted class: tensor([8], device='cuda:0')
Model Layers
Let’s break down the layers in the FashionMNIST model. To illustrate it, we will take a sample minibatch of 3 images of size 28x28 and see what happens to it as we pass it through the network.

input_image = torch.rand(3,28,28)
print(input_image.size())
torch.Size([3, 28, 28])
nn.Flatten
We initialize the nn.Flatten layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values ( the minibatch dimension (at dim=0) is maintained).

flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())
torch.Size([3, 784])
nn.Linear
The linear layer is a module that applies a linear transformation on the input using its stored weights and biases.

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())
torch.Size([3, 20])
nn.ReLU
Non-linear activations are what create the complex mappings between the model’s inputs and outputs. They are applied after linear transformations to introduce nonlinearity, helping neural networks learn a wide variety of phenomena.

In this model, we use nn.ReLU between our linear layers, but there’s other activations to introduce non-linearity in your model.

print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")
Before ReLU: tensor([[-0.5505, -0.7272, -0.1615,  0.4415, -0.0863, -0.1780, -0.1386, -0.2004,
         -0.4512,  0.6216, -0.0559, -0.3879,  0.0137,  0.2598,  0.0069, -0.3948,
         -0.2508,  0.2038,  0.3499,  0.0961],
        [-0.1887, -0.6773, -0.0020,  0.8442, -0.0948, -0.4265,  0.0204,  0.0560,
         -0.3261,  0.6697, -0.3558, -0.2869, -0.2903,  0.4202,  0.3272, -0.1288,
         -0.5280,  0.5138,  0.1161,  0.1258],
        [-0.7124, -0.5589, -0.2584,  0.3523, -0.0340, -0.6297, -0.2042,  0.1980,
         -0.1395,  0.7085, -0.0525, -0.1916, -0.1759,  0.1344,  0.1759, -0.3950,
         -0.5298,  0.3580, -0.0062, -0.0546]], grad_fn=<AddmmBackward0>)


After ReLU: tensor([[0.0000, 0.0000, 0.0000, 0.4415, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
         0.6216, 0.0000, 0.0000, 0.0137, 0.2598, 0.0069, 0.0000, 0.0000, 0.2038,
         0.3499, 0.0961],
        [0.0000, 0.0000, 0.0000, 0.8442, 0.0000, 0.0000, 0.0204, 0.0560, 0.0000,
         0.6697, 0.0000, 0.0000, 0.0000, 0.4202, 0.3272, 0.0000, 0.0000, 0.5138,
         0.1161, 0.1258],
        [0.0000, 0.0000, 0.0000, 0.3523, 0.0000, 0.0000, 0.0000, 0.1980, 0.0000,
         0.7085, 0.0000, 0.0000, 0.0000, 0.1344, 0.1759, 0.0000, 0.0000, 0.3580,
         0.0000, 0.0000]], grad_fn=<ReluBackward0>)
nn.Sequential
nn.Sequential is an ordered container of modules. The data is passed through all the modules in the same order as defined. You can use sequential containers to put together a quick network like seq_modules.

seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)
nn.Softmax
The last linear layer of the neural network returns logits - raw values in [-infty, infty] - which are passed to the nn.Softmax module. The logits are scaled to values [0, 1] representing the model’s predicted probabilities for each class. dim parameter indicates the dimension along which the values must sum to 1.

softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)
Model Parameters
Many layers inside a neural network are parameterized, i.e. have associated weights and biases that are optimized during training. Subclassing nn.Module automatically tracks all fields defined inside your model object, and makes all parameters accessible using your model’s parameters() or named_parameters() methods.

In this example, we iterate over each parameter, and print its size and a preview of its values.

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")
Model structure: NeuralNetwork(
  (flatten): Flatten(start_dim=1, end_dim=-1)
  (linear_relu_stack): Sequential(
    (0): Linear(in_features=784, out_features=512, bias=True)
    (1): ReLU()
    (2): Linear(in_features=512, out_features=512, bias=True)
    (3): ReLU()
    (4): Linear(in_features=512, out_features=10, bias=True)
  )
)


Layer: linear_relu_stack.0.weight | Size: torch.Size([512, 784]) | Values : tensor([[ 0.0266, -0.0344,  0.0286,  ...,  0.0198, -0.0012, -0.0254],
        [-0.0271, -0.0346, -0.0329,  ...,  0.0210,  0.0219,  0.0061]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.0.bias | Size: torch.Size([512]) | Values : tensor([-0.0054, -0.0200], device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.weight | Size: torch.Size([512, 512]) | Values : tensor([[-0.0125,  0.0032, -0.0037,  ..., -0.0385,  0.0132, -0.0152],
        [-0.0281, -0.0116, -0.0435,  ...,  0.0085,  0.0300, -0.0320]],
       device='cuda:0', grad_fn=<SliceBackward0>)

Layer: linear_relu_stack.2.bias | Size: torch.Size([512]) | Values : tensor([0.0210, 0.0409], device='cuda:0', grad_fn=<SliceBackward0>)