from torch import nn
from torch.nn import functional as F
from typing import NamedTuple

class MaxOut(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        split = x.shape[1]//2
        return x[:,:split]

class ImageShape(NamedTuple):
    height: int
    width: int
    channels: int

class ShallowModel(nn.Module):
    def __init__(self):

        super().__init__()
        self.input_shape = ImageShape(height=96, width=96, channels=3)

        # layers
        self.conv1 = nn.Conv2d(
            in_channels=self.input_shape.channels,
            out_channels=32,
            kernel_size=(5, 5),
            padding=(2, 2),
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels= 64,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        self.conv3 = nn.Conv2d(
            in_channels = 64,
            out_channels= 128,
            kernel_size = (3,3),
            padding = (1,1)
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(3,3), stride=(2,2))
        self.fc1 = nn.Linear(15488, 4608)
        self.maxout = MaxOut()
        self.fc2 = nn.Linear(2304, 2304)

        # init weights/biases 
        for layer in self.modules():
            self.initialise_layer(layer)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = self.pool3(x)
        x = flatten(x,start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.maxout(x)
        x = self.fc2(x)
        return x

# The weights in all layers are initialized from a normal
# Gaussian distribution with zero mean and a standard deviation of 0.01, with biases initialized to 0.1
    @staticmethod
    def initialise_layer(layer):
        if hasattr(layer, "bias"):
            layer.bias.data.fill_(0.1)
        if hasattr(layer, "weight"):
            layer.weight.data.normal_(0.0, 0.01)

