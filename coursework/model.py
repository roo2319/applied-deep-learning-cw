from torch import nn, flatten, max as tmax
from torch.nn import functional as F
from typing import NamedTuple

class MaxOut(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x):
        split = x.shape[1]//2
        a = x[:,:split]
        b = x[:,split:]
        return a.max(b)

class ShallowModel(nn.Module):
    def __init__(self):

        super().__init__()


        # layers
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=(5, 5),
            padding=2,
        )
        self.pool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.conv2 = nn.Conv2d(
            in_channels = 32,
            out_channels= 64,
            kernel_size = (3,3),
            padding = 1
        )
        self.pool2 = nn.MaxPool2d(kernel_size=(3,3), stride=2)
        self.conv3 = nn.Conv2d(
            in_channels = 64,
            out_channels= 128,
            kernel_size = (3,3),
            padding = 1
        )
        self.pool3 = nn.MaxPool2d(kernel_size=(3,3), stride=2)
        self.fc1 = nn.Linear(128*11*11, 48*48*2)
        self.maxout = MaxOut()
        self.fc2 = nn.Linear(48*48, 48*48)
        self.initialise_layer(self.fc1)
        self.initialise_layer(self.fc2)


        # init weights/biases 
    
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = flatten(x,start_dim=1) # ??????????????
        x = self.fc1(x)
        x = F.relu(self.maxout(x))
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

# class Discriminator(nn.Module):
#     def __init__(self):

#         super().__init__()

#         self.conv1_1 = nn.Conv2d(
#             in_channels=,
#             out_channels=,
#             kernel_size=(,),
#             padding=,
#         )
#         self.conv1_2 = nn.Conv2d(
#             in_channels=,
#             out_channels=,
#             kernel_size=(,),
#             padding=,
#         )
#         self.pool1 = nn.MaxPool2d(kernel_size=(, ), stride=)

#         self.conv2_1
#         self.conv2_2
#         self.pool2

#         self.conv3_1
#         self.conv3_2
#         self.pool3

#         self.fc1
#         self.fc2
#         self.fc3

#     def forward(x):
#         x=F.relu(self.conv1_1(x))
#         x=F.relu(self.conv1_2(x))
#         x=self.pool1(x)

#         x=F.relu(self.conv2_1(x))
#         x=F.relu(self.conv2_2(x))
#         x=self.pool2(x)

#         x=F.relu(self.conv3_1(x))
#         x=F.relu(self.conv3_2(x))
#         x=self.pool3(x)

#         x=F.tanh(self.fc1(x))
#         x=F.tanh(self.fc2(x))
#         x=F.sigmoid(self.fc3(x))
#         return x





